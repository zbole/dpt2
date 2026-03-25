import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import AutoModel
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree
from plyfile import PlyData

# ========================================================
# 1. 路径与全局配置
# ========================================================
dataset_root = Path("/lus/lfs1aip2/projects/b6ae/datasets/sensaturban").resolve()
RAW_DATA_ROOT = dataset_root / "raw"
# 🚀 你的全新 1024D 极净数据集目录
OUT_DATA_ROOT = dataset_root / "processed_1024D_Full" 

# 模型权重路径 (与脚本同目录的 weights 文件夹)
WEIGHT_PATH = Path("./weights").resolve()
DEVICE = "cuda"

# ========================================================
# 2. 纯 Numpy 几何先验提取
# ========================================================
def extract_geometric_features_vectorized(points, k=16):
    """提取基础的 PCA 几何特征 (用于 PTV3 3D 主干)"""
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k, workers=-1)
    neighbors = points[idx] 
    
    centered = neighbors - np.mean(neighbors, axis=1, keepdims=True)
    cov = np.matmul(centered.transpose(0, 2, 1), centered) / (k - 1)
    
    evals = np.linalg.eigvalsh(cov)
    evals = np.flip(evals, axis=1) 
    evals = np.maximum(evals, 1e-8)
    
    sum_vals = np.sum(evals, axis=1, keepdims=True) + 1e-8
    l1, l2, l3 = evals[:, 0:1], evals[:, 1:2], evals[:, 2:3]
    
    linearity = (l1 - l2) / sum_vals
    planarity = (l2 - l3) / sum_vals
    scattering = l3 / sum_vals
    
    return np.concatenate([linearity, planarity, scattering], axis=1).astype(np.float32)

# ========================================================
# 3. DINOv3 提取器 (彻底修复 Patch Size 与动态张量截取)
# ========================================================
class DINOv3FeatureExtractor:
    def __init__(self, model_path_or_name: str, device: str = "cuda"):
        self.device = torch.device(device)
        
        print(f"🚀 Loading ViT model from: {model_path_or_name}")
        self.model = AutoModel.from_pretrained(
            model_path_or_name,
            local_files_only=True,
            device_map=self.device
        )
        self.model.eval()

        # 🚀 动态读取模型真实的 patch_size，彻底杜绝硬编码冲突！
        self.patch_size = self.model.config.patch_size 
        print(f"✅ Detected model patch_size: {self.patch_size}")

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False), 
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def point_cloud_to_bev_topz(self, coords: torch.Tensor, rgb: torch.Tensor, resolution: float):
        current_device = coords.device
        coords_min = coords.min(dim=0, keepdim=True)[0]
        coords_shifted = coords - coords_min

        grid_xy = torch.floor(coords_shifted[:, :2] / resolution).long()
        W = grid_xy[:, 0].max().item() + 1
        H = grid_xy[:, 1].max().item() + 1
        point_to_pixel_idx = grid_xy[:, 1] * W + grid_xy[:, 0]

        z_vals = coords[:, 2]
        sorted_indices = torch.argsort(z_vals, descending=True)
        point_to_pixel_sorted = point_to_pixel_idx[sorted_indices]

        unique_pixels, inverse_indices = torch.unique(point_to_pixel_sorted, return_inverse=True)
        first_occurrence_idx = torch.zeros_like(unique_pixels)
        seq = torch.arange(point_to_pixel_sorted.size(0), device=current_device)
        first_occurrence_idx.scatter_reduce_(0, inverse_indices, seq, reduce="amin", include_self=False)

        valid_point_idx = sorted_indices[first_occurrence_idx]
        bev_features_flat = torch.zeros((H * W, 3), dtype=rgb.dtype, device=current_device)
        bev_features_flat[unique_pixels] = rgb[valid_point_idx]
        bev_image = bev_features_flat.view(H, W, 3).permute(2, 0, 1)

        return bev_image, point_to_pixel_idx, (H, W)

    @torch.inference_mode()
    def extract_and_lift_features(self, coords: torch.Tensor, rgb: torch.Tensor, resolution: float = 0.5):
        bev_image, point_to_pixel_idx, (H, W) = self.point_cloud_to_bev_topz(coords, rgb, resolution)

        pad_H = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_W = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_H > 0 or pad_W > 0:
            bev_image = F.pad(bev_image, (0, pad_W, 0, pad_H))

        padded_H, padded_W = bev_image.shape[1], bev_image.shape[2]
        input_tensor = self.transform(bev_image).unsqueeze(0).to(self.device)

        grid_H, grid_W = input_tensor.shape[2] // self.patch_size, input_tensor.shape[3] // self.patch_size

        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model(pixel_values=input_tensor, interpolate_pos_encoding=True)

        # 🚀 动态截取正确的 Patch tokens (无视前面有几个特殊 Token)
        num_patches = grid_H * grid_W
        patch_tokens = outputs.last_hidden_state[:, -num_patches:, :]
        Hidden_Dim = patch_tokens.shape[-1]

        # 现在 reshape 绝对安全，不会再报错 size invalid
        feature_map_2d = patch_tokens.reshape(1, grid_H, grid_W, Hidden_Dim).permute(0, 3, 1, 2)
        feature_map_upsampled = F.interpolate(feature_map_2d.float(), size=(padded_H, padded_W), mode='bilinear', align_corners=False)
        feature_map_cropped = feature_map_upsampled[0, :, :H, :W]
        flat_features = feature_map_cropped.permute(1, 2, 0).reshape(H * W, Hidden_Dim)

        flat_features_cpu = flat_features.cpu().half()
        point_level_dino_features = flat_features_cpu[point_to_pixel_idx]

        return point_level_dino_features

# ========================================================
# 4. 全量端到端处理流水线 (从 Raw 到 1024D Chunk)
# ========================================================
def process_scene(scene_path, extractor):
    scene_path = Path(scene_path)
    scene_name = scene_path.stem.lower()
    
    # 划分数据集 (Birmingham & Cambridge)
    if "train" in str(scene_path):
        VAL_LIST = ["birmingham_block_1", "birmingham_block_6", "cambridge_block_12", "cambridge_block_6"]
        split = "val" if any(b in scene_name for b in VAL_LIST) else "train"
    else:
        split = "test"
        
    save_dir = OUT_DATA_ROOT / split
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Loading RAW file: {scene_name} ...")
    try:
        with open(str(scene_path), 'rb') as f:
            v = PlyData.read(f)['vertex']
        pts = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
        rgb = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.uint8)
        sem = v['class'].astype(np.int16) if 'class' in v else np.full(len(pts), 255)
    except Exception as e: 
        print(f"❌ Load Error: {e}")
        return

    # 1. 基础分辨率降采样 (0.1m, 用于保留建筑细节和 3D 主干输入)
    pts -= pts.min(axis=0)
    grid_coord = np.floor(pts / 0.1).astype(int)
    _, idx, counts = np.unique(grid_coord, axis=0, return_index=True, return_counts=True)
    pts, rgb, sem = pts[idx], rgb[idx], sem[idx]
    
    # 2. 计算 5D 几何先验特征 (在全局尺度计算，避免切块边缘截断效应)
    print("📐 Computing global 5D Geometric Features...")
    density = (counts.astype(np.float32).reshape(-1, 1) / counts.max()).astype(np.float32)
    eigen = extract_geometric_features_vectorized(pts)
    
    # 🚀 补充：提取 Z 坐标作为全局高度先验 (Global_Rel_Z)
    global_rel_z = pts[:, 2:3].astype(np.float32) 
    
    # 拼接为 5D: [Linearity, Planarity, Scattering, Density, Global_Rel_Z]
    geo_prior = np.concatenate([eigen, density, global_rel_z], axis=1).astype(np.float32)
    
    # 3. 空间切块并提取 DINO (50m x 50m)
    x_max, y_max = pts.max(axis=0)[:2]
    chunk_count = 0
    
    print("✂️ Chunking & Extracting DINOv3 1024D...")
    for x in np.arange(0, x_max, 25):
        for y in np.arange(0, y_max, 25):
            mask = (pts[:, 0] >= x) & (pts[:, 0] < x + 50) & (pts[:, 1] >= y) & (pts[:, 1] < y + 50)
            if np.sum(mask) < 1024: continue 
            
            chunk_folder = save_dir / f"{scene_name}_{chunk_count}"
            
            # 断点续传保护
            if (chunk_folder / "dino_1024d.npy").exists():
                chunk_count += 1
                continue
                
            chunk_folder.mkdir(exist_ok=True)
            
            pts_chunk = pts[mask]
            rgb_chunk = rgb[mask]
            geo_chunk = geo_prior[mask]
            sem_chunk = sem[mask]
            
            # 🚀 纯 Numpy 模拟 Open3D voxel_down_sample(0.5m) 用于 DINO 提特征
            grid_coord_05 = np.floor(pts_chunk / 0.5).astype(int)
            _, idx_05 = np.unique(grid_coord_05, axis=0, return_index=True)
            down_coords_np = pts_chunk[idx_05]
            down_colors_np = rgb_chunk[idx_05].astype(np.float32) / 255.0 # DINO 需要 0-1 范围
            
            down_coords = torch.tensor(down_coords_np, dtype=torch.float32)
            down_colors = torch.tensor(down_colors_np, dtype=torch.float32)
            
            # 🚀 极速提取 1024D DINO 先验
            down_features = extractor.extract_and_lift_features(down_coords, down_colors, resolution=0.5)
            
            # 🚀 映射回 0.1m 分辨率的点云
            kdtree = cKDTree(down_coords_np)
            _, indices = kdtree.query(pts_chunk, k=1, workers=-1)
            full_res_dino = down_features[indices].numpy()
            
            # 保存到全新目录！极其干净的数据结构
            np.save(chunk_folder / "coord.npy", pts_chunk)
            np.save(chunk_folder / "color.npy", rgb_chunk)
            np.save(chunk_folder / "segment.npy", sem_chunk)
            np.save(chunk_folder / "extra_feat.npy", extra_feat) # 🚀 替代了原来的 geo 和 dino 散件
            
            chunk_count += 1

    print(f"✅ {scene_name} processed: {chunk_count} chunks generated.")

if __name__ == "__main__":
    extractor = DINOv3FeatureExtractor(model_path_or_name=str(WEIGHT_PATH))
    ply_files = sorted(list(RAW_DATA_ROOT.glob("**/*.ply")))
    
    print(f"\n🚀 Found {len(ply_files)} RAW PLY files. Starting End-to-End Pipeline...")
    for f in ply_files:
        process_scene(f, extractor)
        
    print(f"\n🎉 All done! New compact dataset saved to: {OUT_DATA_ROOT}")