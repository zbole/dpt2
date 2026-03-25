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
RAW_TEST_ROOT = dataset_root / "raw" / "test"  # 🚀 精准锁定 test 目录
# 🚀 直接输出到你的最终 1035D 目录！
OUT_DATA_ROOT = dataset_root / "processed_1035D_with_relativeZ" / "test"

# 模型权重路径
WEIGHT_PATH = Path("../../../../DINO/weights").resolve()
DEVICE = "cuda"

# ========================================================
# 2. 特征提取组件
# ========================================================
def extract_geometric_features_vectorized(points, k=16):
    """提取基础的 PCA 几何特征 (Linearity, Planarity, Scattering)"""
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

class DINOv3FeatureExtractor:
    def __init__(self, model_path_or_name: str, device: str = "cuda"):
        self.device = torch.device(device)
        print(f"🚀 Loading ViT model from: {model_path_or_name}")
        self.model = AutoModel.from_pretrained(
            model_path_or_name, local_files_only=True, device_map=self.device
        )
        self.model.eval()
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

        num_patches = grid_H * grid_W
        patch_tokens = outputs.last_hidden_state[:, -num_patches:, :]
        Hidden_Dim = patch_tokens.shape[-1]

        feature_map_2d = patch_tokens.reshape(1, grid_H, grid_W, Hidden_Dim).permute(0, 3, 1, 2)
        feature_map_upsampled = F.interpolate(feature_map_2d.float(), size=(padded_H, padded_W), mode='bilinear', align_corners=False)
        feature_map_cropped = feature_map_upsampled[0, :, :H, :W]
        flat_features = feature_map_cropped.permute(1, 2, 0).reshape(H * W, Hidden_Dim)

        flat_features_cpu = flat_features.cpu().half()
        return flat_features_cpu[point_to_pixel_idx]

# ========================================================
# 3. 核心处理逻辑：读取 -> 计算 Z_floor -> 分块 -> DINO -> 保存
# ========================================================
def process_test_scene(scene_path, extractor):
    scene_path = Path(scene_path)
    scene_name = scene_path.stem.lower()
    OUT_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 [TEST SET] Loading RAW file: {scene_name} ...")
    try:
        with open(str(scene_path), 'rb') as f:
            v = PlyData.read(f)['vertex']
        
        pts_absolute = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
        rgb = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.uint8)
        
        # 🚀 捍卫真实标签！如果有 class 就读 class，绝不瞎写 255
        if 'class' in v:
            sem = v['class'].astype(np.int16)
            print(f"🎯 检测到真实标签！(有效类别: {np.unique(sem)})")
        else:
            sem = np.full(len(pts_absolute), 255, dtype=np.int16)
            print("⚠️ 未检测到真实标签，填充 255。")
            
    except Exception as e: 
        print(f"❌ Load Error: {e}")
        return

    # 🚀 步骤 A: 计算宏观底盘高度 (Z-Floor)
    z_floor = float(np.percentile(pts_absolute[:, 2], 0.1))
    print(f"🌍 宏观底盘 Z-Floor 计算完成: {z_floor:.2f}")

    # 🚀 步骤 B: 0.1m 降采样
    pts_min = pts_absolute.min(axis=0)
    pts_shifted = pts_absolute - pts_min # 用于网格划分的相对坐标
    
    grid_coord = np.floor(pts_shifted / 0.1).astype(int)
    _, idx, counts = np.unique(grid_coord, axis=0, return_index=True, return_counts=True)
    
    pts_shifted = pts_shifted[idx]
    pts_absolute = pts_absolute[idx]
    rgb = rgb[idx]
    sem = sem[idx]
    
    # 🚀 步骤 C: 计算全局 4D 几何特征与 1D 相对高程
    print("📐 Computing global Geometric Features & Relative Z...")
    density = (counts.astype(np.float32).reshape(-1, 1) / counts.max()).astype(np.float32)
    eigen = extract_geometric_features_vectorized(pts_shifted) # 3D
    geo_4d = np.concatenate([eigen, density], axis=1).astype(np.float32) # [Linearity, Planarity, Scattering, Density]
    
    global_rel_z = (pts_absolute[:, 2:3] - z_floor).astype(np.float32) # 1D 真实相对高程

    # 🚀 步骤 D: 空间切块并提取 DINO (50m x 50m)
    x_max, y_max = pts_shifted.max(axis=0)[:2]
    chunk_count = 0
    
    print("✂️ Chunking & Extracting DINOv3 1024D...")
    for x in np.arange(0, x_max, 25):
        for y in np.arange(0, y_max, 25):
            mask = (pts_shifted[:, 0] >= x) & (pts_shifted[:, 0] < x + 50) & \
                   (pts_shifted[:, 1] >= y) & (pts_shifted[:, 1] < y + 50)
            
            if np.sum(mask) < 1024: continue 
            
            chunk_folder = OUT_DATA_ROOT / f"{scene_name}_{chunk_count}"
            if (chunk_folder / "extra_feat.npy").exists():
                chunk_count += 1
                continue
                
            chunk_folder.mkdir(exist_ok=True)
            
            # 提取 Chunk 数据
            pts_chunk = pts_shifted[mask]  # Pointcept 默认保存 shifted 坐标
            rgb_chunk = rgb[mask]
            sem_chunk = sem[mask]
            geo_4d_chunk = geo_4d[mask]
            rel_z_chunk = global_rel_z[mask]
            
            # 提取 DINO
            grid_coord_05 = np.floor(pts_chunk / 0.5).astype(int)
            _, idx_05 = np.unique(grid_coord_05, axis=0, return_index=True)
            down_coords_np = pts_chunk[idx_05]
            down_colors_np = rgb_chunk[idx_05].astype(np.float32) / 255.0 
            
            down_coords = torch.tensor(down_coords_np, dtype=torch.float32)
            down_colors = torch.tensor(down_colors_np, dtype=torch.float32)
            
            down_features = extractor.extract_and_lift_features(down_coords, down_colors, resolution=0.5)
            
            kdtree = cKDTree(down_coords_np)
            _, indices = kdtree.query(pts_chunk, k=1, workers=-1)
            dino_1024_chunk = down_features[indices].numpy()
            
            # 🚀 步骤 E: 终极大一统！生成 1029 维 extra_feat
            # 结构: [4D 形状, 1D 高度, 1024D DINO]
            extra_feat_chunk = np.concatenate([geo_4d_chunk, rel_z_chunk, dino_1024_chunk], axis=1).astype(np.float32)
            
            # 保存到最终目录
            np.save(chunk_folder / "coord.npy", pts_chunk)
            np.save(chunk_folder / "color.npy", rgb_chunk)
            np.save(chunk_folder / "segment.npy", sem_chunk) # 完美的真值！
            np.save(chunk_folder / "extra_feat.npy", extra_feat_chunk) 
            
            chunk_count += 1

    print(f"✅ {scene_name} processed: {chunk_count} chunks generated.")

if __name__ == "__main__":
    extractor = DINOv3FeatureExtractor(model_path_or_name=str(WEIGHT_PATH))
    ply_files = sorted(list(RAW_TEST_ROOT.glob("**/*.ply")))
    
    if not ply_files:
        print(f"❌ 找不到文件！请检查路径: {RAW_TEST_ROOT}")
    else:
        print(f"\n🚀 Found {len(ply_files)} RAW PLY files in TEST set. Starting End-to-End Pipeline...")
        for f in ply_files:
            process_test_scene(f, extractor)
            
        print(f"\n🎉 Test Set All done! New compact dataset saved to: {OUT_DATA_ROOT}")