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
RAW_DATA_ROOT = dataset_root / "special"
# 🚀 你的全新轻量化、统一 1025D 数据集目录
OUT_DATA_ROOT = dataset_root / "special_out"

# 模型权重路径 (与脚本同目录的 weights 文件夹)
WEIGHT_PATH = Path("./weights").resolve()
DEVICE = "cuda"

# ========================================================
# 2. DINOv3 提取器 (保持不变)
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
        point_level_dino_features = flat_features_cpu[point_to_pixel_idx]

        return point_level_dino_features

# ========================================================
# 3. 全量端到端处理流水线 (兼容 .ply 和 .bin)
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

    print(f"\n📂 Loading RAW file: {scene_path.name} -> saving to [{split}] ...")
    try:
        # 🚀 新增：文件格式判断与加载逻辑
        if scene_path.suffix.lower() == '.ply':
            with open(str(scene_path), 'rb') as f:
                v = PlyData.read(f)['vertex']
            pts = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
            rgb = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.uint8)
            sem = v['class'].astype(np.int16) if 'class' in v else np.full(len(pts), 255)
            
        elif scene_path.suffix.lower() == '.bin':
            # 假设是 Raw Binary 格式: [X, Y, Z, R, G, B] (float32)
            raw_data = np.fromfile(scene_path, dtype=np.float32)
            
            # 推断通道数 (通常是 6 通道 XYZRGB)
            if len(raw_data) % 6 == 0:
                raw_data = raw_data.reshape(-1, 6)
                pts = raw_data[:, 0:3]
                # 某些 bin 文件的 RGB 是 0-1 之间的 float，这里统一转为 0-255 uint8
                if raw_data[:, 3:6].max() <= 1.0:
                    rgb = (raw_data[:, 3:6] * 255).astype(np.uint8)
                else:
                    rgb = raw_data[:, 3:6].astype(np.uint8)
                sem = np.full(len(pts), 255)
            else:
                raise ValueError(f"❌ Binary data shape mismatch! Total elements {len(raw_data)} cannot be divided by 6.")
        else:
            print(f"❌ Unsupported format: {scene_path.suffix}")
            return
            
    except Exception as e: 
        print(f"❌ Load Error: {e}")
        # 如果走到这里报错，99% 的概率是因为这个 bin 是 CloudCompare 的专属工程文件！
        print("💡 Hint: If this .bin file was saved directly from CloudCompare, it cannot be read! Please export it as a .PLY file instead.")
        return

    print(f"📊 Loaded {len(pts)} points successfully.")

    # 1. 基础分辨率降采样 (0.1m)
    pts -= pts.min(axis=0)
    grid_coord = np.floor(pts / 0.1).astype(int)
    _, idx = np.unique(grid_coord, axis=0, return_index=True)
    pts, rgb, sem = pts[idx], rgb[idx], sem[idx]
    
    # 2. 计算全局高度归一化先验 (统一的 1D 物理特征)
    print("📐 Computing Physical Priors (Rel_Z)...")
    z_floor = np.percentile(pts[:, 2], 0.1)
    global_rel_z = (pts[:, 2:3] - z_floor).astype(np.float32)
    global_rel_z = np.clip(global_rel_z / 100.0, 0.0, 1.0) # 归一化防止梯度爆炸
    
    # 3. 空间切块并提取 DINO (50m x 50m)
    x_max, y_max = pts.max(axis=0)[:2]
    chunk_count = 0
    
    # 动态步长: 训练集重叠增强(25m)，验证集/测试集无缝拼接(50m)避免测试泄露
    stride = 25 if split == "train" else 50
    
    print(f"✂️ Chunking (Stride: {stride}m) & Extracting DINOv3...")
    for x in np.arange(0, x_max, stride):
        for y in np.arange(0, y_max, stride):
            mask = (pts[:, 0] >= x) & (pts[:, 0] < x + 50) & (pts[:, 1] >= y) & (pts[:, 1] < y + 50)
            if np.sum(mask) < 1024: continue 
            
            chunk_folder = save_dir / f"{scene_name}_{chunk_count}"
            
            # 断点续传保护
            if (chunk_folder / "extra_feat.npy").exists():
                chunk_count += 1
                continue
                
            chunk_folder.mkdir(exist_ok=True)
            
            pts_chunk = pts[mask]
            rgb_chunk = rgb[mask]
            sem_chunk = sem[mask]
            rel_z_chunk = global_rel_z[mask]
            
            # 模拟 voxel_down_sample(0.5m) 用于 DINO 提特征
            grid_coord_05 = np.floor(pts_chunk / 0.5).astype(int)
            _, idx_05 = np.unique(grid_coord_05, axis=0, return_index=True)
            down_coords_np = pts_chunk[idx_05]
            down_colors_np = rgb_chunk[idx_05].astype(np.float32) / 255.0 
            
            down_coords = torch.tensor(down_coords_np, dtype=torch.float32)
            down_colors = torch.tensor(down_colors_np, dtype=torch.float32)
            
            # 极速提取 1024D DINO 先验
            down_features = extractor.extract_and_lift_features(down_coords, down_colors, resolution=0.5)
            
            # 映射回 0.1m 分辨率的点云
            kdtree = cKDTree(down_coords_np)
            _, indices = kdtree.query(pts_chunk, k=1, workers=-1)
            full_res_dino = down_features[indices].numpy()
            
            # 🚀 统一输出结构：[Rel_Z (1), DINO (1024)] = 1025D
            extra_feat = np.concatenate([rel_z_chunk, full_res_dino], axis=1).astype(np.float32)
            
            # 保存数据
            np.save(chunk_folder / "coord.npy", pts_chunk)
            np.save(chunk_folder / "color.npy", rgb_chunk)
            np.save(chunk_folder / "segment.npy", sem_chunk)
            np.save(chunk_folder / "extra_feat.npy", extra_feat) 
            
            chunk_count += 1

    print(f"✅ {scene_name} processed: {chunk_count} chunks generated.")

# ========================================================
# 4. 主函数启动器
# ========================================================
if __name__ == "__main__":
    extractor = DINOv3FeatureExtractor(model_path_or_name=str(WEIGHT_PATH))
    
    # 🚀 新增：同时搜索文件夹下的 .ply 和 .bin 文件
    data_files = list(RAW_DATA_ROOT.glob("**/*.ply")) + list(RAW_DATA_ROOT.glob("**/*.bin"))
    data_files = sorted(data_files)
    
    print(f"\n🚀 Found {len(data_files)} RAW files (PLY/BIN). Starting End-to-End Pipeline...")
    for f in data_files:
        process_scene(f, extractor)
        
    print(f"\n🎉 All done! New compact dataset saved to: {OUT_DATA_ROOT}")