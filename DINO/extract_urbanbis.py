import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchvision.transforms import v2
from transformers import AutoModel
from pathlib import Path
from scipy.spatial import cKDTree

# ========================================================
# 1. 路径与全局配置 (针对 UrbanBIS 修改)
# ========================================================
# 🚀 替换为 UrbanBIS 的真实路径
dataset_root = Path("/lus/lfs1aip2/projects/b6ae/datasets/UrbanBIS").resolve()
# 针对你截图中的目录结构
RAW_DATA_ROOT = dataset_root / "raw" / "Lihu"
# 最终保存的目录
OUT_DATA_ROOT = dataset_root / "processed_1025D_Pure" 

WEIGHT_PATH = Path("./weights").resolve()
DEVICE = "cuda"

# ========================================================
# 2. DINOv3 提取器 (必须包含这个类！)
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
# 3. UrbanBIS 专属场景处理函数
# ========================================================
def process_urbanbis_scene(txt_path, extractor):
    txt_path = Path(txt_path)
    scene_name = txt_path.stem
    # 自动识别当前是在 train 还是 test 文件夹下
    split = txt_path.parent.name 
    
    save_dir = OUT_DATA_ROOT / split
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Loading TXT file: [{split}] {scene_name} ...")
    
    try:
        # 使用 pandas 极速读取 txt (跳过任何可能的脏数据行)
        df = pd.read_csv(str(txt_path), sep='\s+', header=None, 
                         names=['x', 'y', 'z', 'r', 'g', 'b', 'sem', 'ins1', 'ins2'],
                         on_bad_lines='skip')
        
        pts = df[['x', 'y', 'z']].values.astype(np.float32)
        rgb = df[['r', 'g', 'b']].values.astype(np.uint8)
        sem = df['sem'].values.astype(np.int16)
        
        # 🚀 极其关键：将 UrbanBIS 特有的 -100 (无标记) 转换为 255 (PyTorch 忽略类)
        sem[sem == -100] = 255
        
        # 🚀 检查：打印真实的标签分布，确保 0-6 的有效类都在，且无标记被正确转为 255
        unique_labels = np.unique(sem)
        print(f"📊 [Debug] 映射后真实存在的标签包含: {unique_labels}")
        
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return

    # 1. 基础分辨率降采样 (0.1m)
    pts -= pts.min(axis=0)
    grid_coord = np.floor(pts / 0.1).astype(int)
    _, idx = np.unique(grid_coord, axis=0, return_index=True)
    pts, rgb, sem = pts[idx], rgb[idx], sem[idx]
    
    # 2. 计算纯净的 1D 相对高程先验
    print("📏 Computing robust Global Relative Z...")
    z_floor = np.percentile(pts[:, 2], 0.1)
    global_rel_z = (pts[:, 2:3] - z_floor).astype(np.float32)
    
    # 🚀 修复 1：高程归一化！防止几百米的绝对高程梯度爆炸
    global_rel_z = np.clip(global_rel_z / 100.0, 0.0, 1.0)
    
    # 3. 空间切块并提取 DINO 
    x_max, y_max = pts.max(axis=0)[:2]
    chunk_count = 0
    
    # 🚀 修复 2：动态步长。Train 重叠增强 (25m)，Test/Val 无缝拼接 (50m) 防止泄露和重复计算
    stride = 25 if split == "train" else 50
    
    print(f"✂️ Chunking (Stride: {stride}m) & Extracting DINOv3 1024D...")
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
            
            # 提取 DINO 先验
            down_features = extractor.extract_and_lift_features(down_coords, down_colors, resolution=0.5)
            
            # 映射回 0.1m 分辨率
            kdtree = cKDTree(down_coords_np)
            _, indices = kdtree.query(pts_chunk, k=1, workers=-1)
            full_res_dino = down_features[indices].numpy()
            
            # 🚀 组合终极 1025D Extra Feature [Rel_Z, DINO]
            extra_feat = np.concatenate([rel_z_chunk, full_res_dino], axis=1).astype(np.float32)
            
            # 保存数据
            np.save(chunk_folder / "coord.npy", pts_chunk)
            np.save(chunk_folder / "color.npy", rgb_chunk)
            np.save(chunk_folder / "segment.npy", sem_chunk)
            np.save(chunk_folder / "extra_feat.npy", extra_feat)
            
            chunk_count += 1

    print(f"✅ {scene_name} processed: {chunk_count} chunks generated.")

# ========================================================
# 4. 主程序入口
# ========================================================
if __name__ == "__main__":
    # 请确保你的环境里安装了 pandas: pip install pandas
    
    extractor = DINOv3FeatureExtractor(model_path_or_name=str(WEIGHT_PATH))
    
    # 查找所有 .txt 文件
    txt_files = sorted(list(RAW_DATA_ROOT.glob("**/*.txt")))
    
    print(f"\n🚀 Found {len(txt_files)} RAW TXT files in UrbanBIS. Starting Fast Pipeline...")
    for f in txt_files:
        process_urbanbis_scene(f, extractor)
        
    print(f"\n🎉 All done! New compact dataset saved to: {OUT_DATA_ROOT}")