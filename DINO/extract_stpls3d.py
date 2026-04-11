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
dataset_root = Path("/lus/lfs1aip2/projects/b6ae/datasets/STPLS3D").resolve()
RAW_DATA_ROOT = dataset_root / "raw" / "Synthetic_v3"
# 🚀 输出目录：包含了完整的 train 和 val
OUT_DATA_ROOT = dataset_root / "processed_1025D_Synthetic_Eval" 

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
# 3. 专属场景处理函数 (自动分发到 train 或 val)
# ========================================================
def process_scene(scene_path, extractor, split):
    scene_path = Path(scene_path)
    scene_name = scene_path.stem
    
    save_dir = OUT_DATA_ROOT / split
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Loading RAW file: {scene_name} -> saving to [{split}] folder ...")
    try:
        with open(str(scene_path), 'rb') as f:
            v = PlyData.read(f)['vertex']
        pts = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
        rgb = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.uint8)
        sem = v['class'].astype(np.int16) if 'class' in v else np.full(len(pts), 255)
        
        # 🚀 打印原始标签，确认数据读取正常
        unique_labels_raw = np.unique(sem)
        print(f"📊 [Debug] 原始标签包含: {unique_labels_raw}")
        
        # 🚀 终极修复：将离散的 18 个有效类别映射到连续的 0~17，其余全标为 255(忽略)
        valid_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]
        mapping = np.full(256, 255, dtype=np.int16)
        for new_id, orig_id in enumerate(valid_labels):
            mapping[orig_id] = new_id
            
        # 执行映射并打印检查
        sem = mapping[sem]
        print(f"✅ [Debug] 映射后标签包含: {np.unique(sem)}")
            
    except Exception as e: 
        print(f"❌ Load Error: {e}")
        return

    # 1. 基础分辨率降采样 (0.1m)
    pts -= pts.min(axis=0)
    grid_coord = np.floor(pts / 0.1).astype(int)
    _, idx = np.unique(grid_coord, axis=0, return_index=True)
    pts, rgb, sem = pts[idx], rgb[idx], sem[idx]
    
    # 2. 相对高程先验
    z_floor = np.percentile(pts[:, 2], 0.1)
    global_rel_z = (pts[:, 2:3] - z_floor).astype(np.float32)
    # 🚀 修复 1: 强制将高度限制并归一化到 [0, 1]，防止巨大的数值方差淹没 DINO 特征
    global_rel_z = np.clip(global_rel_z / 100.0, 0.0, 1.0)
    
    # 3. 空间切块并提取 DINO (50m x 50m)
    x_max, y_max = pts.max(axis=0)[:2]
    chunk_count = 0
    
    # 🚀 修复 2: 动态步长。训练集 25m 步长(重叠增强)，验证集/测试集 50m 步长(无重叠)
    stride = 25 if split == "train" else 50
    
    for x in np.arange(0, x_max, stride):
        for y in np.arange(0, y_max, stride):
            mask = (pts[:, 0] >= x) & (pts[:, 0] < x + 50) & (pts[:, 1] >= y) & (pts[:, 1] < y + 50)
            if np.sum(mask) < 1024: continue 
            
            chunk_folder = save_dir / f"{scene_name}_{chunk_count}"
            
            if (chunk_folder / "extra_feat.npy").exists():
                chunk_count += 1
                continue
                
            chunk_folder.mkdir(exist_ok=True)
            
            pts_chunk = pts[mask]
            rgb_chunk = rgb[mask]
            sem_chunk = sem[mask]
            rel_z_chunk = global_rel_z[mask]
            
            grid_coord_05 = np.floor(pts_chunk / 0.5).astype(int)
            _, idx_05 = np.unique(grid_coord_05, axis=0, return_index=True)
            down_coords_np = pts_chunk[idx_05]
            down_colors_np = rgb_chunk[idx_05].astype(np.float32) / 255.0 
            
            down_coords = torch.tensor(down_coords_np, dtype=torch.float32)
            down_colors = torch.tensor(down_colors_np, dtype=torch.float32)
            
            down_features = extractor.extract_and_lift_features(down_coords, down_colors, resolution=0.5)
            
            kdtree = cKDTree(down_coords_np)
            _, indices = kdtree.query(pts_chunk, k=1, workers=-1)
            full_res_dino = down_features[indices].numpy()
            
            extra_feat = np.concatenate([rel_z_chunk, full_res_dino], axis=1).astype(np.float32)
            
            np.save(chunk_folder / "coord.npy", pts_chunk)
            np.save(chunk_folder / "color.npy", rgb_chunk)
            np.save(chunk_folder / "segment.npy", sem_chunk)
            np.save(chunk_folder / "extra_feat.npy", extra_feat)
            
            chunk_count += 1

    print(f"✅ {scene_name} processed: {chunk_count} chunks generated.")

# ========================================================
# 4. 主干分发逻辑：精准拆分 Train, Val 和 Test
# ========================================================
if __name__ == "__main__":
    extractor = DINOv3FeatureExtractor(model_path_or_name=str(WEIGHT_PATH))
    
    # 🚀 定义评估要求的 5 个特权测试场景 (全部放入 test 文件夹)
    test_suite_ids = ["5", "10", "15", "20", "25"]
    
    # 🚀 挑选 1 个场景用于训练期间的快速验证 (放入 val 文件夹，大幅加速每个 Epoch 的评测)
    val_suite_id = "5" 
    
    # 获取所有的 .ply 文件
    all_ply_files = sorted(list(RAW_DATA_ROOT.glob("*.ply")))
    
    if not all_ply_files:
        print(f"❌ 在 {RAW_DATA_ROOT} 下找不到任何 .ply 文件，请检查路径！")
    else:
        print(f"\n🚀 发现 {len(all_ply_files)} 个场景文件。正在按照你的新规则进行分配...")
        
        for f_path in all_ply_files:
            scene_name = f_path.stem  # 获取文件名，例如 "1_points_GTv3"
            
            try:
                scene_id = scene_name.split('_')[0]
            except Exception:
                print(f"⚠️ 无法解析场景编号: {scene_name}，跳过。")
                continue
                
            # 🚀 核心逻辑：分发到 train, val, test
            if scene_id in test_suite_ids:
                # 只要在这 5 个里面，就放一份进 test 做最终测试
                process_scene(f_path, extractor, "test")
                
                # 如果刚好是你选中的那个 val 场景，额外放一份进 val 做训练期验证
                if scene_id == val_suite_id:
                    process_scene(f_path, extractor, "val")
            else:
                # 其他剩下的所有场景，全放进 train 做训练！
                process_scene(f_path, extractor, "train")
                
        print(f"\n🎉 完美收工！数据已严格按要求分为 train, val 和 test，保存在: {OUT_DATA_ROOT}")