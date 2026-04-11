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
import shutil

# ========================================================
# 1. 路径与全局配置 (精准匹配你的实际目录结构)
# ========================================================
dataset_root = Path("/lus/lfs1aip2/projects/b6ae/datasets/STPLS3D").resolve()

# 🚀 输入路径：真实数据与三个版本的合成数据
RAW_REAL_ROOT = dataset_root / "raw" / "RealWorldData"
RAW_SYN_ROOTS = [
    dataset_root / "raw" / "Synthetic_v1",
    dataset_root / "raw" / "Synthetic_v2",
    dataset_root / "raw" / "Synthetic_v3"
]

# 🚀 底层特征池路径 (耗时的 DINO 处理结果统一存放在这里)
POOL_ROOT = dataset_root / "processed_pool_1025D"

# 🚀 最终给 PTV3 训练用的数据集路径 (包含 train 和 test)
FINAL_DATASET_ROOT = dataset_root / "WMSC_Hybrid_Dataset"

WEIGHT_PATH = Path("./weights").resolve()
DEVICE = "cuda"

# ⚠️ 明确指定 WMSC 测试文件的关键字
WMSC_KEYWORDS = ["WMSC"] 

# ========================================================
# 2. DINOv3 提取器 (保持你的 SOTA 设计)
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
# 3. 场景处理函数 (存入 Pool，区分数据属性)
# ========================================================
def process_scene_to_pool(scene_path, extractor, pool_category):
    """
    pool_category 只能是: 'real_train', 'wmsc_test', 'syn_train'
    """
    scene_path = Path(scene_path)
    scene_name = scene_path.stem
    
    save_dir = POOL_ROOT / pool_category
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Processing [{pool_category}] file: {scene_name} ...")
    try:
        with open(str(scene_path), 'rb') as f:
            v = PlyData.read(f)['vertex']
        pts = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
        rgb = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.uint8)
        sem = v['class'].astype(np.int16) if 'class' in v else np.full(len(pts), 255)
        
        # 标签映射：保留 18 个有效类，其余设为 255
        valid_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]
        mapping = np.full(256, 255, dtype=np.int16)
        for new_id, orig_id in enumerate(valid_labels):
            mapping[orig_id] = new_id
        sem = mapping[sem]
            
    except Exception as e: 
        print(f"❌ Load Error: {e}")
        return

    # 1. 基础降采样 (0.1m)
    pts -= pts.min(axis=0)
    grid_coord = np.floor(pts / 0.1).astype(int)
    _, idx = np.unique(grid_coord, axis=0, return_index=True)
    pts, rgb, sem = pts[idx], rgb[idx], sem[idx]
    
    # 2. 宏观相对高程先验 (SuperCPE 依赖项: 0.1% 分位数作为物理底盘)
    z_floor = np.percentile(pts[:, 2], 0.1)
    global_rel_z = np.clip((pts[:, 2:3] - z_floor).astype(np.float32) / 100.0, 0.0, 1.0)
    
    x_max, y_max = pts.max(axis=0)[:2]
    chunk_count = 0
    
    # 🚀 策略：测试集(WMSC)无重叠采样(50m)，训练集有重叠增强(25m)
    stride = 50 if pool_category == "wmsc_test" else 25
    
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
            
            # DINO 提取
            grid_coord_05 = np.floor(pts_chunk / 0.5).astype(int)
            _, idx_05 = np.unique(grid_coord_05, axis=0, return_index=True)
            down_coords_np = pts_chunk[idx_05]
            down_colors_np = rgb_chunk[idx_05].astype(np.float32) / 255.0 
            
            down_coords = torch.tensor(down_coords_np, dtype=torch.float32)
            down_colors = torch.tensor(down_colors_np, dtype=torch.float32)
            
            down_features = extractor.extract_and_lift_features(down_coords, down_colors, resolution=0.5)
            
            # 还原到原始点
            kdtree = cKDTree(down_coords_np)
            _, indices = kdtree.query(pts_chunk, k=1, workers=-1)
            full_res_dino = down_features[indices].numpy()
            
            # 拼接高程先验与 DINO 特征 [1维高程 + 1024维DINO = 1025维]
            extra_feat = np.concatenate([rel_z_chunk, full_res_dino], axis=1).astype(np.float32)
            
            np.save(chunk_folder / "coord.npy", pts_chunk)
            np.save(chunk_folder / "color.npy", rgb_chunk)
            np.save(chunk_folder / "segment.npy", sem_chunk)
            np.save(chunk_folder / "extra_feat.npy", extra_feat)
            
            chunk_count += 1

    print(f"✅ {scene_name} processed: {chunk_count} chunks generated.")

# ========================================================
# 4. 终极软链接分发逻辑 (零额外存储，秒级组装数据集)
# ========================================================
def create_hybrid_symlinks(output_dir, train_pools, test_pools):
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    
    # 每次运行前清理旧链接，保持干净
    if output_dir.exists():
        shutil.rmtree(output_dir)
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    
    print(f"\n🔗 Assembling Hybrid Dataset at: {output_dir.name}")
    
    # 组装 Train (包含 Real + Syn)
    for pool_name in train_pools:
        pool_path = POOL_ROOT / pool_name
        if pool_path.exists():
            for chunk_dir in pool_path.iterdir():
                if chunk_dir.is_dir():
                    os.symlink(chunk_dir, train_dir / chunk_dir.name)
                    
    # 组装 Test (纯 WMSC)
    for pool_name in test_pools:
        pool_path = POOL_ROOT / pool_name
        if pool_path.exists():
            for chunk_dir in pool_path.iterdir():
                if chunk_dir.is_dir():
                    os.symlink(chunk_dir, test_dir / chunk_dir.name)
                    
    print(f"   📊 Train chunks total: {len(list(train_dir.iterdir()))}")
    print(f"   📊 Test chunks total:  {len(list(test_dir.iterdir()))}")

def is_wmsc(filename):
    """判断文件是否属于 WMSC 测试区"""
    return any(keyword in filename for keyword in WMSC_KEYWORDS)


if __name__ == "__main__":
    extractor = DINOv3FeatureExtractor(model_path_or_name=str(WEIGHT_PATH))
    
    # ----------------------------------------------------
    # 阶段 1: 抽取所有数据到 Pool (只执行一次，中断可续传)
    # ----------------------------------------------------
    print("========================================")
    print("🌟 Phase 1: Feature Extraction & Chunking")
    print("========================================")
    
    # 1. 处理 RealWorldData
    if RAW_REAL_ROOT.exists():
        for f_path in RAW_REAL_ROOT.glob("*.ply"):
            if is_wmsc(f_path.name):
                # 如果是 WMSC，放入测试池
                process_scene_to_pool(f_path, extractor, "wmsc_test")
            else:
                # OCCC, RA, USC 等放入真实训练池
                process_scene_to_pool(f_path, extractor, "real_train")
    else:
        print(f"⚠️ 找不到真实数据路径: {RAW_REAL_ROOT}")

    # 2. 处理 Synthetic 数据 (V1 - V3) 放入合成训练池
    for syn_root in RAW_SYN_ROOTS:
        if syn_root.exists():
            for f_path in syn_root.glob("*.ply"):
                process_scene_to_pool(f_path, extractor, "syn_train")
        else:
            print(f"⚠️ 找不到合成数据路径: {syn_root}")

    # ----------------------------------------------------
    # 阶段 2: 构建终极混合评测目录 (满足你的三个核心要求)
    # ----------------------------------------------------
    print("\n========================================")
    print("🌟 Phase 2: Assembling WMSC Hybrid Evaluation Dataset")
    print("========================================")
    
    # 完美满足：
    # 1. 训练包含 Real (除WMSC外)
    # 2. 训练包含 Syn (V1-V3)
    # 3. 混合训练，WMSC纯测试
    create_hybrid_symlinks(
        FINAL_DATASET_ROOT, 
        train_pools=["real_train", "syn_train"], 
        test_pools=["wmsc_test"]
    )
    
    print(f"\n🎉 All Done! 最终数据集已准备就绪: {FINAL_DATASET_ROOT}")
    print("👉 请在你的 PTV3 训练 YAML 中将 data_root 指向此目录。祝 SOTA！")