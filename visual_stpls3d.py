import numpy as np
import os
import glob
from plyfile import PlyData, PlyElement
import time
from collections import defaultdict

RESULTS_DIR = "./exp/stpls3d/STPLS3D_DSGG-PT_Final/result"
DATA_DIR = "/lus/lfs1aip2/projects/b6ae/datasets/STPLS3D/processed_1025D_Synthetic_Eval"
OUTPUT_DIR = "/lus/lfs1aip2/projects/b6ae/datasets/STPLS3D/result_DPT"

def save_multi_label_ply(coords, rgbs, gts, preds, save_path):
    """
    保存为多标签 PLY：一份坐标，同时携带 RGB颜色、GT标签、预测标签
    """
    num_points = coords.shape[0]

    # 'u1' 代表 uint8 (0-255)
    vertex = np.zeros(num_points, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('gt_class', 'u1'), ('pred_class', 'u1') 
    ])
    
    vertex['x'] = coords[:, 0].astype(np.float32)
    vertex['y'] = coords[:, 1].astype(np.float32)
    vertex['z'] = coords[:, 2].astype(np.float32)
    vertex['red'] = rgbs[:, 0].astype(np.uint8)
    vertex['green'] = rgbs[:, 1].astype(np.uint8)
    vertex['blue'] = rgbs[:, 2].astype(np.uint8)
    
    # 🚀 修复1：STPLS3D v3 对齐后的有效标签是 0~18。
    # 大于 18 的异常值，统一设为 255 (Ignore)
    vertex['gt_class'] = np.where(gts <= 18, gts, 255).astype(np.uint8)
    vertex['pred_class'] = np.where(preds <= 18, preds, 255).astype(np.uint8)

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=False).write(save_path)

def find_sub_dir(big_block, sub_id):
    for split in ['train', 'val', 'test']:
        candidate_dir = os.path.join(DATA_DIR, split, f"{big_block}_{sub_id}")
        if os.path.isdir(candidate_dir):
            return candidate_dir
    return None

def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pred_files = glob.glob(os.path.join(RESULTS_DIR, "*_pred.npy"))
    if not pred_files:
        print(f"❌ Error: 找不到预测文件！")
        return

    blocks_dict = defaultdict(list)
    for p_file in pred_files:
        basename = os.path.basename(p_file) 
        base_no_ext = basename.replace("_pred.npy", "") 
        parts = base_no_ext.split('_')
        blocks_dict["_".join(parts[:-1])].append((parts[-1], p_file))

    print(f"[{time.strftime('%H:%M:%S')}] 开始生成 Multi-label PLY...")

    for big_block, sub_list in blocks_dict.items():
        all_coords, all_preds, all_gts, all_rgbs = [], [], [], []
        for sub_id, p_file in sub_list:
            try:
                preds = np.load(p_file)
                sub_dir = find_sub_dir(big_block, sub_id)
                
                if not sub_dir: 
                    print(f"  ⚠️ 跳过: 找不到 {big_block}_{sub_id} 的原始数据集路径")
                    continue
                
                coord_path = os.path.join(sub_dir, "coord.npy")
                gt_path = os.path.join(sub_dir, "segment.npy")
                color_path = os.path.join(sub_dir, "color.npy")
                
                # 检查原始 npy 文件是否存在
                if not (os.path.exists(coord_path) and os.path.exists(gt_path) and os.path.exists(color_path)):
                    print(f"  ⚠️ 跳过: {sub_dir} 下缺少 coord/segment/color.npy")
                    continue

                coords = np.load(coord_path)
                gts = np.load(gt_path)
                rgbs = np.load(color_path)
                
                if rgbs.max() <= 1.0: rgbs = (rgbs * 255.0).astype(np.uint8)
                else: rgbs = rgbs.astype(np.uint8)
                
                # 检查点云数量是否严格一致
                if coords.shape[0] == preds.shape[0] == gts.shape[0]:
                    all_coords.append(coords)
                    all_preds.append(preds)
                    all_gts.append(gts)
                    all_rgbs.append(rgbs)
                else:
                    print(f"  ❌ 维度不匹配 {big_block}_{sub_id}: coords({coords.shape[0]}), preds({preds.shape[0]}), gts({gts.shape[0]})")
                    
            except Exception as e:
                print(f"  🔥 代码报错 {big_block}_{sub_id}: {e}")
                continue
        
        if not all_coords: continue
            
        base_coords = np.vstack(all_coords)
        base_preds = np.concatenate(all_preds)
        base_gts = np.concatenate(all_gts)
        base_rgbs = np.vstack(all_rgbs)
        
        # 🚀 修复2：强制对齐标签体系到官方的 0~18 
        # 1. 处理 GT：将 255(被当做ignore的Ground) 强行映射回 0。原本的 0~17 整体 +1
        base_gts_aligned = np.where(base_gts == 255, 0, base_gts + 1).astype(np.uint8)
        
        # 2. 处理 Preds：模型输出的是 0~17，整体 +1 变成 1~18 对齐真实语义
        base_preds_aligned = (base_preds + 1).astype(np.uint8)
        
        # 保存为极其精简的多标签 PLY (注意这里传入的是 aligned 之后的标签)
        output_ply_path = os.path.join(OUTPUT_DIR, f"{big_block}_MultiLabel.ply")
        save_multi_label_ply(base_coords, base_rgbs, base_gts_aligned, base_preds_aligned, output_ply_path)
        print(f"✅ 已保存至: {output_ply_path} (点数: {base_coords.shape[0]:,})")

    print(f"\n🎉 耗时: {(time.time() - start_time)/60:.2f} 分钟")

if __name__ == "__main__":
    main()