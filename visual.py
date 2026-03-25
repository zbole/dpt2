import numpy as np
import os
import glob
from plyfile import PlyData, PlyElement
import time
from collections import defaultdict

RESULTS_DIR = "./exp/sensaturban/PTV3_1035D_z_DINO1024_DynGCDM_v4/result"
DATA_DIR = "../datasets/sensaturban/processed_1035D_with_relativeZ"
OUTPUT_DIR = "./visualized_results"

def save_multi_label_ply(coords, rgbs, gts, preds, save_path):
    """
    保存为多标签 PLY：一份坐标，同时携带 RGB颜色、GT标签、预测标签
    """
    num_points = coords.shape[0]

    # 🚀 定义包含自定义属性的数据结构
    # 'u1' 代表 uint8 (0-255)
    vertex = np.zeros(num_points, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('gt_class', 'u1'), ('pred_class', 'u1') # 👈 这里是我们追加的自定义标签！
    ])
    
    vertex['x'] = coords[:, 0].astype(np.float32)
    vertex['y'] = coords[:, 1].astype(np.float32)
    vertex['z'] = coords[:, 2].astype(np.float32)
    vertex['red'] = rgbs[:, 0].astype(np.uint8)
    vertex['green'] = rgbs[:, 1].astype(np.uint8)
    vertex['blue'] = rgbs[:, 2].astype(np.uint8)
    
    # 过滤异常值，统一归为 13 (Ignore)
    vertex['gt_class'] = np.where((gts >= 0) & (gts <= 12), gts, 13).astype(np.uint8)
    vertex['pred_class'] = np.where((preds >= 0) & (preds <= 12), preds, 13).astype(np.uint8)

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
                if not sub_dir: continue
                
                coords = np.load(os.path.join(sub_dir, "coord.npy"))
                gts = np.load(os.path.join(sub_dir, "segment.npy"))
                rgbs = np.load(os.path.join(sub_dir, "color.npy"))
                
                if rgbs.max() <= 1.0: rgbs = (rgbs * 255.0).astype(np.uint8)
                else: rgbs = rgbs.astype(np.uint8)
                
                if coords.shape[0] == preds.shape[0] == gts.shape[0]:
                    all_coords.append(coords)
                    all_preds.append(preds)
                    all_gts.append(gts)
                    all_rgbs.append(rgbs)
            except Exception as e:
                continue
        
        if not all_coords: continue
            
        base_coords = np.vstack(all_coords)
        base_preds = np.concatenate(all_preds)
        base_gts = np.concatenate(all_gts)
        base_rgbs = np.vstack(all_rgbs)
        
        # 保存为极其精简的多标签 PLY
        output_ply_path = os.path.join(OUTPUT_DIR, f"{big_block}_MultiLabel.ply")
        save_multi_label_ply(base_coords, base_rgbs, base_gts, base_preds, output_ply_path)
        print(f"✅ 已保存至: {output_ply_path} (点数: {base_coords.shape[0]:,})")

    print(f"\n🎉 耗时: {(time.time() - start_time)/60:.2f} 分钟")

if __name__ == "__main__":
    main()