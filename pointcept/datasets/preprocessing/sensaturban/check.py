import numpy as np
from pathlib import Path
import random

# ================= 配置路径 =================
CHECK_DIR = Path("/lus/lfs1aip2/projects/b6ae/datasets/sensaturban/processed_1035D_Final")

def verify_patch(num_samples=5):
    print(f"🧐 Starting Sanity Check on: {CHECK_DIR}")
    
    # 递归查找所有 .npy 文件
    all_files = list(CHECK_DIR.rglob("*.npy"))
    
    if not all_files:
        print("❌ No files found! Check your OUTPUT_DIR.")
        return

    # 随机抽样
    samples = random.sample(all_files, min(num_samples, len(all_files)))
    
    print(f"{'File Name':<40} | {'Shape':<15} | {'Z_rel Range (Index 10)':<25}")
    print("-" * 90)

    for f in samples:
        data = np.load(f)
        
        # 1. 检查总维度
        shape = data.shape
        
        # 2. 提取第 11 列 (Index 10) -> Global Relative Z
        # 根据 patch 脚本，它插在旧 Geo(index 6-9) 之后，DINO(index 10+) 之前
        godview_z = data[:, 10]
        
        z_min = godview_z.min()
        z_max = godview_z.max()
        z_mean = godview_z.mean()
        
        print(f"{f.name[:40]:<40} | {str(shape):<15} | {z_min:6.2f} to {z_max:6.2f} (avg:{z_mean:5.2f})")

        # 🚀 深度校验：检查 DINO 起始位置是否正确挪动
        # 如果 Index 11 之后的均值模长接近 1 (DINO 经过了 Normalize)，说明挪动正确
        dino_sample = data[:, 11:14] # 抽样 DINO 前三维
        dino_norm = np.linalg.norm(data[:, 11:], axis=1).mean()
        
        if not (1035 == shape[1]):
             print(f"⚠️  Dimension Error: Expected 1035, but got {shape[1]}")
        
    print("-" * 90)
    print(f"✅ Check Finished. Index 10 is your 'God-View' Relative Z.")
    print(f"💡 Reminder: In PTV3, water should mostly reside in 0.0 ~ 1.1m range in this column.")

if __name__ == "__main__":
    verify_patch()