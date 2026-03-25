import os
import json
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData
from multiprocessing import Pool

# ================= 路径配置 =================
RAW_DIR = Path("/lus/lfs1aip2/projects/b6ae/datasets/sensaturban/raw")
INPUT_DIR = Path("/lus/lfs1aip2/projects/b6ae/datasets/sensaturban/processed_1024D_Full")
OUTPUT_DIR = Path("/lus/lfs1aip2/projects/b6ae/datasets/sensaturban/processed_1035D_with_relativeZ")
Z_FLOOR_CACHE = Path("./macro_z_floors.json") # 用于保存底盘高度的字典文件

# ================= 工具函数 =================
def get_macro_block_name(chunk_name):
    """
    从小豆腐块名字还原宏观 Block 名字。
    例: birmingham_block_2_0 -> birmingham_block_2
    """
    parts = chunk_name.split('_')
    return "_".join(parts[:-1])


# ================= Step 1: 扫描原始 PLY 计算全局底盘 =================
def step1_calculate_macro_floors():
    print("🚀 [Step 1] 开始扫描原始 PLY 文件，计算宏观底盘高度...")
    
    # 查找 raw 文件夹下所有的 .ply 文件 (支持遍历子文件夹)
    ply_files = list(RAW_DIR.rglob("*.ply"))
    if not ply_files:
        print(f"❌ 未在 {RAW_DIR} 中找到任何 .ply 文件，请检查路径。")
        return False
        
    z_floors = {}
    for ply_path in tqdm(ply_files, desc="Parsing PLY Files"):
        # 假设文件名就是 macro block 的名字，如 birmingham_block_2.ply
        macro_name = ply_path.stem 
        
        try:
            plydata = PlyData.read(str(ply_path))
            z_coords = np.asarray(plydata['vertex']['z']).astype(np.float32)
            
            # 使用 0.1% 分位数作为稳健的底盘高度，过滤极个别离群噪点
            z_floor = float(np.percentile(z_coords, 0.1))
            z_floors[macro_name] = z_floor
            
        except Exception as e:
            print(f"⚠️ 读取 {ply_path.name} 时出错: {e}")
            
    # 将字典保存到本地
    with open(Z_FLOOR_CACHE, 'w') as f:
        json.dump(z_floors, f, indent=4)
        
    print(f"✅ Step 1 完成！已成功提取 {len(z_floors)} 个宏观场景的底盘高度，保存在 {Z_FLOOR_CACHE}。")
    return True


# ================= Step 2: 多进程生成 Pointcept 数据集 =================
# 全局变量供多进程读取
GLOBAL_Z_FLOORS = {}

def process_single_chunk(chunk_path_str):
    chunk_path = Path(chunk_path_str)
    rel_path = chunk_path.relative_to(INPUT_DIR)
    
    try:
        macro_name = get_macro_block_name(chunk_path.name)
        
        # 获取宏观底盘高度。如果万一匹配不上，默认给 0.0 并打印警告
        if macro_name in GLOBAL_Z_FLOORS:
            z_floor = GLOBAL_Z_FLOORS[macro_name]
        else:
            print(f"⚠️ 警告: 找不到 {macro_name} 的底盘数据，Z-floor 设为 0.0")
            z_floor = 0.0

        # 1. 极速读取 Numpy 数组
        coord = np.load(chunk_path / "coord.npy")         # (N, 3) 
        geo   = np.load(chunk_path / "geo_prior.npy")     # (N, 4)
        dino  = np.load(chunk_path / "dino_1024d.npy")    # (N, 1024)

        # 2. 计算精准上帝视角相对高度 (1D)
        global_rel_z = (coord[:, 2] - z_floor).reshape(-1, 1).astype(np.float32)

        # 3. 融合生成 1029 维 extra_feat: [Geo(4), RelZ(1), DINO(1024)]
        extra_feat = np.concatenate([geo, global_rel_z, dino], axis=1).astype(np.float32)

        # 4. 拷贝与写入
        out_chunk_path = OUTPUT_DIR / rel_path
        out_chunk_path.mkdir(parents=True, exist_ok=True)

        shutil.copy2(chunk_path / "coord.npy", out_chunk_path / "coord.npy")
        shutil.copy2(chunk_path / "color.npy", out_chunk_path / "color.npy")
        shutil.copy2(chunk_path / "segment.npy", out_chunk_path / "segment.npy")
        np.save(out_chunk_path / "extra_feat.npy", extra_feat) 
        
        return True
    except Exception as e:
        print(f"❌ 处理 {chunk_path.name} 失败: {e}")
        return False


def step2_generate_pointcept_data():
    global GLOBAL_Z_FLOORS
    print(f"\n🚀 [Step 2] 开始生成 1035D 数据集...")
    
    if not Z_FLOOR_CACHE.exists():
        print(f"❌ 找不到底盘缓存文件 {Z_FLOOR_CACHE}，请先运行 Step 1。")
        return
        
    with open(Z_FLOOR_CACHE, 'r') as f:
        GLOBAL_Z_FLOORS = json.load(f)
        
    chunk_folders = [p.parent for p in INPUT_DIR.rglob("coord.npy")]
    chunk_paths_str = [str(p) for p in chunk_folders]
    total_chunks = len(chunk_paths_str)
    
    if total_chunks == 0:
        print(f"❌ 在 {INPUT_DIR} 找不到任何 chunk。")
        return
        
    print(f"⚡ 共找到 {total_chunks} 个 chunks。启动多进程处理...")
    
    # 启动 12 进程狂飙
    with Pool(processes=12) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_chunk, chunk_paths_str), total=total_chunks))
        
    print(f"\n🎉 数据集处理完毕! 成功率: {sum(results)}/{total_chunks}")
    print(f"📁 输出路径: {OUTPUT_DIR}")


if __name__ == "__main__":
    # 如果你已经跑过了 Step 1 生成了 json 文件，可以直接注释掉下面这行
    if step1_calculate_macro_floors():
        step2_generate_pointcept_data()