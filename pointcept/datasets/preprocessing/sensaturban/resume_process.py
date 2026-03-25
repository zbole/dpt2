import os
from pathlib import Path
from tqdm import tqdm

from preprocess_sensaturban import process_scene, RAW_DATA_ROOT, OUT_DATA_ROOT

def resume_processing():
    # ==========================================
    # 1. 锁定最后 5 个漏网之鱼 (Missing Scenes)
    # ==========================================
    target_scenes = {
        "cambridge_block_0", 
        "cambridge_block_1", 
        "cambridge_block_8", 
        "cambridge_block_9", 
        "cambridge_block_34"
    }

    # 扫描原始目录，过滤出 target_scenes
    all_ply_files = sorted(list(RAW_DATA_ROOT.glob("**/*.ply")))
    resume_ply_files = [f for f in all_ply_files if f.stem.lower() in target_scenes]

    print(f"🚀 Resuming Pipeline for the final {len(resume_ply_files)} PLY files...")
    
    # ==========================================
    # 2. 开始收尾处理
    # ==========================================
    for f in tqdm(resume_ply_files):
        result_msg = process_scene(f)
        print(result_msg)

if __name__ == "__main__":
    resume_processing()