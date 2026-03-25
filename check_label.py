from plyfile import PlyData
from pathlib import Path

# 🚀 指向你最原始的、还没处理过的 RAW PLY 文件
raw_test_file = Path("/lus/lfs1aip2/projects/b6ae/datasets/sensaturban/raw/test/birmingham_block_2.ply")

print(f"🧐 正在深度探测原始文件结构: {raw_test_file.name}")

try:
    with open(str(raw_test_file), 'rb') as f:
        plydata = PlyData.read(f)
    
    # 获取 'vertex' 元素下的所有属性名称
    properties = [prop.name for prop in plydata['vertex'].properties]
    
    print("\n📜 该 PLY 文件包含的所有字段 (Properties):")
    print("-" * 50)
    for p in properties:
        # 标记出可能是标签的字段
        is_label = any(keyword in p.lower() for keyword in ["class", "label", "seg", "id", "type"])
        prefix = "🎯 [可能是标签] ->" if is_label else "   "
        print(f"{prefix} {p}")
    print("-" * 50)

    # 如果找到了疑似标签，顺便看看它的取值范围
    for p in properties:
        if any(keyword in p.lower() for keyword in ["class", "label", "seg"]):
            data = plydata['vertex'][p]
            import numpy as np
            unique_vals = np.unique(data)
            print(f"\n💡 字段 '{p}' 的取值范围: {unique_vals}")
            if len(unique_vals) <= 1 and 255 in unique_vals:
                print("⚠️ 警告：该字段虽然存在，但里面全是 255 (Ignore Index)。")

except Exception as e:
    print(f"❌ 探测失败: {e}")