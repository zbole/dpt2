import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# ========================================================
# 1. 路径与全局配置
# ========================================================
dataset_root = Path("/lus/lfs1aip2/projects/b6ae/datasets/sensaturban").resolve()
IN_DATA_ROOT = dataset_root / "processed_1024D_Full"    

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 15  # 🚀 增加训练轮数，确保网络看遍伯明翰和剑桥
LR = 0.001   # 🚀 降低学习率，追求更稳定的收敛

TARGET_NAMES = ["Other", "Parking", "TrafficRoad", "Footpath", "Water"]

def map_labels(sem):
    mapped = np.zeros_like(sem)  
    mapped[sem == 5] = 1
    mapped[sem == 7] = 2
    mapped[sem == 10] = 3
    mapped[sem == 12] = 4
    mapped[sem == 255] = 255  
    return mapped

# ========================================================
# 2. 增强版 DINO-MLP
# ========================================================
class DinoPromptMLP(nn.Module):
    def __init__(self, in_dim=1024, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),       # 扩容
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),              # 增强抗过拟合
            nn.Linear(512, 128),          # 扩容
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ========================================================
# 3. 极速降采样 + 均衡 mmap 数据集
# ========================================================
class ChunkDataset(Dataset):
    def __init__(self, split="train", max_points=8192, voxel_size=0.5):
        self.split_dir = IN_DATA_ROOT / split
        self.chunk_dirs = sorted([d for d in self.split_dir.iterdir() if d.is_dir()])
        self.max_points = max_points
        self.voxel_size = voxel_size  # 🚀 动态降采样分辨率

    def __len__(self):
        return len(self.chunk_dirs)

    def __getitem__(self, idx):
        chunk = self.chunk_dirs[idx]
        try:
            # 1. 全量读取坐标和标签 (文件很小)
            coord = np.load(chunk / "coord.npy")
            sem = np.load(chunk / "segment.npy")
            mapped_sem = map_labels(sem).astype(np.int64)
            
            valid_mask = mapped_sem != 255
            if not valid_mask.any(): 
                return torch.zeros((1, 1024), dtype=torch.float32), torch.zeros((1,), dtype=torch.long)
                
            valid_indices = np.where(valid_mask)[0]
            valid_coords = coord[valid_indices]
            
            # 🚀 2. 空间体素降采样 (On-the-fly Voxelization)
            # 抹平高频边界噪声，只留 0.5m 网格内的一个代表点
            grid_coords = np.floor(valid_coords / self.voxel_size).astype(np.int32)
            _, unique_idx = np.unique(grid_coords, axis=0, return_index=True)
            
            # 拿到降采样后在原数组中的真实索引
            downsampled_indices = valid_indices[unique_idx]
            downsampled_labels = mapped_sem[downsampled_indices]
            
            # 🚀 3. 类别均衡抽样 (Stratified Sampling)
            # 绝不让 Other 类淹没 Water 类
            target_per_class = self.max_points // 5
            final_sampled_list = []
            
            for c in range(5):
                c_mask = downsampled_labels == c
                c_idx = downsampled_indices[c_mask]
                
                if len(c_idx) == 0:
                    continue
                elif len(c_idx) > target_per_class:
                    c_sampled = np.random.choice(c_idx, target_per_class, replace=False)
                else:
                    c_sampled = c_idx 
                    
                final_sampled_list.append(c_sampled)
                
            final_sampled_idx = np.concatenate(final_sampled_list)
            
            # 🚀 4. 排序后利用 mmap 极速提取那部分极度纯净的 1024D 特征
            final_sampled_idx = np.sort(final_sampled_idx) 
            dino_feat_mmap = np.load(chunk / "dino_1024d.npy", mmap_mode='r')
            
            valid_feats = dino_feat_mmap[final_sampled_idx].astype(np.float32)
            valid_labels = mapped_sem[final_sampled_idx]
                
            return torch.from_numpy(valid_feats), torch.from_numpy(valid_labels)
        except Exception:
            return torch.zeros((1, 1024), dtype=torch.float32), torch.zeros((1,), dtype=torch.long)

def chunk_collate(batch):
    feats = torch.cat([item[0] for item in batch], dim=0)
    labels = torch.cat([item[1] for item in batch], dim=0)
    return feats, labels

def calculate_miou(y_true, y_pred, num_classes=5):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    ious = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        ious.append(float('nan') if union == 0 else intersection / union)
    return np.array(ious)

# ========================================================
# 4. 训练与验证逻辑
# ========================================================
def train_and_eval():
    model = DinoPromptMLP(in_dim=1024, out_dim=5).to(DEVICE)
    
    # 🚀 温和且平衡的权重：让数据分布自己说话
    weights = torch.tensor([1.0, 1.5, 1.5, 2.0, 2.0], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 🚀 极速 DataLoader：高并发，大 Batch Size
    train_loader = DataLoader(ChunkDataset("train", max_points=8192, voxel_size=0.5), 
                              batch_size=256, shuffle=True, 
                              num_workers=16, collate_fn=chunk_collate, pin_memory=True, prefetch_factor=2)
                              
    val_loader = DataLoader(ChunkDataset("val", max_points=8192, voxel_size=0.5), 
                            batch_size=256, shuffle=False, 
                            num_workers=16, collate_fn=chunk_collate, pin_memory=True, prefetch_factor=2)
    
    print(f"\n🚀 Starting PURE DINO Probe Training! Total Train Chunks: {len(train_loader.dataset)}")
    best_miou = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for feats, labels in pbar:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(feats)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        
        # 验证阶段
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for feats, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Eval]"):
                preds = model(feats.to(DEVICE)).argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.numpy())
                
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        ious = calculate_miou(all_labels, all_preds)
        
        mean_iou = np.nanmean(ious)
        print(f"📉 Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")
        print(f"📊 Epoch {epoch+1} mIoU: {mean_iou:.4f}")
        for i, name in enumerate(TARGET_NAMES):
            print(f"   - {name}: {ious[i]:.4f}")
            
        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), "best_dino_mlp_pure.pth")
            print("🌟 Best model saved!")

if __name__ == "__main__":
    train_and_eval()
    print("\n🎉 Pure DINO Evaluation Finished!")