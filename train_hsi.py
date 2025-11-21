import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# 引用修改过的模型文件
from models.fastvit_hsi import fastvit_ma36
import wandb
from tqdm import tqdm

# === 1. 工具函数: 计算 Top-K 准确率 ===
def accuracy(output, target, topk=(1,)):
    """计算指定 k 值的准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# === 2. 自定义数据集 (读取 .npy) ===
class HSIDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.data = []
        self.labels = []
        self.root_dir = os.path.join(root_dir, split)
        
        if not os.path.exists(self.root_dir):
            raise ValueError(f"找不到数据目录: {self.root_dir}")
            
        classes = sorted(os.listdir(self.root_dir))
        class_to_idx = {cls_name: int(cls_name) - 1 for cls_name in classes}
        
        print(f"[INFO] 正在加载 {split} 数据...")
        for cls_name in classes:
            cls_folder = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_folder): continue
            
            for fname in os.listdir(cls_folder):
                if fname.endswith('.npy'):
                    path = os.path.join(cls_folder, fname)
                    self.data.append(path)
                    self.labels.append(class_to_idx[cls_name])
        
        print(f"-> 已加载 {len(self.data)} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        label = self.labels[idx]
        img = np.load(path).astype(np.float32)
        img = img / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1) # (H,W,C) -> (C,H,W)
        return img, label

# === 3. 训练主逻辑 ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/indian_pines_pca30_npy')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--checkpoint', type=str, default='./fastvit_ma36.pth.tar')
    parser.add_argument('--exp-name', type=str, default='HSI_PCA30_Test')
    parser.add_argument('--project', type=str, default='FastViT-HSI-Comparison', help='WandB Project Name')
    args = parser.parse_args()

    # 初始化 WandB
    wandb.init(project=args.project, name=args.exp_name, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_set = HSIDataset(args.data, 'train')
    val_set = HSIDataset(args.data, 'val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    print(f"[INFO] 初始化 FastViT-MA36 (in_chans=30)...")
    model = fastvit_ma36(num_classes=16, in_chans=30, fork_feat=False)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[INFO] 加载权重: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint: checkpoint = checkpoint['state_dict']
        
        model_dict = model.state_dict()
        # 智能过滤形状不匹配的层
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        print(f"[INFO] 成功复用层数: {len(pretrained_dict)} / {len(model_dict)}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    
    # === 训练循环 ===
    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        losses = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            losses.update(loss.item(), inputs.size(0))
            pbar.set_postfix({'Loss': loss.item()})
            
        # --- Validate ---
        model.eval()
        val_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 计算准确率 (Top-1 和 Top-5)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                
                val_losses.update(loss.item(), inputs.size(0))
                top1.update(acc1.item(), inputs.size(0))
                top5.update(acc5.item(), inputs.size(0))
        
        scheduler.step()
        
        # [关键] WandB 日志命名严格对齐 timm 官方格式
        wandb.log({
            "train_loss": losses.avg,      # 参数 1
            "eval_loss": val_losses.avg,   # 参数 2 (注意是 eval 不是 val)
            "eval_top1": top1.avg,         # 参数 3 (注意是 top1 不是 acc)
            "eval_top5": top5.avg,         # 参数 4
            "lr": optimizer.param_groups[0]['lr'], # 参数 5
            "epoch": epoch + 1
        })
        
        print(f" -> Eval Top1: {top1.avg:.2f}% (Best: {best_acc:.2f}%)")
        
        if top1.avg > best_acc:
            best_acc = top1.avg
            torch.save(model.state_dict(), f"./best_hsi_30ch_{args.exp_name}.pth")

    print(f"[DONE] 最佳准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main()