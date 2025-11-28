#
# Training Script for FastViT with RepSS-Mixer (SOTA Metrics Version)
# [升级] 集成 OA, AA, Kappa 三大 SOTA 核心指标
#

import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import numpy as np
# [修改] 导入 Day 7 的小切片专用模型文件
import models.fastvit_repss_small

import torch
import torch.nn as nn
import torchvision.utils
from torch.utils.data import Dataset, DataLoader

# [新增] 引入 scikit-learn 计算专业指标
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score

from timm.data import Mixup
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    model_parameters,
)
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

# 引入 RepSS 模型
try:
    from models.fastvit_repss import fastvit_ma36 
except ImportError:
    print("[ERROR] 找不到 models/fastvit_repss.py，请检查 models 文件夹！")
    exit(1)

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("train")

# =========================================================================
# HSI 数据集类
# =========================================================================
class HSIDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.data = []
        self.labels = []
        self.root_dir = os.path.join(root_dir, split)
        
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Data directory not found: {self.root_dir}")
            
        classes = sorted(os.listdir(self.root_dir))
        class_to_idx = {cls_name: int(cls_name) - 1 for cls_name in classes}
        
        print(f"[INFO] Loading {split} data from {self.root_dir}...")
        for cls_name in classes:
            cls_folder = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_folder): continue
            
            for fname in os.listdir(cls_folder):
                if fname.endswith('.npy'):
                    path = os.path.join(cls_folder, fname)
                    self.data.append(path)
                    self.labels.append(class_to_idx[cls_name])
        
        print(f"-> Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        label = self.labels[idx]
        img = np.load(path).astype(np.float32)
        img = img / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, label

# =========================================================================
# 参数定义
# =========================================================================
config_parser = parser = argparse.ArgumentParser(description="Training Config", add_help=False)
parser.add_argument("-c", "--config", default="", type=str, metavar="FILE", help="YAML config file")

parser = argparse.ArgumentParser(description="PyTorch HSI Training (RepSS-Mixer)")

# Dataset
parser.add_argument("--data-dir", metavar="DIR", default="./data/indian_pines_pca30_npy", help="path to dataset")
parser.add_argument("--train-split", metavar="NAME", default="train")
parser.add_argument("--val-split", metavar="NAME", default="val")

# Model
parser.add_argument("--model", default="fastvit_ma36", type=str, metavar="MODEL")
parser.add_argument("--initial-checkpoint", default="", type=str, metavar="PATH")
parser.add_argument("--resume", default="", type=str, metavar="PATH")
parser.add_argument("--num-classes", type=int, default=16, metavar="N")
parser.add_argument("--gp", default=None, type=str, metavar="POOL")
parser.add_argument("--drop", type=float, default=0.0, metavar="PCT")
parser.add_argument("--drop-path", type=float, default=0.0, metavar="PCT")
parser.add_argument("--drop-block", type=float, default=None, metavar="PCT")
parser.add_argument("--bn-momentum", type=float, default=None)
parser.add_argument("--bn-eps", type=float, default=None)

# Optimizer
parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER")
parser.add_argument("--opt-eps", default=None, type=float, metavar="EPSILON")
parser.add_argument("--opt-betas", default=None, type=float, nargs="+", metavar="BETA")
parser.add_argument("--momentum", type=float, default=0.9, metavar="M")
parser.add_argument("--weight-decay", type=float, default=0.05)
parser.add_argument("--clip-grad", type=float, default=None, metavar="NORM")
parser.add_argument("--clip-mode", type=str, default="norm")

# LR Scheduler
parser.add_argument("--sched", default="cosine", type=str, metavar="SCHEDULER")
parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")
parser.add_argument("--min-lr", type=float, default=1e-5, metavar="LR")
parser.add_argument("--warmup-lr", type=float, default=1e-6, metavar="LR")
parser.add_argument("--warmup-epochs", type=int, default=5, metavar="N")
parser.add_argument("--epochs", type=int, default=100, metavar="N")
parser.add_argument("--start-epoch", default=None, type=int, metavar="N")
parser.add_argument("--cooldown-epochs", type=int, default=10, metavar="N")
parser.add_argument("--decay-rate", type=float, default=0.1)

# Augmentation
parser.add_argument("--no-aug", action="store_true", default=True)
parser.add_argument("--mixup", type=float, default=0.0, help="mixup alpha")
parser.add_argument("--cutmix", type=float, default=0.0, help="cutmix alpha")
parser.add_argument("--mixup-prob", type=float, default=1.0)
parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
parser.add_argument("--mixup-mode", type=str, default="batch")
parser.add_argument("--smoothing", type=float, default=0.1)

# Model EMA
parser.add_argument("--model-ema", action="store_true", default=False)
parser.add_argument("--model-ema-decay", type=float, default=0.9995)
parser.add_argument("--model-ema-force-cpu", action="store_true", default=False)

# System
parser.add_argument("-b", "--batch-size", type=int, default=64, metavar="N")
parser.add_argument("-vb", "--validation-batch-size", type=int, default=None, metavar="N")
parser.add_argument("-j", "--workers", type=int, default=8, metavar="N")
parser.add_argument("--amp", action="store_true", default=False)
parser.add_argument("--pin-mem", action="store_true", default=False)
parser.add_argument("--output", default="", type=str, metavar="PATH")
parser.add_argument("--experiment", default="", type=str, metavar="NAME")
parser.add_argument("--eval-metric", default="OA", type=str, help="Best metric (OA, AA, Kappa)")
parser.add_argument("--log-interval", type=int, default=50)
parser.add_argument("--log-wandb", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint-hist", type=int, default=10)
parser.add_argument("--no-prefetcher", action="store_true", default=False)

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    return args

# =========================================================================
# 主训练流程
# =========================================================================
def main():
    setup_default_logging()
    args = _parse_args()

    if args.log_wandb and has_wandb:
        project_name = os.environ.get("WANDB_PROJECT", args.experiment)
        wandb.init(project=project_name, config=args)

    args.distributed = False
    args.device = "cuda:0"
    random_seed(args.seed, 0)

    _logger.info(f"Creating RepSS-Mixer model: {args.model} with in_chans=30")
    
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path="",
        in_chans=30,
        strict=False
    )

    # 权重加载
    if args.initial_checkpoint and os.path.exists(args.initial_checkpoint):
        print(f"[INFO] Loading checkpoint: {args.initial_checkpoint}")
        checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint: checkpoint = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        print(f"[INFO] Successfully loaded {len(pretrained_dict)} layers.")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    else:
        print("[WARNING] No checkpoint loaded, training from scratch.")

    model.cuda()
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # AMP
    amp_autocast = suppress
    loss_scaler = None
    if args.amp and torch.cuda.is_available():
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        _logger.info("Using Native AMP.")

    # EMA
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay)

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = args.start_epoch or 0

    # Dataset
    dataset_train = HSIDataset(args.data_dir, split=args.train_split)
    dataset_eval = HSIDataset(args.data_dir, split=args.val_split)

    # Mixup
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0.0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode, label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )

    loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True
    )
    loader_eval = DataLoader(
        dataset_eval, batch_size=args.validation_batch_size or args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=args.pin_mem
    )

    if mixup_fn is not None:
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # Saver Setup
    # [修改] eval_metric 默认为 OA (即 top1)
    best_metric = 0.0
    saver = None
    if args.experiment:
        output_dir = os.path.join("./output", args.experiment)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema,
            amp_scaler=loss_scaler, checkpoint_dir=output_dir, max_history=args.checkpoint_hist,
            decreasing=False # Accuracy 越高越好
        )

    try:
        for epoch in range(start_epoch, num_epochs):
            
            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=None,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler,
                model_ema=model_ema, mixup_fn=mixup_fn
            )

            # [修改] 使用新的 validate_hsi 函数计算 SOTA 指标
            eval_metrics = validate_hsi(
                model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast
            )

            # EMA 验证
            ema_eval_metrics = None
            if model_ema is not None:
                ema_eval_metrics = validate_hsi(
                    model_ema.module, loader_eval, validate_loss_fn, args, 
                    amp_autocast=amp_autocast, log_suffix=" (EMA)"
                )

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics["OA"])

            # [修改] WandB 日志记录所有 SOTA 指标
            if args.log_wandb and has_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics['loss'],
                    "lr": optimizer.param_groups[0]['lr'],
                    
                    "eval_loss": eval_metrics['loss'],
                    "eval_OA": eval_metrics['OA'],     # Overall Accuracy
                    "eval_AA": eval_metrics['AA'],     # Average Accuracy
                    "eval_Kappa": eval_metrics['Kappa'], # Kappa Coefficient
                }
                if ema_eval_metrics:
                    log_dict.update({
                        "eval_loss_ema": ema_eval_metrics['loss'],
                        "eval_OA_ema": ema_eval_metrics['OA'],
                        "eval_AA_ema": ema_eval_metrics['AA'],
                        "eval_Kappa_ema": ema_eval_metrics['Kappa'],
                    })
                wandb.log(log_dict)

            # 保存最佳模型 (依据 OA)
            if saver is not None:
                save_metric = eval_metrics["OA"]
                best_metric, _ = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass

def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args,
                    lr_scheduler=None, saver=None, output_dir=None,
                    amp_autocast=suppress, loss_scaler=None, model_ema=None, mixup_fn=None):
    losses_m = AverageMeter()
    model.train()

    for batch_idx, (input, target) in enumerate(loader):
        input, target = input.cuda(), target.cuda()
        if mixup_fn is not None:
            input, target = mixup_fn(input, target)

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)

        losses_m.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model_parameters(model))
        else:
            loss.backward()
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        if batch_idx % args.log_interval == 0:
            _logger.info(
                "Train: {} [{:>4d}/{} ({:>3.0f}%)]  Loss: {loss.val:#.4g} ({loss.avg:#.3g})  LR: {lr:.3e}".format(
                    epoch, batch_idx, len(loader), 100.0 * batch_idx / len(loader),
                    loss=losses_m, lr=optimizer.param_groups[0]['lr']
                )
            )
    return OrderedDict([("loss", losses_m.avg)])

# =========================================================================
# [新增] 专用验证函数: 计算 OA, AA, Kappa
# =========================================================================
def validate_hsi(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=""):
    losses_m = AverageMeter()
    model.eval()
    
    # 收集所有预测结果
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input, target = input.cuda(), target.cuda()
            with amp_autocast():
                output = model(input)
            
            loss = loss_fn(output, target)
            losses_m.update(loss.item(), input.size(0))
            
            # 获取预测类别 (argmax)
            _, pred = output.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 计算 SOTA 指标
    # 1. OA (Overall Accuracy)
    overall_acc = accuracy_score(all_targets, all_preds) * 100.0
    
    # 2. Kappa Coefficient
    kappa = cohen_kappa_score(all_targets, all_preds) * 100.0
    
    # 3. AA (Average Accuracy)
    # 计算混淆矩阵，对角线元素除以该类总数得到每一类的 Recall/Accuracy，然后求平均
    matrix = confusion_matrix(all_targets, all_preds)
    # 避免除以0 (防止某些类在batch中没出现)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = matrix.diagonal() / matrix.sum(axis=1)
    # 过滤掉 NaN (如果有类别缺失)
    per_class_acc = per_class_acc[~np.isnan(per_class_acc)]
    average_acc = np.mean(per_class_acc) * 100.0

    _logger.info(
        "Test{}:  Loss: {loss.avg:>7.4f}  OA: {oa:>7.2f}%  AA: {aa:>7.2f}%  Kappa: {k:>7.2f}".format(
            log_suffix, loss=losses_m, oa=overall_acc, aa=average_acc, k=kappa
        )
    )

    return OrderedDict([
        ("loss", losses_m.avg), 
        ("OA", overall_acc), 
        ("AA", average_acc), 
        ("Kappa", kappa),
        # 兼容性保留，防止 lr_scheduler 报错
        ("top1", overall_acc),
        ("top5", 0.0) 
    ])

if __name__ == "__main__":
    main()