#
# Training Script for FastViT with RepSS-Mixer (Day 4 Architecture)
# [Pro Version] 标准化 SOTA 训练脚本
# 集成: Mixup, Cutmix, Model EMA, Native AMP, Cosine Schedule
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

import torch
import torch.nn as nn
import torchvision.utils
from torch.utils.data import Dataset, DataLoader

# 引入 timm 的高级数据增强和模型工具
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

# [关键] 导入 Day 4 的新模型定义文件 (RepSS-Mixer)
# 这会自动将 fastvit_ma36 注册到 timm 中
import models.fastvit_repss 

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("train")

# =========================================================================
# 自定义 HSI 数据集类 (保持不变)
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
# 参数定义 (已清理冗余参数)
# =========================================================================
config_parser = parser = argparse.ArgumentParser(description="Training Config", add_help=False)
parser.add_argument("-c", "--config", default="", type=str, metavar="FILE", help="YAML config file")

parser = argparse.ArgumentParser(description="PyTorch HSI Training (RepSS-Mixer Pro)")

# Dataset
parser.add_argument("--data-dir", metavar="DIR", default="./data/indian_pines_pca30_npy", help="path to dataset")
parser.add_argument("--train-split", metavar="NAME", default="train")
parser.add_argument("--val-split", metavar="NAME", default="val")

# Model Parameters
parser.add_argument("--model", default="fastvit_ma36", type=str, metavar="MODEL")
parser.add_argument("--initial-checkpoint", default="", type=str, metavar="PATH", help="Pretrained weights")
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

# Augmentation (Mixup/Cutmix)
parser.add_argument("--no-aug", action="store_true", default=True)
parser.add_argument("--mixup", type=float, default=0.0, help="mixup alpha")
parser.add_argument("--cutmix", type=float, default=0.0, help="cutmix alpha")
parser.add_argument("--cutmix-minmax", type=float, nargs="+", default=None)
parser.add_argument("--mixup-prob", type=float, default=1.0)
parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
parser.add_argument("--mixup-mode", type=str, default="batch")
parser.add_argument("--mixup-off-epoch", default=0, type=int)
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
parser.add_argument("--eval-metric", default="top1", type=str)
parser.add_argument("--log-interval", type=int, default=50)
parser.add_argument("--log-wandb", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint-hist", type=int, default=10)
parser.add_argument("--no-prefetcher", action="store_true", default=False)

# [删除] Distillation 相关参数
# 原因: 我们只进行微调(Fine-tuning)，不进行在线蒸馏(Online Distillation)
# 已删除: --teacher-model, --teacher-path, --distillation-type 等

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    return args

def accuracy(output, target, topk=(1,)):
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

def main():
    setup_default_logging()
    args = _parse_args()

    if args.log_wandb and has_wandb:
        project_name = os.environ.get("WANDB_PROJECT", args.experiment)
        wandb.init(project=project_name, config=args)

    args.prefetcher = not args.no_prefetcher
    args.distributed = False 
    args.device = "cuda:0"
    
    random_seed(args.seed, 0)

    # =================================================================
    # 创建模型 (使用 create_model, 已修复参数传递 Bug)
    # =================================================================
    _logger.info(f"Creating model {args.model} with in_chans=30")
    
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
        checkpoint_path="", # [关键] 手动加载权重，避免自动加载报错
        in_chans=30,        # [关键] 30通道
        strict=False
    )

    # =================================================================
    # 智能权重加载 (Smart Checkpoint Loading)
    # =================================================================
    if args.initial_checkpoint and os.path.exists(args.initial_checkpoint):
        print(f"[INFO] Loading checkpoint: {args.initial_checkpoint}")
        checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint: checkpoint = checkpoint['state_dict']
        
        model_dict = model.state_dict()
        # 过滤: 形状不匹配的层 (如 RepSSMixer 替换了 RepMixer，部分权重无法加载)
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        print(f"[INFO] Successfully loaded {len(pretrained_dict)} layers.")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    else:
        print("[WARNING] No checkpoint loaded, training from scratch.")

    model.cuda()

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # AMP (自动混合精度)
    amp_autocast = suppress
    loss_scaler = None
    if args.amp and torch.cuda.is_available():
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        _logger.info("Using Native AMP.")

    # Model EMA (影子模型)
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay)

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = args.start_epoch or 0

    dataset_train = HSIDataset(args.data_dir, split=args.train_split)
    dataset_eval = HSIDataset(args.data_dir, split=args.val_split)

    # Mixup (数据增强)
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

    eval_metric = args.eval_metric
    best_metric = None
    saver = None
    if args.experiment:
        output_dir = os.path.join("./output", args.experiment)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        decreasing = True if eval_metric == "loss" else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema,
            amp_scaler=loss_scaler, checkpoint_dir=output_dir, max_history=args.checkpoint_hist,
            decreasing=decreasing
        )

    try:
        for epoch in range(start_epoch, num_epochs):
            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=None,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler,
                model_ema=model_ema, mixup_fn=mixup_fn
            )

            # [关键] 分开验证主模型和 EMA 模型
            eval_metrics = validate(
                model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast
            )

            ema_eval_metrics = None
            if model_ema is not None:
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, 
                    amp_autocast=amp_autocast, log_suffix=" (EMA)"
                )

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            # [关键] 分开记录日志，避免曲线混淆
            if args.log_wandb and has_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics['loss'],
                    "lr": optimizer.param_groups[0]['lr'],
                    "eval_loss": eval_metrics['loss'],
                    "eval_top1": eval_metrics['top1'],
                    "eval_top5": eval_metrics['top5'],
                }
                if ema_eval_metrics:
                    log_dict.update({
                        "eval_loss_ema": ema_eval_metrics['loss'],
                        "eval_top1_ema": ema_eval_metrics['top1'],
                        "eval_top5_ema": ema_eval_metrics['top5'],
                    })
                wandb.log(log_dict)

            if saver is not None:
                saver.save_checkpoint(epoch, metric=eval_metrics[eval_metric])

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

def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=""):
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input, target = input.cuda(), target.cuda()
            with amp_autocast():
                output = model(input)
            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses_m.update(loss.item(), input.size(0))
            top1_m.update(acc1.item(), input.size(0))
            top5_m.update(acc5.item(), input.size(0))
    _logger.info(
        "Test{}:  Loss: {loss.avg:>7.4f}  Acc@1: {top1.avg:>7.4f}".format(
            log_suffix, loss=losses_m, top1=top1_m
        )
    )
    return OrderedDict([("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)])

if __name__ == "__main__":
    main()