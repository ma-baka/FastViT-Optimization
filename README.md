## 2025-11-20 (Day 1): 环境搭建与模型初步探索

### 1. 环境与数据排坑 (Setup & Debugging)
- **平台**: AutoDL (RTX 4090 D / 24GB VRAM)
- **环境修复**:
  - 解决 NumPy 2.x 与 PyTorch 1.11 不兼容报错，降级 NumPy。
  - 解决 DataLoader 报错：发现并删除了数据集中隐藏的 .ipynb_checkpoints 文件夹。
- **数据**: 成功配置 Imagenette (10类) 数据集。
- **工具**: 配置 GitHub SSH Key 实现代码同步；配置 .gitignore 过滤大文件。

### 2. 实验 A: 快速试错 (Baseline)
- **模型**: fastvit_t8 (Tiny版)
- **参数**: Batch Size 128, 从零训练 (Scratch)。
- **现象**:
  - 训练初期准确率极低 (~10%)。
  - **WandB 可视化**: 成功接入 WandB，观察到 train_loss 呈下降趋势，验证代码无 Bug，但模型收敛缓慢。

## 3. 实验 B: 性能压榨与模型升级 (Optimization)
- **模型**: `fastvit_ma36`
- **配置**: 
  - Batch Size: 64
  - Epochs: 100 + 10 (Cooldown)
  - 策略: 从零训练 (Training from Scratch)
- **最终结果 (Epoch 109)**:
  - **Top-1 准确率**: **87.18%** 🚀
  - **Top-5 准确率**: 98.73%
  - **Eval Loss**: 0.594
- **结果分析**:
  - 相比 T8 模型 (~10%) 提升巨大，证明了 MA36 架构在特征提取上的强悍能力。
  - 曲线显示在最后 10 个 Cooldown Epochs 阶段，准确率有显著拉升，证明移除数据增强后的微调策略非常有效。
  - `Eval Loss` (0.59) 远低于 `Train Loss` (1.60)，说明强数据增强（Mixup/Cutmix）有效防止了过拟合，模型泛化能力极强。

### 下一步计划
- 观察 MA36 在无预训练情况下的收敛极限。
- 计划引入 ImageNet 预训练权重进行微调 (Fine-tuning)，以大幅提升准确率。