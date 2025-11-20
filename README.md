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

### 3. 实验 B: 性能压榨与模型升级 (Optimization)
- **决策**: 放弃小模型 T8，转向参数量更大、性能更强的 fastvit_ma36。
- **硬件分析 (nvidia-smi)**:
  - 观察到显存占用与 Batch Size 的关系。
  - **最终配置**:
    - **Batch Size**: 调整为 64。显存占用约 24GB (97%)，达到硬件极限。
    - **Workers**: 设置为 8。GPU 利用率稳定在 96%，数据加载无瓶颈。
- **当前状态**: 正在从零训练 fastvit_ma36，等待模型收敛。

### 下一步计划
- 观察 MA36 在无预训练情况下的收敛极限。
- 计划引入 ImageNet 预训练权重进行微调 (Fine-tuning)，以大幅提升准确率。