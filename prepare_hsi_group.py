import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# =========================================================================
# Day 6 配置: 分组 PCA (Group-wise PCA)
# 策略: 将 200 个波段分为 30 组，每组内部 PCA 降维至 1 维
# 目的: 既保留全波段信息，又保持光谱序列性，完美适配 RepSS-Mixer
# =========================================================================

DATA_PATH = 'raw_data/Indian_pines_corrected.mat'
GT_PATH = 'raw_data/Indian_pines_gt.mat'
OUTPUT_DIR = 'data/indian_pines_group_pca30_npy' # 新目录
PATCH_SIZE = 32
TARGET_CHANNELS = 30  # 最终输出通道数
TEST_SIZE = 0.9       # 1:9 划分 (SOTA 标准)

def apply_group_pca(data, target_channels):
    """
    执行分组 PCA。
    输入: (H, W, C_in=200)
    输出: (H, W, C_out=30)
    """
    h, w, c = data.shape
    
    # 1. 将 200 个波段索引划分为 30 个连续的组
    # 例如: [0,1,2,3,4,5,6], [7,8,9,10,11,12] ...
    band_indices = np.arange(c)
    groups = np.array_split(band_indices, target_channels)
    
    print(f"[INFO] Performing Group-wise PCA: {c} bands -> {target_channels} groups")
    
    processed_bands = []
    
    for i, group in enumerate(tqdm(groups, desc="Processing Groups")):
        # 取出当前组的波段数据 (H, W, k)
        group_data = data[:, :, group]
        
        # Reshape for PCA: (N, k)
        flat_data = group_data.reshape(-1, group_data.shape[-1])
        
        # 组内 PCA: 提取 1 个主成分
        pca = PCA(n_components=1, whiten=True)
        reduced_data = pca.fit_transform(flat_data)
        
        # 还原形状 (H, W, 1)
        restored_data = reduced_data.reshape(h, w, 1)
        processed_bands.append(restored_data)
        
    # 将 30 个组的结果在通道维度拼接: (H, W, 30)
    final_data = np.concatenate(processed_bands, axis=-1)
    
    return final_data

def create_patches_npy(data, gt, output_dir):
    # 归一化 (Min-Max -> 0-255)
    # 保持光谱曲线的相对形状
    print("[INFO] Normalizing data...")
    data = data.astype(np.float32)
    data = ((data - data.min()) / (data.max() - data.min()) * 255.0).astype(np.uint8)

    # 镜像填充
    margin = PATCH_SIZE // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
    
    # 获取样本索引
    indices = [i for i, label in enumerate(gt.ravel()) if label != 0]
    labels = [gt.ravel()[i] for i in indices]
    
    # SOTA 标准划分 (10% 训练)
    X_train, X_val, y_train, y_val = train_test_split(
        indices, labels, 
        test_size=TEST_SIZE, 
        stratify=labels, 
        random_state=42
    )
    datasets = {'train': X_train, 'val': X_val}
    
    print(f"[INFO] Generating Patches to {output_dir}...")
    
    for split, idx_list in datasets.items():
        for idx in tqdm(idx_list, desc=split):
            h, w = gt.shape[0], gt.shape[1]
            r, c = idx // w, idx % w
            label = gt[r, c]
            
            # 提取 Patch (32, 32, 30)
            patch = padded_data[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            
            # 保存
            save_dir = os.path.join(output_dir, split, str(label))
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"{r}_{c}.npy"), patch)

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        print("[ERROR] 找不到 raw_data，请先下载数据！")
    else:
        print("[1/3] Loading Raw Data...")
        data = sio.loadmat(DATA_PATH)['indian_pines_corrected']
        gt = sio.loadmat(GT_PATH)['indian_pines_gt']
        
        # 分组 PCA
        print("[2/3] Spectral Reduction (Group-wise PCA)...")
        data_group_pca = apply_group_pca(data, TARGET_CHANNELS)
        print(f"     Shape after reduction: {data_group_pca.shape}")
        
        # 切片
        print("[3/3] Creating Patches...")
        create_patches_npy(data_group_pca, gt, OUTPUT_DIR)
        print("Done! Group-wise PCA data ready.")