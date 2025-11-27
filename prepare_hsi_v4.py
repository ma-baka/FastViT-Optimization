import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# ==========================================
# Day 7 配置: Small Patch (13x13)
# 策略: 原始波段选择 + 微型切片
# ==========================================
DATA_PATH = 'raw_data/Indian_pines_corrected.mat'
GT_PATH = 'raw_data/Indian_pines_gt.mat'
OUTPUT_DIR = 'data/indian_pines_bands30_13x13' # 新的数据存放点
PATCH_SIZE = 13     # <--- [核心修改] 从 32 改为 13
TARGET_BANDS = 30   # 保持 30 个原始波段
TEST_SIZE = 0.9     # 1:9 划分

def select_bands(data, num_bands):
    """均匀波段选择"""
    total_bands = data.shape[2]
    selected_indices = np.linspace(0, total_bands - 1, num_bands, dtype=int)
    print(f"[INFO] Selected Bands: {selected_indices}")
    return data[:, :, selected_indices]

def create_patches_npy(data, gt, output_dir):
    # 归一化
    print("[INFO] Normalizing data...")
    data = data.astype(np.float32)
    data = ((data - data.min()) / (data.max() - data.min()) * 255.0).astype(np.uint8)

    # 镜像填充 (Padding)
    # 13 // 2 = 6，边缘填充 6 个像素
    margin = PATCH_SIZE // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
    
    indices = [i for i, label in enumerate(gt.ravel()) if label != 0]
    labels = [gt.ravel()[i] for i in indices]
    
    # 1:9 划分
    X_train, X_val, y_train, y_val = train_test_split(
        indices, labels, test_size=TEST_SIZE, stratify=labels, random_state=42
    )
    datasets = {'train': X_train, 'val': X_val}
    
    print(f"[INFO] Generating {PATCH_SIZE}x{PATCH_SIZE} patches...")
    
    for split, idx_list in datasets.items():
        for idx in tqdm(idx_list, desc=split):
            h, w = gt.shape[0], gt.shape[1]
            r, c = idx // w, idx % w
            label = gt[r, c]
            
            # 提取 Patch (13, 13, 30)
            patch = padded_data[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            
            save_dir = os.path.join(output_dir, split, str(label))
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"{r}_{c}.npy"), patch)

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        print("[ERROR] 找不到 raw_data！")
    else:
        data = sio.loadmat(DATA_PATH)['indian_pines_corrected']
        gt = sio.loadmat(GT_PATH)['indian_pines_gt']
        
        # 1. 波段选择
        data_selected = select_bands(data, TARGET_BANDS)
        
        # 2. 生成切片
        create_patches_npy(data_selected, gt, OUTPUT_DIR)
        print("Done! 13x13 Data Ready.")