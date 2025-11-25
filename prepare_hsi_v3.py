import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# === Day 5 配置: 波段选择 (Band Selection) ===
DATA_PATH = 'raw_data/Indian_pines_corrected.mat'
GT_PATH = 'raw_data/Indian_pines_gt.mat'
# [修改] 输出目录改名，以示区分
OUTPUT_DIR = 'data/indian_pines_bands30_npy' 
PATCH_SIZE = 32
TARGET_BANDS = 30   # 目标通道数
TEST_SIZE = 0.9     # 10% 训练

def select_bands(data, num_bands):
    """
    均匀波段选择 (Uniform Band Selection)
    保留光谱的物理顺序，从 200 个波段中均匀抽取 30 个
    """
    total_bands = data.shape[2]
    # 生成均匀间隔的索引，例如 [0, 6, 12, ..., 198]
    selected_indices = np.linspace(0, total_bands - 1, num_bands, dtype=int)
    
    print(f"[INFO] 原始波段数: {total_bands}")
    print(f"[INFO] 选取的波段索引: {selected_indices}")
    
    new_data = data[:, :, selected_indices]
    return new_data

def create_patches_npy(data, gt, output_dir):
    # 归一化 (Min-Max) 到 0-255，方便统一处理
    # 注意：这里也可以选择不转 uint8 直接存 float，但为了和之前保持一致，先转 uint8
    data = data.astype(np.float32)
    data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

    # 镜像填充
    margin = PATCH_SIZE // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
    
    # 获取样本坐标
    indices = [i for i, label in enumerate(gt.ravel()) if label != 0]
    labels = [gt.ravel()[i] for i in indices]
    
    # 划分数据集 (1:9)
    X_train, X_val, y_train, y_val = train_test_split(indices, labels, test_size=TEST_SIZE, stratify=labels, random_state=42)
    datasets = {'train': X_train, 'val': X_val}
    
    print(f"[INFO] 开始生成 {TARGET_BANDS}通道 (Raw Bands) 数据...")
    print(f"输出目录: {output_dir}")
    
    for split, idx_list in datasets.items():
        for idx in tqdm(idx_list, desc=split):
            h, w = gt.shape[0], gt.shape[1]
            r, c = idx // w, idx % w
            label = gt[r, c]
            
            # 提取 Patch
            patch = padded_data[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            
            # 保存
            save_dir = os.path.join(output_dir, split, str(label))
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"{r}_{c}.npy"), patch)

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        print("[ERROR] 找不到 raw_data，请先下载 Indian Pines 数据集！")
    else:
        print("[1/3] Loading data...")
        data = sio.loadmat(DATA_PATH)['indian_pines_corrected']
        gt = sio.loadmat(GT_PATH)['indian_pines_gt']
        
        print(f"[2/3] Selecting {TARGET_BANDS} Bands (Preserving Spectral Order)...")
        # [修改] 这里不再调用 PCA，而是调用 select_bands
        data_selected = select_bands(data, TARGET_BANDS)
        
        print("[3/3] Generating .npy patches...")
        create_patches_npy(data_selected, gt, OUTPUT_DIR)
        print("[Done] Band Selection data ready.")