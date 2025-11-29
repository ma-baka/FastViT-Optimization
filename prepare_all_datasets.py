import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# =========================================================
# 全能配置: 支持 IP, PU, SA 三大数据集
# 策略: PCA (30ch) + Small Patch (13x13) + 1:9 Split
# =========================================================

# 基础参数
RAW_DATA_DIR = 'raw_data'
OUTPUT_ROOT = 'data'
PATCH_SIZE = 13      # SOTA 标准小切片
PCA_COMPONENTS = 30  # 降维目标
TEST_SIZE = 0.9      # 1:9 划分

# 数据集配置字典 (文件名 -> 内部变量名)
DATASETS_CONFIG = {
    "IndianPines": {
        "img_file": "Indian_pines_corrected.mat",
        "gt_file": "Indian_pines_gt.mat",
        "img_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt"
    },
    "PaviaU": {
        "img_file": "PaviaU.mat",
        "gt_file": "PaviaU_gt.mat",
        "img_key": "paviaU",
        "gt_key": "paviaU_gt"
    },
    "Salinas": {
        "img_file": "Salinas_corrected.mat",
        "gt_file": "Salinas_gt.mat",
        "img_key": "salinas_corrected",
        "gt_key": "salinas_gt"
    }
}

def apply_pca(data, components):
    """PCA 降维"""
    print(f"   [PCA] Reducing {data.shape[-1]} -> {components} channels...")
    h, w, c = data.shape
    reshaped_data = np.reshape(data, (-1, c))
    pca = PCA(n_components=components, whiten=True)
    new_data = pca.fit_transform(reshaped_data)
    new_data = np.reshape(new_data, (h, w, components))
    return new_data

def create_patches(data, gt, save_folder_name):
    # 1. 归一化 (0-255)
    print("   [Norm] Normalizing data...")
    data = data.astype(np.float32)
    data = ((data - data.min()) / (data.max() - data.min()) * 255.0).astype(np.uint8)

    # 2. 镜像填充
    margin = PATCH_SIZE // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
    
    # 3. 获取样本索引 (排除背景 0)
    indices = [i for i, label in enumerate(gt.ravel()) if label != 0]
    labels = [gt.ravel()[i] for i in indices]
    
    # 4. 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        indices, labels, test_size=TEST_SIZE, stratify=labels, random_state=42
    )
    datasets = {'train': X_train, 'val': X_val}
    
    final_output_dir = os.path.join(OUTPUT_ROOT, save_folder_name)
    print(f"   [Save] Generating patches to: {final_output_dir}")
    
    # 5. 生成并保存
    for split, idx_list in datasets.items():
        for idx in tqdm(idx_list, desc=f"   Processing {split}", leave=False):
            h, w = gt.shape[0], gt.shape[1]
            r, c = idx // w, idx % w
            label = gt[r, c]
            
            patch = padded_data[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            
            # 路径: data/DatasetName_pca30_13x13/train/1/r_c.npy
            save_dir = os.path.join(final_output_dir, split, str(label))
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"{r}_{c}.npy"), patch)

def main():
    if not os.path.exists(RAW_DATA_DIR):
        print(f"[ERROR] 找不到 {RAW_DATA_DIR} 文件夹！")
        return

    # 遍历所有数据集进行处理
    for ds_name, config in DATASETS_CONFIG.items():
        print(f"\n{'='*40}")
        print(f" 处理数据集: {ds_name}")
        print(f"{'='*40}")
        
        img_path = os.path.join(RAW_DATA_DIR, config["img_file"])
        gt_path = os.path.join(RAW_DATA_DIR, config["gt_file"])
        
        # 检查文件是否存在
        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            print(f"[SKIP] 文件缺失，跳过 {ds_name}")
            continue
            
        # 加载
        try:
            data = sio.loadmat(img_path)[config["img_key"]]
            gt = sio.loadmat(gt_path)[config["gt_key"]]
        except KeyError:
            print(f"[ERROR] .mat 键名不匹配，请检查 {ds_name} 的变量名！")
            continue
            
        # PCA
        data_pca = apply_pca(data, PCA_COMPONENTS)
        
        # 切片并保存
        # 文件夹命名格式: PaviaU_pca30_13x13
        folder_name = f"{ds_name}_pca{PCA_COMPONENTS}_{PATCH_SIZE}x{PATCH_SIZE}"
        create_patches(data_pca, gt, folder_name)
        
        print(f"{ds_name} 处理完成！")

if __name__ == '__main__':
    main()