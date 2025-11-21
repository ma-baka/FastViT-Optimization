import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# === Day 3 配置: 进阶学术标准 ===
DATA_PATH = 'raw_data/Indian_pines_corrected.mat'
GT_PATH = 'raw_data/Indian_pines_gt.mat'
OUTPUT_DIR = 'data/indian_pines_pca30_npy'  # 名字改了，标识 npy 格式
PATCH_SIZE = 32       # 保持 32x32 适配 FastViT 下采样
PCA_COMPONENTS = 30   # <--- 核心改变: 保留 30 个光谱主成分
TEST_SIZE = 0.9       # 10% 训练 (学术硬核模式)

def apply_pca(data, components):
    """PCA 降维: (H, W, 200) -> (H, W, 30)"""
    new_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=components, whiten=True)
    new_data = pca.fit_transform(new_data)
    new_data = np.reshape(new_data, (data.shape[0], data.shape[1], components))
    return new_data

def create_patches_npy(data, gt, output_dir):
    # Padding (镜像填充，防止边缘信息丢失)
    margin = PATCH_SIZE // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
    
    # 获取所有样本坐标
    indices = [i for i, label in enumerate(gt.ravel()) if label != 0]
    labels = [gt.ravel()[i] for i in indices]
    
    # 划分数据集 (1:9)
    X_train, X_val, y_train, y_val = train_test_split(indices, labels, test_size=TEST_SIZE, stratify=labels, random_state=42)
    datasets = {'train': X_train, 'val': X_val}
    
    print(f"开始生成 30通道 .npy 数据...")
    print(f"训练集: {len(X_train)} | 测试集: {len(X_val)}")
    print(f"输出目录: {output_dir}")
    
    for split, idx_list in datasets.items():
        for idx in tqdm(idx_list, desc=split):
            h, w = gt.shape[0], gt.shape[1]
            r, c = idx // w, idx % w
            label = gt[r, c]
            
            # 提取 Patch (32, 32, 30)
            patch = padded_data[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            
            # 保存为 .npy 文件 (不再是 jpg)
            # 结构: data/train/class_id/r_c.npy
            save_dir = os.path.join(output_dir, split, str(label))
            os.makedirs(save_dir, exist_ok=True)
            
            np.save(os.path.join(save_dir, f"{r}_{c}.npy"), patch)

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        print("找不到 raw_data，请先下载 Indian Pines 数据集！")
    else:
        print("1. Loading data...")
        data = sio.loadmat(DATA_PATH)['indian_pines_corrected']
        gt = sio.loadmat(GT_PATH)['indian_pines_gt']
        
        print(f"2. PCA Reducing: {data.shape[-1]} -> {PCA_COMPONENTS} channels...")
        data_pca = apply_pca(data, PCA_COMPONENTS)
        
        print("3. Generating .npy patches...")
        create_patches_npy(data_pca, gt, OUTPUT_DIR)
        print("Done! 30-channel data ready.")