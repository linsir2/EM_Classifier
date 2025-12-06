import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_and_normalize_data(data_dir: str | Path):
    """
    加载电磁波频谱图数据和标签(memmap 模式，防止大数据 OOM)
    """
    data_dir = Path(data_dir).resolve().parent.parent / 'Data'
    spec_path = data_dir / 'em_signals' / 'em_spectrograms.dat'
    label_path = data_dir / 'em_signals' / 'em_spectrograms_labels.npy'

    if not spec_path.exists() or not label_path.exists():
        raise FileNotFoundError(f'❌ 找不到 {spec_path} 或 {label_path}')
    
    # 动态读取形状（避免硬编码）
    em_labels = np.load(label_path)
    TOTAL = len(em_labels)
    
    # 从.memmap文件推断形状
    em_spectrograms = np.memmap(
        spec_path, dtype=np.float32, mode='r'
    )
    # 根据已知维度重新reshape
    em_spectrograms = em_spectrograms.reshape(TOTAL, 1, 129, 128)

    em_labels = np.load(label_path)

    return em_spectrograms, em_labels


def split_combine_data(em_data, em_labels, train_ratio=0.7, val_ratio=0.15, random_state=42):
    """
    划分数据并转换为 Tensor。
    注意:em_data 是 memmap,不能一次性转成 torch.tensor(会 OOM),
    所以需要在划分后再转。
    """

    # 第一步：用 sklearn 做划分
    idx_train, idx_temp = train_test_split(
        np.arange(len(em_labels)),
        train_size=train_ratio,
        stratify=em_labels,
        random_state=random_state
    )
    idx_val, idx_test = train_test_split(
        idx_temp, train_size=val_ratio/(1-train_ratio),
        stratify=em_labels[idx_temp], random_state=random_state
    )

    # 转换为 Tensor
    X_train = torch.tensor(em_data[idx_train], dtype=torch.float32)
    X_val = torch.tensor(em_data[idx_val], dtype=torch.float32)
    X_test = torch.tensor(em_data[idx_test], dtype=torch.float32)
    y_train = torch.tensor(em_labels[idx_train], dtype=torch.long)
    y_val = torch.tensor(em_labels[idx_val], dtype=torch.long)
    y_test = torch.tensor(em_labels[idx_test], dtype=torch.long)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    """
    创建小型数据集的 DataLoader
    """
    print("\nCreating DataLoaders for small dataset...")
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train loader: {len(train_loader)} batches | Batch size: {batch_size}")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader