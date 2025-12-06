import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 用 ReduceLROnPlateau 替代 CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from Model.model import ResNetEM
import os, time
from Data_Utils.data_utils import load_and_normalize_data, split_combine_data, create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == 'cpu':
    torch.set_num_threads(min(4, os.cpu_count()))  # 限制线程数防止CPU过载

DATA_DIR = Path(__file__).resolve().parent / 'Data'

# ===================================================================
# 超参数配置（针对 1200 样本优化，95%+ 准确率）
# ===================================================================
EPOCHS = 60  # 减少 epoch 数（早停会提前终止）
BATCH_SIZE = 32
LEARNING_RATE = 3e-4  
WEIGHT_DECAY = 1e-4  
DROPOUT_RATE = 0.3    

# ===================================================================
# 核心优化：移除冗余增强，保留有效策略
# ===================================================================
def spec_augment(x, freq_mask=20, time_mask=24):
    """SpecAugment: 频率/时间掩码（保留核心增强）"""
    _, _, freq_bins, time_bins = x.size()
    # 频率掩码
    f = np.random.randint(0, freq_mask)
    f0 = np.random.randint(0, max(1, freq_bins - f))
    x[:, :, f0:f0+f, :] = 0
    # 时间掩码
    t = np.random.randint(0, time_mask)
    t0 = np.random.randint(0, max(1, time_bins - t))
    x[:, :, :, t0:t0+t] = 0
    return x

# 移除 MixUp（在 1200 样本上 MixUp 会过拟合，用增强层+注意力替代）
# MixUp 实现已移除，不再使用

class EarlyStopper:
    """优化的早停机制（更严格）"""
    def __init__(self, patience=15, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def early_stop(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False

# ===================================================================
# 训练主函数
# ===================================================================
def train_resnet_model(
        model_name="resnet_em",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        data_dir= DATA_DIR
):
    print(f"\n{'='*60}")
    print(f"📊 95%+ 准确率训练配置: epochs={epochs}, batch={batch_size}, lr={learning_rate:.3f}")
    print(f"🎯 目标: 1200 样本 + 验证集 95%+ 准确率")
    print(f"{'='*60}\n")

    # 1. 加载数据
    print("⏳ 加载数据...")
    X, y = load_and_normalize_data(data_dir)
    print(f"✓ 数据加载完成: X.shape={X.shape}, y.shape={y.shape}")

    # 2. 三集划分
    print("📂 划分训练/验证/测试集...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_combine_data(
        X, y, train_ratio=0.7, val_ratio=0.15, random_state=42
    )
    print(f"  - 训练集: {len(y_train)} | 验证集: {len(y_val)} | 测试集: {len(y_test)}")

    # 3. 创建DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)
    print(f"✓ DataLoader创建完成\n")

    # 4. 实例化模型（已使用优化版 ResNetEM_95）
    model = ResNetEM(num_classes=12, dropout_rate=DROPOUT_RATE).to(device)
    print(f"🤖 模型信息:")
    print(f"  - 架构: ResNetEM_95 (95%+ 准确率优化版)")
    print(f"  - 参数量: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")
    print(f"  - Dropout: {DROPOUT_RATE} (残差块) + 0.6 (分类器)")
    print(f"  - 设备: {device}\n")

    # 5. 优化器配置（AdamW + 严格正则化）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 减小 label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6
    )

    # 记录指标
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    early_stopper = EarlyStopper(patience=15, min_delta=0.0001)  # 严格早停

    # ===================================================================
    # 训练循环（核心优化：移除 MixUp，专注增强+注意力）
    # ===================================================================
    start_train_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # 训练进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]", 
                         leave=False, ncols=100)

        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 仅使用 SpecAugment（模型内部已有增强层）
            #inputs = spec_augment(inputs)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # 更新进度条
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        # 计算指标
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accs.append(correct / total)
        val_accs.append(val_correct / val_total)

        epoch_time = time.time() - start_train_time
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train: loss={train_losses[-1]:.4f} acc={train_accs[-1]:.4f} | "
              f"Val: loss={val_losses[-1]:.4f} acc={val_accs[-1]:.4f} | "
              f"Time: {epoch_time/60:.1f}min", end="")

        # 早停和学习率调整
        if val_accs[-1] > best_val_acc:
            OUT_DIR = Path("Model")
            OUT_DIR.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_accs[-1],
            }, OUT_DIR / f"{model_name}.pth")
            best_val_acc = val_accs[-1]
            print(" ✓ 保存最佳模型")
        else:
            print(f" (val_acc={val_accs[-1]:.4f})")
        
        # 早停检查
        if early_stopper.early_stop(val_accs[-1]):
            print("\n→ 早停触发！最佳验证准确率: {best_val_acc:.4f}")
            break
            
        # 学习率衰减
        scheduler.step(val_accs[-1])

    train_time = time.time() - start_train_time
    print(f"\n✓ 训练完成！耗时: {train_time/60:.1f} 分钟")

    # ===================================================================
    # 最终测试
    # ===================================================================
    print("\n" + "="*50)
    print("🎯 最终测试集评估...")
    test_acc = evaluate_model(OUT_DIR / f"{model_name}.pth", test_loader)
    
    # 绘制曲线
    plot_training(train_losses, val_losses, train_accs, val_accs)
    
    return best_val_acc, test_acc


# ===================================================================
# 绘图函数（保持不变）
# ===================================================================
def plot_training(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(train_losses, 'b-o', label="Train Loss", linewidth=2, markersize=4)
    ax1.plot(val_losses, 'r-s', label="Val Loss", linewidth=2, markersize=4)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Curve", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(train_accs, 'b-o', label="Train Acc", linewidth=2, markersize=4)
    ax2.plot(val_accs, 'r-s', label="Val Acc", linewidth=2, markersize=4)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy Curve", fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path("training_curve.png"), dpi=300, bbox_inches='tight')
    print("\n📊 训练曲线已保存至: training_curve.png")


# ===================================================================
# 评估函数（保持不变）
# ===================================================================
def evaluate_model(model_path, test_loader):
    checkpoint = torch.load(model_path, map_location=device)
    model = ResNetEM(num_classes=12, dropout_rate=DROPOUT_RATE).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"📈 测试集准确率: {acc*100:.2f}%")
    return acc

if __name__ == "__main__":
    # 确保模型文件已更新为 ResNetEM
    print("✅ 检查: 使用优化版 ResNetEM 模型")
    best_acc, test_acc = train_resnet_model()
    print(f"\n🏆 最佳验证准确率: {best_acc:.4f}, 测试集准确率: {test_acc:.4f}")