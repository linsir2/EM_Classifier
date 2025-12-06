import torch
from Model.model_train import train_resnet_model
from pathlib import Path

def main():
    print("=" * 60)
    print("  电磁波信号分类 - 深度学习训练系统")
    print("=" * 60)
    print(f"📟 设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")

    DATA_DIR = Path(__file__).resolve().parent / 'Data'

    # ================================
    # 训练配置（优化版）
    # ================================
    MODEL_NAME = "resnet_em"
    EPOCHS = 60  # 减少epoch，早停会提前结束
    BATCH_SIZE = 32  # CPU友好
    LEARNING_RATE = 1e-4  # 推荐学习率

    # ================================
    # 训练 ResNetEM 模型
    # ================================
    print("🔥 开始训练 ResNetEM 模型...")
    print(f"   配置: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}\n")
    
    best_val_acc, test_acc = train_resnet_model(
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        data_dir=DATA_DIR
    )

    # ================================
    # 训练报告
    # ================================
    print("\n" + "=" * 60)
    print("🏆 训练完成报告")
    print(f"   最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"   测试集准确率:   {test_acc*100:.2f}%")
    print("=" * 60)
    
    print("\n✨ 模型文件: Model/resnet_em.pth")
    print("📊 训练曲线: training_curve.png")
    print("\n🎉 模型已准备好部署与推理！")


if __name__ == "__main__":
    main()