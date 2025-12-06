import numpy as np
import os
from scipy import signal

# 创建输出数据的目录
os.makedirs(r'Data/em_signals', exist_ok=True)

# 配置参数
SAMPLE_RATE = 100000  # 100k采样率
DURATION = 0.01 # 10ms信号持续时间
SAMPLES_PER_CLASS = 100 # 12类电磁波信号，每类10000个样本
SIGNAL_LEN = int(SAMPLE_RATE * DURATION)  # 每个信号的采样点数

# 电磁波标签定义,根据不同频段和调制方式分类
EM_LABELS = {
    'E0': 'Low_Unmodulated',
    'E1': 'Low_AM',
    'E2': 'Low_FM',
    'E3': 'Low_FSK',
    'E4': 'Mid_Unmodulated',
    'E5': 'Mid_AM',
    'E6': 'Mid_FM',
    'E7': 'Mid_FSK',
    'E8': 'High_Unmodulated',
    'E9': 'High_AM',
    'E10': 'High_FM',
    'E11': 'High_FSK'
}

# 频段中心
FREQ_CENTER = {
    'Low': 5500,       # 5.5kHz
    'Mid': 20000,      # 20kHz
    'High': 40000      # 40kHz
}

# FSK频率偏移
FSK_SHIFT = {
    'Low': 1000,
    'Mid': 2000,
    'High': 3000
}

# ==============================
# 工具函数：生成噪声
# ==============================
def add_noise(signal_data, snr_db):
    """
    根据目标 SNR 添加高斯白噪声
    """
    signal_power = np.mean(signal_data ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal_data))
    return signal_data + noise


# ==============================
# 核心：生成 EM 信号
# ==============================
def generate_em_signal(freq_band, mod_type):
    t = np.linspace(0, DURATION, SIGNAL_LEN, endpoint=False)

    # --- 随机频率偏移（增强多样性） ---
    center_freq = FREQ_CENTER[freq_band]
    center_freq *= (1 + np.random.uniform(-0.05, 0.05))  # ±5% 偏移

    # --- 调制 ---
    if mod_type == "Unmodulated":
        carrier = np.sin(2 * np.pi * center_freq * t)
        return carrier

    elif mod_type == "AM":
        mod_freq = np.random.uniform(50, 200)       # 随机调制频率
        mod_depth = np.random.uniform(0.2, 0.8)     # 随机调制深度
        mod_signal = mod_depth * np.sin(2 * np.pi * mod_freq * t)
        return (1 + mod_signal) * np.sin(2 * np.pi * center_freq * t)

    elif mod_type == "FM":
        mod_freq = np.random.uniform(50, 200)
        freq_deviation = np.random.uniform(200, 1000)  # 随机频偏
        mod_signal = np.sin(2 * np.pi * mod_freq * t)
        inst_freq = center_freq + freq_deviation * mod_signal
        phase = 2 * np.pi * np.cumsum(inst_freq) / SAMPLE_RATE
        return np.sin(phase)

    elif mod_type == "FSK":
        shift = FSK_SHIFT[freq_band]
        shift *= np.random.uniform(0.8, 1.2)  # 随机偏移
        control = np.sign(np.sin(2 * np.pi * 100 * t))  # 100Hz 控制信号
        inst_freq = center_freq + control * shift
        phase = 2 * np.pi * np.cumsum(inst_freq) / SAMPLE_RATE
        return np.sin(phase)

    
# ==============================
# 主程序
# ==============================
TOTAL_SAMPLES = len(EM_LABELS) * SAMPLES_PER_CLASS

X = np.zeros((TOTAL_SAMPLES, SIGNAL_LEN), dtype=np.float32)
y = np.zeros(TOTAL_SAMPLES, dtype=np.int32)

print(f"⚡ 即将生成 {TOTAL_SAMPLES} 条 EM 信号...")
print(f"➡ 每类 {SAMPLES_PER_CLASS} 条")

idx = 0
for label_idx, (label_key, label_name) in enumerate(EM_LABELS.items()):

    freq_band, mod_type = label_name.split("_")
    print(f"\n⏳ 生成类别 {label_key}: {label_name}")

    for _ in range(SAMPLES_PER_CLASS):

        sig = generate_em_signal(freq_band, mod_type)

        # --- 随机 SNR 增强多样性 ---
        snr_db = np.random.uniform(0, 30)  # 0–30 dB
        sig = add_noise(sig, snr_db)

        X[idx] = sig
        y[idx] = label_idx
        idx += 1


# ==============================
# 保存数据
# ==============================
np.save(r'Data/em_signals/X_em_signals.npy', X)
np.save(r'Data/em_signals/y_em_signals.npy', y)

print("\n🎉 电磁波信号生成完成!")
print(f"数据已保存至 Data/em_signals/")
print(f"X shape = {X.shape}, y shape = {y.shape}")