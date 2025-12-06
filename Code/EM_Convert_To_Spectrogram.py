import numpy as np
from scipy import signal
import os
from tqdm import tqdm

# ================================
# 参数定义
# ================================
RAW_SIGNAL_PATH = r'Data/em_signals/X_em_signals.npy'
RAW_LABEL_PATH = r'Data/em_signals/y_em_signals.npy'

SAVE_SPEC_PATH = r'Data/em_signals/em_spectrograms.dat'
SAVE_LABEL_PATH = r'Data/em_signals/em_spectrograms_labels.npy'

BATCH_SIZE = 500
SAMPLE_RATE = 100000

# STFT 参数
N_FFT = 256
N_PERSEG = 32
N_OVERLAP = 28

# 频谱图尺寸
TARGET_TIME_FRAMES = 128
FREQ_BINS = N_FFT // 2 + 1  # 129

# ================================
# 加载原始数据
# ================================
print("Loading raw EM signals...")
X_raw = np.load(RAW_SIGNAL_PATH, mmap_mode='r')
y_raw = np.load(RAW_LABEL_PATH)

TOTAL = X_raw.shape[0]
print(f"Total samples: {TOTAL}")

# ================================
# 创建 memmap 输出文件
# ================================
print("Allocating memmap array for spectrograms...")
spectrograms = np.memmap(
    SAVE_SPEC_PATH,
    dtype=np.float32,
    mode='w+',
    shape=(TOTAL, 1, FREQ_BINS, TARGET_TIME_FRAMES)
)

# ================================
# 批处理生成频谱图
# ================================
print("Generating spectrograms in batches...")

for start in tqdm(range(0, TOTAL, BATCH_SIZE)):
    end = min(start + BATCH_SIZE, TOTAL)
    batch = X_raw[start:end]

    for i in range(len(batch)):
        f, t, Zxx = signal.spectrogram( # 使用的是STFT:短时傅里叶变换
            batch[i],
            fs=SAMPLE_RATE,
            nperseg=N_PERSEG,
            noverlap=N_OVERLAP,
            nfft=N_FFT,
            scaling='spectrum',
            mode='magnitude'     # 幅度谱
        )

        Zxx_db = 10 * np.log10(Zxx + 1e-10)

        if Zxx_db.shape[1] < TARGET_TIME_FRAMES:
            pad = TARGET_TIME_FRAMES - Zxx_db.shape[1]
            Zxx_db = np.pad(Zxx_db, ((0, 0), (0, pad)), 'constant')
        else:
            Zxx_db = Zxx_db[:, :TARGET_TIME_FRAMES]

        spectrograms[start + i] = Zxx_db.astype(np.float32)[np.newaxis, :, :]

# ================================
# 归一化（正确写回 memmap）
# ================================
print("Computing normalization statistics...")
mean = float(np.mean(spectrograms))
std = float(np.std(spectrograms))

print(f"Global mean = {mean:.5f}, std = {std:.5f}")

print("Applying normalization...")
spectrograms[:] = (spectrograms[:] - mean) / std  # ✔ 正确方式
spectrograms.flush()                               # ✔ 保存到磁盘

# ================================
# 保存标签
# ================================
print("Saving labels...")
np.save(SAVE_LABEL_PATH, y_raw)
print("\n✅ All done!")
print(f"Spectrograms saved to: {SAVE_SPEC_PATH}")
print(f"Labels saved to: {SAVE_LABEL_PATH}")
