import numpy as np
import matplotlib.pyplot as plt

A = 1                       # Signal amplitude
fm = 5                      # Message signal frequency (Hz)
duration = 1                # Signal duration (s)

# Continuous-time samples (for smooth plotting)
fs_cont = 1000
t = np.linspace(0, duration, fs_cont)
x = A * np.sin(2 * np.pi * fm * t)

nyquist_rate = 2 * fm
fs = 40           # Sampling frequency (20 > 10)

# Sampled Signal
ts = np.arange(0, duration, 1 / fs)
xs = A * np.sin(2 * np.pi * fm * ts)

# Uniform Quantization
bits = 3
L = 2 ** bits

xmin = -A
xmax = A
step = (xmax - xmin) / (L - 1)
indices = np.round((xs - xmin) / step)
indices = np.clip(indices, 0, L - 1).astype(int)

quantized = xmin + indices * step

# PCM Encoding
binary_codes = [format(i, f'0{bits}b') for i in indices]

decoded = xmin + indices * step

reconstructed = np.interp(t, ts, decoded)

plt.figure(figsize=(14, 12))
plt.subplot(4, 1, 1)
plt.plot(t, x, 'b', linewidth=2)
plt.title("Original Continuous Signal")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, x, color='lightgray')
plt.stem(ts, xs, linefmt='r-', markerfmt='ro', basefmt=' ')
plt.title("Sampled Signal")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.step(ts, quantized, where='mid', color='green', linewidth=2)
plt.scatter(ts, quantized, color='red', s=25)
for level in np.arange(xmin, xmax, step):
    plt.axhline(level, color='gray', linewidth=0.3)
plt.title("Quantized Signal")
plt.ylabel("Amplitude")
plt.grid(True)

# ----------------------------------------------------------
# 4. Decoded Signal
# ----------------------------------------------------------
# plt.subplot(5, 1, 4)
# plt.step(ts, decoded, where='mid', color='purple', linewidth=2)
# plt.scatter(ts, decoded, color='black', s=20)
# plt.title("Decoded Signal")
# plt.ylabel("Amplitude")
# plt.grid(True)

# ----------------------------------------------------------
# 5. Reconstructed Signal
# ----------------------------------------------------------
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed, 'r', linewidth=2, label='Reconstructed')
plt.plot(t, x, '--k', linewidth=1, label='Original')
plt.title("Reconstructed Signal (Receiver)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
print("Sample Binary Codes (First 10):", binary_codes[:10])
plt.show()