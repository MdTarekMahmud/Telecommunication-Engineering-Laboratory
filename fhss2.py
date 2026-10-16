import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Frequency Hopped Spread Spectrum (FHSS) with BPSK
# ------------------------------------------------------------
np.random.seed(42)
NBITS = 20          # number of bits / hops
N = 120             # samples per bit


def gray(x):
    """Binary to Gray code."""
    return x ^ (x >> 1)
# 1. Random hopping codes (values 1..7)
cp = np.random.randint(1, 8, NBITS)

# Manual: odd positions (1st, 3rd, ...) keep cp,
#         even positions (2nd, 4th, ...) are Gray coded.
# For values 1..7 the Gray code stays in 1..7, so no remapping is needed.
dualcode = cp.copy()
dualcode[1::2] = gray(cp[1::2])

# 2. Random data bits
s = np.random.randint(0, 2, NBITS)

# 3. BPSK modulation: 1 carrier cycle per bit
t = np.linspace(0, 2 * np.pi, N)
signal = np.repeat(2 * s - 1, N)               # NRZ data, +1 / -1
carrier = np.tile(np.cos(t), NBITS)
bpsk_sig = signal * carrier

# 4. Seven carriers (points per cycle -> cycles per bit)
#    5 pts -> 24, 10 -> 12, 20 -> 6, 30 -> 4, 40 -> 3, 60 -> 2, 120 -> 1
pts = [5, 10, 20, 30, 40, 60, 120]
carriers = [np.tile(np.cos(np.linspace(0, 2 * np.pi, p)), N // p) for p in pts]

# 5. Spread signal: one carrier per bit, chosen by the hopping code
spread_signal = np.concatenate([carriers[c - 1] for c in dualcode])

# 6. Frequency hopped signal
freq_hopped_sig = bpsk_sig * spread_signal

# 7. Plots

plt.figure(figsize=(10, 9))

plt.subplot(4, 1, 1)
plt.plot(signal, color='black', linewidth=0.8)
plt.title("Original Bit Sequence")
plt.xlim(-100, 2500)
plt.ylim(-1.5, 1.5)
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(bpsk_sig, color='red', linewidth=0.8)
plt.title("BPSK Modulated Signal")
plt.xlim(-100, 2500)
plt.ylim(-1.5, 1.5)
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(spread_signal, color='green', linewidth=0.8)
plt.title("Spread Signal with 7 Frequencies")
plt.xlim(-100, 2500)
plt.ylim(-1.5, 1.5)
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(freq_hopped_sig, color='blue', linewidth=0.8)
plt.title("Frequency Hopped Spread Spectrum Signal")
plt.xlim(-100, 2500)
plt.ylim(-1.5, 1.5)
plt.xlabel("Sample Number")
plt.grid(True)

plt.tight_layout()
plt.show()

print("Original bits   :", s)
# print("Random codes cp :", cp)
print("Hopping sequence:", dualcode)