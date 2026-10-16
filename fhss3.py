
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ------------------------------------------------------------
# Frequency Hopped Spread Spectrum (FHSS) with FSK
# ------------------------------------------------------------

NBITS = 20          # Number of bits / hops
N = 120             # Samples per bit

def gray(x):
    """Binary to Gray code."""
    return x ^ (x >> 1)

# 1. Random hopping codes (values 1..7)
cp = np.random.randint(1, 8, NBITS)

# Odd positions keep cp; even positions use Gray code
dualcode = cp.copy()
dualcode[1::2] = gray(cp[1::2])

# 2. Random data bits
s = np.random.randint(0, 2, NBITS)

# 3. FSK modulation
# Bit 0 -> 2 cycles per bit
# Bit 1 -> 4 cycles per bit
t = np.arange(N) / N
f0 = 2
f1 = 4

signal = np.repeat(s, N)  # Original binary data

fsk_sig = np.concatenate([
    np.cos(2 * np.pi * (f1 if bit == 1 else f0) * np.arange(N) / N)
    for bit in s
])

# 4. Generate seven hopping carriers
# Each carrier has a different number of cycles per bit
hop_cycles = [1, 2, 3, 4, 5, 6, 7]

carriers = [
    np.cos(2 * np.pi * cycles * np.arange(N) / N)
    for cycles in hop_cycles
]

# 5. Select one hopping carrier per bit
spread_signal = np.concatenate([
    carriers[c - 1] for c in dualcode
])

# 6. Generate FHSS-FSK signal
# Multiply the FSK waveform by the selected hopping carrier
freq_hopped_sig = fsk_sig * spread_signal

# 7. Plot signals
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(signal, color="black", linewidth=1)
plt.title("Original Data Bits")
plt.xlim(0, NBITS * N)
plt.ylim(-0.2, 1.2)
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(fsk_sig, color="red", linewidth=0.8)
plt.title("FSK Modulated Signal")
plt.xlim(0, NBITS * N)
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(spread_signal, color="green", linewidth=0.8)
plt.title("Hopping Carrier Signal (7 Frequencies)")
plt.xlim(0, NBITS * N)
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(freq_hopped_sig, color="blue", linewidth=0.8)
plt.title("Frequency Hopped FSK Signal")
plt.xlim(0, NBITS * N)
plt.xlabel("Sample Number")
plt.grid(True)

plt.tight_layout()
plt.show()

print("Original bits   :", s)
print("Hopping sequence:", dualcode)
