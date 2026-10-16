import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ----------------------------
# 1. Parameters
# ----------------------------
data_bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
num_bits = len(data_bits)
N_c = 10                   # Spreading factor (chips per bit)
sigma = 1.5                # Noise standard deviation per chip

# ----------------------------
# 2. PN code generation
# ----------------------------
pn_code = np.random.choice([-1, 1], size=N_c)

# ----------------------------
# 3. Transmitter: DSSS spreading
# ----------------------------
data_bipolar = 2 * data_bits - 1
spread_signal = np.kron(data_bipolar, pn_code)

# ----------------------------
# 4. Channel: AWGN
# ----------------------------
noise = np.random.normal(0, sigma, spread_signal.size)
received_signal = spread_signal + noise

# ----------------------------
# 5. Receiver: despreading
# ----------------------------
chunks = received_signal.reshape(num_bits, N_c)
correlation = np.sum(chunks * pn_code, axis=1)
received_bits = (correlation > 0).astype(int)

# ----------------------------
# Results
# ----------------------------
bit_errors = np.sum(data_bits != received_bits)

print("PN code:             ", pn_code)
print("Original data bits:  ", data_bits)
print("Recovered data bits: ", received_bits)
print("Correlator outputs:  ", np.round(correlation, 2))
print(f"Bit errors:           {bit_errors} / {num_bits}")

# ----------------------------
# 6. Visualization
# ----------------------------
t_chips = np.arange(num_bits * N_c)
data_chips = np.repeat(data_bits, N_c)
recovered_chips = np.repeat(received_bits, N_c)
bit_centers = np.arange(num_bits) * N_c + N_c / 2

fig, ax = plt.subplots(5, 1, figsize=(14, 13), sharex=True)

ax[0].step(t_chips, data_chips, where="post",
           color="blue", linewidth=2)
ax[0].set_title("1. Original Data Bits")
ax[0].set_ylim(-0.2, 1.2)

ax[1].step(t_chips, spread_signal, where="post",
           color="green", linewidth=1.5)
ax[1].set_title(f"2. Spread Signal (N_c = {N_c})")

ax[2].step(t_chips, received_signal, where="post",
           color="red", linewidth=1, alpha=0.8)
ax[2].set_title(f"3. Received Signal (AWGN, sigma = {sigma})")

ax[3].stem(bit_centers, correlation, basefmt="k-")
ax[3].axhline(0, color="black", linewidth=0.8)
ax[3].set_title("4. Correlator Output (threshold = 0)")

ax[4].step(t_chips, recovered_chips, where="post",
           color="purple", linewidth=2)
ax[4].set_title(f"5. Recovered Data Bits (errors: {bit_errors})")
ax[4].set_ylim(-0.2, 1.2)
ax[4].set_xlabel("Chip Index")

for a in ax:
    a.set_ylabel("Amplitude")
    a.grid(True)
    for b in range(num_bits + 1):
        a.axvline(b * N_c, color="gray",
                  linestyle=":", linewidth=0.6)

ax[3].set_ylabel("Correlation")   # correlator output is a sum, not an amplitude

plt.tight_layout(h_pad=2.5)
plt.show()