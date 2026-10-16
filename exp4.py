import numpy as np
import matplotlib.pyplot as plt

K = 3                                                            # constraint length
CODES = {"1/2": [7, 5], "1/3": [7, 7, 5], "1/4": [7, 7, 5, 5]}  # octal generators
N_BITS = 20000                                                   # bits per simulation

def step(state, bit, gens):                    # one encoder step
    reg = (bit << (K - 1)) | state
    return reg >> 1, [bin(reg & g).count("1") % 2 for g in gens]

def encode(bits, gens):                        # convolutional encoder
    state, coded = 0, []
    for b in list(bits) + [0] * (K - 1):       # tail bits
        state, out = step(state, b, gens)
        coded += out
    return np.array(coded)

def bpsk_mod(bits):                            # 0 -> +1, 1 -> -1
    return 1 - 2 * bits

def awgn(signal, ebn0_db, rate):               # add Gaussian noise
    sigma = np.sqrt(1 / (2 * rate * 10 ** (ebn0_db / 10)))
    return signal + sigma * np.random.randn(len(signal))

def bpsk_demod(rx):                            # hard decision
    return (rx < 0).astype(int)

def viterbi(rx_bits, gens):                    # Viterbi decoder
    rx = rx_bits.reshape(-1, len(gens))
    n_states = 2 ** (K - 1)
    metric = [0] + [1e9] * (n_states - 1)
    paths = [[] for _ in range(n_states)]
    for r in rx:
        new_metric, new_paths = [1e9] * n_states, [None] * n_states
        for s in range(n_states):
            for bit in (0, 1):
                ns, out = step(s, bit, gens)
                m = metric[s] + np.sum(r != out)
                if m < new_metric[ns]:
                    new_metric[ns], new_paths[ns] = m, paths[s] + [bit]
        metric, paths = new_metric, new_paths
    return np.array(paths[0][:-(K - 1)])

def simulate(ebn0_db, rate_name):              # returns number of bit errors
    data = np.random.randint(0, 2, N_BITS)
    if rate_name == "uncoded":
        rate, coded = 1.0, data
        decoded = bpsk_demod(awgn(bpsk_mod(coded), ebn0_db, rate))
    else:
        gens = CODES[rate_name]
        rate, coded = 1 / len(gens), encode(data, gens)
        decoded = viterbi(bpsk_demod(awgn(bpsk_mod(coded), ebn0_db, rate)), gens)
    return np.sum(decoded != data)

def plot_signals(ebn0_db=2, rate_name="1/2", n=20):   # plot all stages
    gens = CODES[rate_name]
    rate = 1 / len(gens)
    data = np.random.randint(0, 2, n)                 # original bits
    tx = bpsk_mod(encode(data, gens))                 # BPSK signal
    rx = awgn(tx, ebn0_db, rate)                      # after AWGN
    decoded = viterbi(bpsk_demod(rx), gens)           # decoded bits
    errors = np.where(decoded != data)[0]

    fig, ax = plt.subplots(4, 1, figsize=(10, 9))
    ax[0].step(range(n), data, where="post")
    ax[0].set_title("1. Original binary data")
    ax[1].step(range(len(tx)), tx, where="post", color="g")
    ax[1].set_title(f"2. BPSK signal (rate {rate_name}: {len(tx)} symbols)")
    ax[2].plot(rx, "o-", color="r")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_title(f"3. Received signal after AWGN (Eb/N0 = {ebn0_db} dB)")
    ax[3].step(range(n), decoded, where="post", color="m")
    ax[3].plot(errors + 0.5, decoded[errors], "kx", ms=10)
    ax[3].set_title(f"4. Decoded data (Viterbi) - {len(errors)} bit errors")
    for a in ax:
        a.grid(True)
    plt.tight_layout()
    plt.savefig("signals_plot.png", dpi=120)

if __name__ == "__main__":
    np.random.seed(1)
    plot_signals(ebn0_db=2, rate_name="1/2")      # signal plots
    plt.figure()                                  # new figure for BER
    # ---- Small numerical example (rate 1/2) ----
    data = np.random.randint(0, 2, 10)
    coded = encode(data, CODES["1/2"])
    rx = awgn(bpsk_mod(coded), 3, 0.5)
    decoded = viterbi(bpsk_demod(rx), CODES["1/2"])
    print("Data bits    :", data)
    print("Coded bits   :", coded)
    print("BPSK symbols :", bpsk_mod(coded))
    print("Received     :", np.round(rx, 2))
    print("Decoded bits :", decoded)
    print("Errors       :", np.sum(decoded != data), "\n")
    # ---- BER table ----
    snr = np.arange(0, 9)                       # Eb/N0 in dB
    names = ["uncoded", "1/2", "1/3", "1/4"]
    ber = {n: [simulate(s, n) / N_BITS for s in snr] for n in names}

    print(f"BER values ({N_BITS} bits per point)")
    print("Eb/N0(dB) | " + " | ".join(f"{n:>9s}" for n in names))
    print("-" * 52)
    for i, s in enumerate(snr):
        print(f"{s:9d} | " + " | ".join(f"{ber[n][i]:9.5f}" for n in names))
    # ---- BER plot ----
    for n in names:
        plt.semilogy(snr, np.maximum(ber[n], 1e-5), "o-", label=n)
    plt.title("BER vs Eb/N0 (BPSK, AWGN, Viterbi)")
    plt.xlabel("Eb/N0 (dB)");
    plt.ylabel("BER")
    plt.grid(True, which="both");
    plt.legend(); plt.ylim(1e-5, 1)
    plt.savefig("ber_table_plot.png", dpi=120)
    plt.show()