import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# ======================================
# Effect Parameters
# ======================================
BITS = 4                     # Bit depth (higher = cleaner)
DOWNSAMPLE_FACTOR = 2 ** 4   # Downsampling factor (2**0 = original rate)
SHRED = 0.5                  # Random amplitude modulation depth
GLITCH = 0.5                 # Probability of sample dropouts
CHAOS = 0.5                  # Random cutoff modulation amount

# Interpolation mode:
# False = Zero-order hold
# True  = Linear interpolation
SMOOTH = False

def main(input_file="audio_input.wav", output_file="audio_output.wav"):
    # ======================================
    # Load and normalize audio
    # ======================================
    filename = input_file
    data, samplerate = sf.read(filename)

    # Convert to mono if needed
    if data.ndim > 1:
        data = data[:, 0]

    # Normalize to [-1, 1]
    data /= np.max(np.abs(data))
    original_clean = data.copy()


    # ======================================
    # Bit depth reduction
    # ======================================
    max_val = 2 ** (BITS - 1) - 1
    data = np.round(data * max_val) / max_val

    output = np.zeros_like(data)


    # ======================================
    # Downsampling + Reconstruction
    # ======================================
    data_ds = data[::DOWNSAMPLE_FACTOR]

    if SMOOTH:
        # Linear interpolation between downsampled points
        ds_indices = np.arange(0, len(data), DOWNSAMPLE_FACTOR)
        if ds_indices[-1] != len(data) - 1:
            ds_indices = np.append(ds_indices, len(data) - 1)
            data_ds = np.append(data_ds, data[-1])

        data = np.interp(np.arange(len(data)), ds_indices, data_ds)
    else:
        # Zero-order hold
        data = np.repeat(data_ds, DOWNSAMPLE_FACTOR)

        if len(data) < len(output):
            data = np.pad(data, (0, len(output) - len(data)), mode="edge")
        else:
            data = data[:len(output)]


    # ======================================
    # Biquad Lowpass Filter
    # ======================================
    class Biquad:
        def __init__(self):
            self.a1 = self.a2 = 0.0
            self.b0 = self.b1 = self.b2 = 0.0
            self.z1 = self.z2 = 0.0


    def init_lowpass(filt, cutoff_hz, fs):
        w0 = 2 * np.pi * cutoff_hz / fs
        alpha = np.sin(w0) / 2
        cos_w0 = np.cos(w0)

        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

        filt.b0 = b0 / a0
        filt.b1 = b1 / a0
        filt.b2 = b2 / a0
        filt.a1 = a1 / a0
        filt.a2 = a2 / a0

        filt.z1 = filt.z2 = 0.0


    def process_biquad(filt, x):
        y = filt.b0 * x + filt.z1
        filt.z1 = filt.b1 * x - filt.a1 * y + filt.z2
        filt.z2 = filt.b2 * x - filt.a2 * y
        return y


    # ======================================
    # Effect Processing Loop
    # ======================================
    lowpass = Biquad()
    cutoff_base = 50.0  # Hz

    for i in range(len(data)):
        # Modulated cutoff frequency
        cutoff = cutoff_base * (1 + CHAOS * (np.random.rand() - 0.5))
        init_lowpass(lowpass, cutoff, samplerate)

        y = process_biquad(lowpass, data[i])

        # Random amplitude modulation
        y *= 1 + SHRED * (np.random.rand() * 2 - 1)

        # Random dropouts
        if np.random.rand() < GLITCH:
            y = 0.0

        output[i] = y


    # ======================================
    # Output normalization and export
    # ======================================
    output /= np.max(np.abs(output))
    sf.write(output_file, output, samplerate, subtype="FLOAT")


    # ======================================
    # Fidelity Analysis
    # ======================================
    from scipy.signal import welch

    # Ensure alignment and equal length
    N = min(len(data), len(output))
    orig = original_clean[:N]
    proc = output[:N]

    # Error signal
    error = orig - proc

    # --- Time-domain metrics ---
    signal_power = np.mean(orig ** 2)
    noise_power = np.mean(error ** 2)

    snr_db = 10 * np.log10(signal_power / noise_power)
    noise_floor_db = 20 * np.log10(np.sqrt(noise_power))

    def crest_factor(x):
        return 20 * np.log10(np.max(np.abs(x)) / np.sqrt(np.mean(x**2)))

    crest_orig = crest_factor(orig)
    crest_proc = crest_factor(proc)

    # --- Frequency-domain metrics ---
    f, psd_orig = welch(orig, samplerate, nperseg=4096)
    _, psd_proc = welch(proc, samplerate, nperseg=4096)
    _, psd_err  = welch(error, samplerate, nperseg=4096)

    # --- Print report ---
    print("\n===== Fidelity Analysis =====")
    print(f"SNR: {snr_db:.2f} dB")
    print(f"Noise Floor (RMS): {noise_floor_db:.2f} dBFS")
    print(f"Crest Factor (Original): {crest_orig:.2f} dB")
    print(f"Crest Factor (Processed): {crest_proc:.2f} dB")
    print("=============================\n")


    # ======================================
    # Visualization
    # ======================================

    plt.figure(figsize=(14, 10))

    # --- Waveform comparison ---
    plt.subplot(4, 1, 1)
    plt.plot(orig, alpha=0.7, label="Original")
    plt.plot(proc, alpha=0.7, label="Processed")
    plt.title("Waveform Comparison")
    plt.legend(frameon=False)

    # --- Error signal ---
    plt.subplot(4, 1, 2)
    plt.plot(error, color="red", alpha=0.8)
    plt.title("Error Signal (Original - Processed)")
    plt.ylim([-1.05, 1.05])

    # --- Power spectral density ---
    plt.subplot(4, 1, 3)
    plt.semilogy(f, psd_orig, label="Original")
    plt.semilogy(f, psd_proc, label="Processed")
    plt.semilogy(f, psd_err, label="Error / Noise", linestyle="--")
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend(frameon=False)

    # --- Spectrogram comparison ---
    plt.subplot(4, 1, 4)
    plt.specgram(error, Fs=samplerate, NFFT=1024, noverlap=512)
    plt.title("Error Spectrogram (Where Fidelity Was Lost)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

