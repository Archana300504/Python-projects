import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import butter, filtfilt
import tkinter as tk
from tkinter import filedialog


# Frequency-Amplitude Plot and Dominant Frequency Detection
def plot_frequency_amplitude(audio, sr):
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft)
    frequency = np.fft.fftfreq(len(magnitude), 1 / sr)

    positive_freq_idx = frequency > 0
    frequency = frequency[positive_freq_idx]
    magnitude = magnitude[positive_freq_idx]
    magnitude = magnitude / np.max(magnitude)

    dominant_freq = frequency[np.argmax(magnitude)]
    print(f"Dominant Frequency: {dominant_freq} Hz")

    plt.figure(figsize=(10, 6))
    plt.plot(frequency, magnitude, label="Frequency-Amplitude")
    plt.title("Frequency vs Amplitude Plot")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 5000)
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    plt.show()

    return dominant_freq


# Preamplification Simulation
def pre_amplification_simulation(dominant_freq, fs=100e3):
    R1, R2 = 1e3, 10e3
    Av = 1 + (R2 / R1)

    t = np.linspace(0, 0.01, int(fs * 0.01))
    Vin = 0.01 * np.sin(2 * np.pi * dominant_freq * t)

    Vout = Av * Vin
    noise = np.random.normal(0, 0.005, len(t))
    Vin_noisy, Vout_noisy = Vin + noise, Av * (Vin + noise)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, Vin_noisy, label='Input Voltage (Vin)', color='blue')
    plt.title('Input Signal (Noisy)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, Vout_noisy, label='Output Voltage (Vout)', color='red')
    plt.title('Output Signal (Amplified)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return Vout_noisy, fs


# Impedance Matching Simulation
def impedance_matching(Vout, fs):
    t = np.linspace(0, 0.01, len(Vout))
    noise = np.random.normal(0, 0.01, len(t))
    Vin_noisy = Vout + noise
    Vout_matched = Vin_noisy

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, Vin_noisy, label='Input Voltage (Vin)', color='blue')
    plt.title('Input Signal (Vin)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, Vout_matched, label='Output Voltage (Vout)', color='green')
    plt.title('Output Signal (Matched)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return Vout_matched


# Anti-Aliasing Filter
def apply_anti_aliasing_filter(Vout_matched, fs, cutoff_freq=1000, order=4):
    def butter_lowpass(cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    t = np.linspace(0, 0.01, len(Vout_matched))
    b, a = butter_lowpass(cutoff_freq, fs, order)
    Vout_filtered = filtfilt(b, a, Vout_matched)

    plt.figure(figsize=(10, 6))
    plt.plot(t, Vout_matched, label='Matched Output Signal', color='gray', alpha=0.6)
    plt.plot(t, Vout_filtered, label='Filtered Output Signal', color='green')
    plt.title('Filtered Output Signal (Anti-aliasing Low-pass Filter)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

    return Vout_filtered


# Include the DC blocking after anti-aliasing
def apply_dc_blocking(Vout_filtered, fs, cutoff_freq=50, order=4):
    def butter_highpass(cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    t = np.linspace(0, 0.01, len(Vout_filtered))
    b, a = butter_highpass(cutoff_freq, fs, order)
    Vout_dc_blocked = filtfilt(b, a, Vout_filtered)

    plt.figure(figsize=(10, 6))
    plt.plot(t, Vout_filtered, label='Filtered Signal (Anti-aliasing)', color='gray', alpha=0.6)
    plt.plot(t, Vout_dc_blocked, label='Signal after DC Blocking', color='blue')
    plt.title('DC Blocking Using High-Pass Filter')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

    return Vout_dc_blocked
# ADC Conversion Simulation
def adc_conversion(Vout_dc_blocked, fs, adc_resolution=12, sample_rate=1e6):
    # Sampling the signal at the ADC sample rate
    t_original = np.linspace(0, 0.01, len(Vout_dc_blocked))
    t_sampled = np.arange(0, 0.01, 1 / sample_rate)
    Vout_sampled = np.interp(t_sampled, t_original, Vout_dc_blocked)

    # Quantization (ADC resolution)
    adc_levels = 2**adc_resolution
    Vout_min, Vout_max = np.min(Vout_sampled), np.max(Vout_sampled)
    Vout_quantized = np.round(
        (Vout_sampled - Vout_min) / (Vout_max - Vout_min) * (adc_levels - 1)
    )

    # Normalize to ADC range
    Vout_discrete = Vout_quantized / (adc_levels - 1) * (Vout_max - Vout_min) + Vout_min

    # Visualize as discrete dots
    plt.figure(figsize=(10, 6))
    plt.plot(t_original, Vout_dc_blocked, label="Analog Signal", color="gray", alpha=0.6)
    plt.scatter(t_sampled, Vout_discrete, color="red", label="Discrete Signal (ADC)", zorder=5)
    plt.title("ADC Conversion: Analog to Discrete Signal")
    plt.xlim(0,0.01)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.show()

    return Vout_discrete, t_sampled

# Bandpass Filter
def apply_bandpass_filter(Vout_discrete, fs, low_cutoff=2900, high_cutoff=3100, order=4):
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    t = np.linspace(0, len(Vout_discrete) / fs, len(Vout_discrete))
    b, a = butter_bandpass(low_cutoff, high_cutoff, fs, order)
    Vout_bandpassed = filtfilt(b, a, Vout_discrete)

    # Frequency Analysis of the Bandpass Filtered Signal
    fft = np.fft.fft(Vout_bandpassed)
    magnitude = np.abs(fft)
    frequency = np.fft.fftfreq(len(magnitude), 1 / fs)

    positive_freq_idx = frequency > 0
    frequency = frequency[positive_freq_idx]
    magnitude = magnitude[positive_freq_idx]

    # Normalize the magnitude for visualization
    magnitude = magnitude / np.max(magnitude)

    plt.figure(figsize=(10, 6))
    plt.plot(frequency, magnitude, label="Frequency vs Magnitude")
    plt.title("Bandpass Filter Output (3 kHz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Normalized)")
    plt.xlim(2500, 3500)  # Focus on the bandpass range
    plt.grid()
    plt.legend()
    plt.show()

    return Vout_bandpassed


def upload_and_analyze():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if not file_path:
        print("No file selected!")
        return

    print(f"File Selected: {file_path}")
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    print(f"Loaded Audio with Sample Rate: {sr}")

    dominant_freq = plot_frequency_amplitude(audio, sr)
    Vout_preamplified, fs = pre_amplification_simulation(dominant_freq)
    Vout_matched = impedance_matching(Vout_preamplified, fs)
    Vout_filtered = apply_anti_aliasing_filter(Vout_matched, fs)
    Vout_dc_blocked = apply_dc_blocking(Vout_filtered, fs)

    # Simulate ADC conversion
    adc_resolution = 12  # Example: 12-bit ADC
    sample_rate = 1e6    # Example: 1 MSPS (Mega Samples Per Second)
    Vout_discrete, t_sampled = adc_conversion(Vout_dc_blocked, fs, adc_resolution, sample_rate)

    # Apply Bandpass Filter
    Vout_bandpassed = apply_bandpass_filter(Vout_discrete, fs)

    print("Processing Completed!")


# GUI Creation
def create_gui():
    root = tk.Tk()
    root.title("Audio Signal Processing Pipeline")
    root.geometry("400x200")

    label = tk.Label(root, text="Upload Audio for Processing", font=("Helvetica", 16))
    label.pack(pady=20)

    upload_button = tk.Button(root, text="Upload Audio File", command=upload_and_analyze, font=("Helvetica", 14))
    upload_button.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    create_gui()
