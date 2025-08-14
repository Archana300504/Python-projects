import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.io import wavfile

# Configuration
MIC_DISTANCE = 0.02  # 2 cm in meters
SOUND_SPEED = 343     # Speed of sound in m/s
SAMPLE_RATE = 44100   # Sampling rate in Hz

# Hexagonal layout of microphones
MIC_POSITIONS = [
    (0, 0),  # Reference microphone (Microphone 1)
    (MIC_DISTANCE, 0),  # Microphone 2
    (MIC_DISTANCE / 2, np.sqrt(3) * MIC_DISTANCE / 2),  # Microphone 3
    (-MIC_DISTANCE / 2, np.sqrt(3) * MIC_DISTANCE / 2),  # Microphone 4
    (-MIC_DISTANCE, 0),  # Microphone 5
    (-MIC_DISTANCE / 2, -np.sqrt(3) * MIC_DISTANCE / 2),  # Microphone 6
]  # Hexagonal layout positions

def load_audio(filepath):
    """Load an audio file and return the normalized signal."""
    sample_rate, data = wavfile.read(filepath)
    if data.ndim > 1:  # Stereo audio
        data = np.mean(data, axis=1)  # Convert to mono
    normalized_data = data / np.max(np.abs(data))  # Normalize signal
    return normalized_data, sample_rate

def calculate_tdoa(signal1, signal2):
    """Calculate the time difference of arrival (TDoA) between two signals."""
    corr = correlate(signal1, signal2, mode="full")
    lag = np.argmax(corr) - len(signal1) + 1  # Find the maximum cross-correlation
    return lag / SAMPLE_RATE

def simulate_microphone_signals(source_angle, source_intensity, input_signal, distance=1.0):
    """
    Simulate signals received by microphones based on a sound source's angle and intensity.
    """
    mic_signals = []
    delays = []
    intensities = []
    source_angle_rad = np.radians(source_angle)
    
    for x, y in MIC_POSITIONS:
        # Calculate distance to the source
        mic_distance = np.sqrt((x - distance * np.cos(source_angle_rad))**2 +
                               (y - distance * np.sin(source_angle_rad))**2)
        
        # Calculate time delay
        delay = mic_distance / SOUND_SPEED
        delays.append(delay)
        
        # Calculate intensity (inverse square law)
        intensity = source_intensity / (mic_distance**2)
        intensities.append(intensity)
        
        # Simulate received signal by shifting and scaling input signal
        delayed_signal = np.roll(input_signal, int(delay * SAMPLE_RATE))
        mic_signals.append(intensity * delayed_signal)
    
    return mic_signals, delays, intensities

def estimate_aoa(tdoas, mic_positions):
    """
    Estimate the angle of arrival (AoA) using TDoAs and microphone positions.
    """
    # Calculate the TDoA between all microphone pairs
    aoa_estimates = []
    for i in range(len(mic_positions)):
        for j in range(i + 1, len(mic_positions)):
            tdoa = tdoas[i][j]  # Time difference of arrival between pair i and j
            # Calculate angle based on TDoA (simplified)
            angle = np.arctan2(mic_positions[j][1] - mic_positions[i][1],
                               mic_positions[j][0] - mic_positions[i][0])
            aoa_estimates.append((angle, tdoa))
    
    # Sort AoA estimates by TDoA (we assume largest TDoA corresponds to most accurate AoA)
    aoa_estimates.sort(key=lambda x: x[1], reverse=True)
    
    # Return the angle of arrival with the highest TDoA
    return np.degrees(aoa_estimates[0][0])

def visualize_microphones_and_aoa(aoa):
    """
    Visualize the microphone array, the estimated AoA, and highlight the microphone with the highest intensity.
    """
    fig, ax = plt.subplots()
    
    # Plot all microphone positions
    for pos in MIC_POSITIONS:
        ax.plot(pos[0], pos[1], 'bo')  # Microphone positions
    
    # Plot AoA direction
    aoa_rad = np.radians(aoa)
    ax.arrow(0, 0, 0.01 * np.cos(aoa_rad), 0.01 * np.sin(aoa_rad), 
             head_width=0.001, color='red')
    
    ax.set_aspect('equal')
    plt.title(f"Microphone Array and AoA: {aoa:.2f}°")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()

def plot_microphone_signals(mic_signals, delays):
    """Plot the microphone signals with time shifts and delays."""
    plt.figure(figsize=(10, 6))
    time = np.linspace(0, len(mic_signals[0]) / SAMPLE_RATE, len(mic_signals[0]))
    
    for i, signal in enumerate(mic_signals):
        plt.plot(time, signal, label=f"Mic {i + 1} (Delay: {delays[i]:.3f}s)")
    
    plt.title("Simulated Microphone Signals")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
def beamform(mic_signals, delays, mic_positions, sound_speed=SOUND_SPEED):
    """Implement beamforming by delaying and summing microphone signals."""
    num_mics = len(mic_signals)
    beamformed_signal = np.zeros_like(mic_signals[0])

    # Apply the delay to each microphone signal and sum them
    for i in range(num_mics):
        # Calculate the time shift (delay) for this microphone
        delay_samples = int(delays[i] * SAMPLE_RATE)
        delayed_signal = np.roll(mic_signals[i], delay_samples)
        
        # Sum the signals
        beamformed_signal += delayed_signal

    # Normalize the beamformed signal
    beamformed_signal /= num_mics
    return beamformed_signal

def main():
    # Automatically generate a source angle (or provide a known angle for testing)
    source_angle = np.random.uniform(0, 360)  # Randomly select an angle between 0 and 360 degrees
    print(f"Assumed Source Angle: {source_angle:.2f} degrees")
    
    # Load and process the uploaded audio signal
    audio_filepath = r"D:\SIH 2024\project code\3 _1.wav"  # Update with the correct file path
    input_signal, sample_rate = load_audio(audio_filepath)
    source_intensity = np.mean(np.abs(input_signal))  # Approximate intensity
    
    # Simulate microphone signals
    mic_signals, delays, intensities = simulate_microphone_signals(
        source_angle, source_intensity, input_signal)
    
    # Plot the microphone signals with their delays
    plot_microphone_signals(mic_signals, delays)
    
    # Apply Beamforming
    beamformed_signal = beamform(mic_signals, delays, MIC_POSITIONS)
    
    # Plot the beamformed signal
    time = np.linspace(0, len(beamformed_signal) / SAMPLE_RATE, len(beamformed_signal))
    plt.figure(figsize=(10, 6))
    plt.plot(time, beamformed_signal, label="Beamformed Signal")
    plt.title("Beamformed Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    
    # Compute TDoAs between all pairs of microphones
    tdoas = np.zeros((len(MIC_POSITIONS), len(MIC_POSITIONS)))
    for i in range(len(mic_signals)):
        for j in range(i + 1, len(mic_signals)):
            tdoa = calculate_tdoa(mic_signals[i], mic_signals[j])
            tdoas[i][j] = tdoa
            tdoas[j][i] = tdoa
    
    # Estimate the angle of arrival (AoA)
    angle_of_arrival = estimate_aoa(tdoas, MIC_POSITIONS)
    print(f"Estimated Angle of Arrival: {angle_of_arrival:.2f} degrees")
    
    # Visualize microphone array and AoA
    visualize_microphones_and_aoa(angle_of_arrival)
    
    # Display intensities and identify the microphone with the highest intensity
    print("\nIntensities Captured by Each Microphone:")
    for i, intensity in enumerate(intensities):
        print(f"Mic {i+1}: {intensity:.4f} (Intensity)")

    max_intensity_mic = np.argmax(intensities) + 1
    print(f"\nMicrophone {max_intensity_mic} captured the highest intensity.")
    
if __name__ == "__main__":
    main()
