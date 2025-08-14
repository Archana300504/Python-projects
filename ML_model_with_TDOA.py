import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.signal import correlate
from scipy.io import wavfile

print(tf.__version__)

# Set path for gunshot audio files
gunshot_audio_path = r'D:/SIH 2024/project code/archive (1)/AK-12'

# Function to load and preprocess audio files
def load_and_preprocess_audio(file_path, target_length=200):
    signal, sr = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000)

    if mel_spectrogram.shape[1] < target_length:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, target_length - mel_spectrogram.shape[1])), mode='constant')
    elif mel_spectrogram.shape[1] > target_length:
        mel_spectrogram = mel_spectrogram[:, :target_length]

    return mel_spectrogram

def load_dataset(gunshot_audio_path):
    X = []
    y = []
    for file in os.listdir(gunshot_audio_path):
        if file.endswith(".wav"):
            file_path = os.path.join(gunshot_audio_path, file)
            mel_spectrogram = load_and_preprocess_audio(file_path)
            X.append(mel_spectrogram)
            label = 1 if "gunshot" in file else 0
            y.append(label)

    return np.array(X), np.array(y)

# Load the dataset (only gunshot audios)
X, y = load_dataset(gunshot_audio_path)

# Resize the input data to be consistent for CNN
X_resized = np.array([np.resize(mel, (128, 128)) for mel in X])

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.2, random_state=42)

# Build the model using TensorFlow
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(128, 128, 1)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Print the model summary
model.summary()

# Reshape the data for CNN
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Save the model in Keras format
model.save('gunshot_classification_model.keras')

# Configuration for microphone processing
MIC_DISTANCE = 0.02  
SOUND_SPEED = 343     
SAMPLE_RATE = 44100   

MIC_POSITIONS = [
    (MIC_DISTANCE * np.cos(np.pi / 3 * i), MIC_DISTANCE * np.sin(np.pi / 3 * i)) 
    for i in range(6)
]

def load_audio(filepath):
    sample_rate, data = wavfile.read(filepath)
    if data.ndim > 1:  
        data = np.mean(data, axis=1)  
    normalized_data = data / np.max(np.abs(data))  
    return normalized_data

def calculate_tdoa(signal1, signal2):
    corr = correlate(signal1, signal2)
    lag = np.argmax(corr) - len(signal1) + 1  
    return lag / SAMPLE_RATE

def simulate_microphone_signals(source_angle, source_intensity, input_signal, distance=1.0):
    """
    Simulate signals received by microphones based on a sound source's angle and intensity.
    distance: The distance from the microphone array to the sound source.
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
        
        # Simulate received signal by shifting the input signal
        delayed_signal = np.roll(input_signal, int(delay * SAMPLE_RATE))
        mic_signals.append(delayed_signal)
    
    return mic_signals

def estimate_aoa(tdoas):
    aoa_estimates = []
    
    for i in range(len(MIC_POSITIONS)):
        for j in range(i + 1, len(MIC_POSITIONS)):
            tdoa = tdoas[i][j]  
            angle = np.arctan2(MIC_POSITIONS[j][1] - MIC_POSITIONS[i][1],
                               MIC_POSITIONS[j][0] - MIC_POSITIONS[i][0])
            aoa_estimates.append((angle))

    return np.degrees(aoa_estimates[0])

def beamform(mic_signals):
    beamformed_signal = np.zeros_like(mic_signals[0])
    
    for signal in mic_signals:
        beamformed_signal += signal
    
    return beamformed_signal / len(mic_signals)

def main():
    audio_filepath = r"D:\SIH 2024\project code\3 _1.wav" 
    input_signal = load_audio(audio_filepath)
    
    source_angle = np.random.uniform(0, 360)  
    source_intensity = np.mean(np.abs(input_signal))
    
    # Define distance from microphone array to sound source (in meters)
    distance_to_source = 1.0  
    
    mic_signals = simulate_microphone_signals(source_angle, source_intensity, input_signal,
                                               distance=distance_to_source)
    
    tdoas = np.zeros((len(MIC_POSITIONS), len(MIC_POSITIONS)))
    
    for i in range(len(mic_signals)):
        for j in range(i + 1, len(mic_signals)):
            tdoa_value = calculate_tdoa(mic_signals[i], mic_signals[j])
            tdoas[i][j] = tdoa_value
            tdoas[j][i] = tdoa_value
    
    angle_of_arrival_estimate = estimate_aoa(tdoas)
    
    print(f"Estimated Angle of Arrival: {angle_of_arrival_estimate:.2f} degrees")
    
    beamformed_signal = beamform(mic_signals)

    # Convert the beamformed signal into a Mel spectrogram
    beamformed_signal_normalized = beamformed_signal / np.max(np.abs(beamformed_signal))

    # Create a Mel spectrogram from the normalized beamformed signal
    beamformed_mel_spectrogram = librosa.feature.melspectrogram(y=beamformed_signal_normalized,
                                                                 sr=SAMPLE_RATE,
                                                                 n_mels=128,
                                                                 fmax=8000)

    # Resize for prediction (add batch dimension and channel dimension)
    beamformed_mel_spectrogram_resized = np.resize(beamformed_mel_spectrogram,
                                                    (128, 128))  
                                                    
    beamformed_mel_spectrogram_expanded = np.expand_dims(np.expand_dims(beamformed_mel_spectrogram_resized,
                                                                      axis=-1), axis=0)

    prediction = model.predict(beamformed_mel_spectrogram_expanded)

    print(f"Prediction (Gunshot Probability): {prediction[0][0]:.4f}")

    # Calculate intensities based on distances from source to microphones using inverse square law.
    intensities = []
    
    for x_pos, y_pos in MIC_POSITIONS:
        mic_distance_to_source = np.sqrt((x_pos - distance_to_source * np.cos(np.radians(source_angle)))**2 +
                                          (y_pos - distance_to_source * np.sin(np.radians(source_angle)))**2)
        intensity_at_mic = source_intensity / (mic_distance_to_source ** 2) if mic_distance_to_source > 0 else 0
        intensities.append(intensity_at_mic)

   # Find which microphone had the highest intensity.
    max_intensity_index = np.argmax(intensities)
   
    print(f"Intensities at microphones: {intensities}")
   
   # Plotting microphone positions and highlighting the one with maximum intensity.
    plt.figure(figsize=(8, 8))
   
   # Plot all microphone positions.
    for i in range(len(MIC_POSITIONS)):
       plt.plot(MIC_POSITIONS[i][0], MIC_POSITIONS[i][1], 'bo') # Microphone positions
   
       # Annotate with intensity value.
       plt.annotate(f'Mic {i+1}: {intensities[i]:.2f}', 
                    (MIC_POSITIONS[i][0], MIC_POSITIONS[i][1]), 
                    textcoords="offset points", 
                    xytext=(0,-10), 
                    ha='center')
        
       # Highlighting maximum intensity microphone.
       if i == max_intensity_index:
           plt.plot(MIC_POSITIONS[i][0], MIC_POSITIONS[i][1], 'ro', markersize=10) # Highlighted microphone
   
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)
    plt.title('Microphone Array and Sound Intensities')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid()
    plt.axhline(0,color='black',linewidth=0.5)
    plt.axvline(0,color='black',linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == "__main__":
   main()
