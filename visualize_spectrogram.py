import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

# Load the audio file
audio_path = 'output.mp3'
y, sr = librosa.load(audio_path)

# Compute the spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(12, 8))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Spectrogram of {audio_path}')

# Save the spectrogram image
plt.tight_layout()
plt.savefig('output_spectrogram.png')
print(f"Spectrogram saved as 'output_spectrogram.png'")