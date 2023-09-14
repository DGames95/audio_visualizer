import numpy as np
from scipy.io import wavfile
import cv2

# Read a WAV file
sample_rate, data = wavfile.read("/home/damian/Downloads/Casio-MT-45-16-Beat.wav")
if len(data.shape) > 1:
    data = data[:, 0]  # Take only one channel if stereo

# Parameters
fft_size = 1024
step_size = fft_size // 2
num_bins = 20
bin_cutoffs = np.logspace(np.log10(100), np.log10(sample_rate // 2), num_bins + 1)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter("audio_visualization.mp4", fourcc, 30.0, (num_bins, 100))

# Calculate FFT and visualize
for start in range(0, len(data) - fft_size, step_size):
    segment = data[start:start + fft_size]

    # Perform FFT
    fft_result = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    # Calculate energy in each bin
    bin_energies = []
    for i in range(num_bins):
        low, high = bin_cutoffs[i], bin_cutoffs[i + 1]
        mask = (freqs >= low) & (freqs < high)
        bin_energies.append(np.sum(fft_result[mask]))

    # Normalization for visualization
    bin_energies = np.array(bin_energies)
    bin_energies = (bin_energies - np.min(bin_energies)) / (np.max(bin_energies) - np.min(bin_energies))

    # Create the frame
    frame = np.zeros((100, num_bins, 3), dtype=np.uint8)
    for i, energy in enumerate(bin_energies):
        intensity = int(255 * energy)
        cv2.rectangle(frame, (i, 100), (i + 1, 100 - int(energy * 100)), (intensity, intensity, intensity), -1)

    # Write the frame
    video_writer.write(frame)

# Release video writer
video_writer.release()
