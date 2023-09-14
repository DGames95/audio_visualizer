import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Read a WAV file
sample_rate, data = wavfile.read("/home/damian/Downloads/Casio-MT-45-16-Beat.wav")
if len(data.shape) > 1:
    data = data[:, 0]  # Take only one channel if stereo

# Parameters
fft_size = 1024
step_size = fft_size // 2
num_bins = 3  # Base, Mid, Treble
bin_cutoffs = [100, 500, 2000, 10000]  # Frequency cutoffs for each bin in Hz

# Calculate FFT and visualize
for start in range(0, len(data) - fft_size, step_size):
    segment = data[start:start + fft_size]
    
    # Perform FFT
    fft_result = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    
    # Calculate energy in each bin
    bin_energies = [0] * num_bins
    for i in range(num_bins):
        low, high = bin_cutoffs[i], bin_cutoffs[i + 1]
        mask = (freqs >= low) & (freqs < high)
        bin_energies[i] = np.sum(fft_result[mask])
    
    # Visualization
    plt.clf()
    plt.title("Audio Spectrum")
    plt.bar(range(num_bins), bin_energies, tick_label=["Base", "Mid", "Treble"])
    plt.ylim(0, np.max(bin_energies) + 1)
    plt.pause(0.01)

plt.show()
