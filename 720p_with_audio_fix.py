import numpy as np
from scipy.io import wavfile
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

def data_check(data):
    # check data all 0 or nan
    has_nan = np.isnan(data).any()
    is_all_zero = np.all(data == 0)

    if has_nan:
        print("The data contains NaN values.")
        exit()

    if is_all_zero:
        print("WARN: The data is all zeros.")

# Read a WAV file
sample_rate, data = wavfile.read("/home/damian/Documents/coding_tests/audio_visualizer/your_audio_file_cut.wav")
if len(data.shape) > 1:
    data = data[:, 0]  # Take only one channel if stereo

start_time = 50
end_time = 80
# Calculate start and end sample indices
start_sample = int(start_time * sample_rate)
end_sample = int(end_time * sample_rate)

# Cut the audio
#data = data[start_sample:end_sample]

# check data
data_check(data)

# Parameters
fft_size = 1024
step_size = fft_size // 2
num_bins = 60
bin_cutoffs = np.logspace(np.log10(100), np.log10(sample_rate // 2), num_bins + 1)

# Calculate the frame rate
total_audio_duration = len(data) / sample_rate  # in seconds
total_frames = len(range(0, len(data) - fft_size, step_size))
frame_rate = int(total_frames / total_audio_duration)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter("audio_visualization_temp.mp4", fourcc, frame_rate, (1280, 720))

# init smoothed
smoothed_energies = np.zeros(num_bins)

# Calculate FFT and visualize
for start in range(0, len(data) - fft_size, step_size):
    segment = data[start:start + fft_size]
    data_check(segment)
    
    #Apply windowing (didnt seem to help much)
    window = np.hanning(fft_size)
    segment = segment * window

    # Perform FFT
    fft_result = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    # Calculate energy in each bin
    bin_energies = []
    for i in range(num_bins):
        low, high = bin_cutoffs[i], bin_cutoffs[i + 1]
        mask = (freqs >= low) & (freqs < high)
        # check the mask has soem true
        if not np.any(mask == True):
            print(i, "no mask")
            
        bin_energies.append(np.sum(fft_result[mask]))
    exit()
    # Normalization for visualization
    bin_energies = np.array(bin_energies)
    data_check(bin_energies)
    if not (np.max(bin_energies) - np.min(bin_energies)) == 0:
        bin_energies = (bin_energies - np.min(bin_energies)) / (np.max(bin_energies) - np.min(bin_energies))
            
    
    
    
    # bass is more recognizable so boost the reaction before logging TODO
    for i, energy in enumerate(bin_energies[:50]):
        bin_energies[i] *= np.minimum(3 - (i/50*3), 1)
    
    #log the mf
    bin_energies = np.log1p(bin_energies)
    
    # cutoff to filter noise, sigmoid?    
    bin_energies = (1/(1 +  np.exp(10*(0.3-bin_energies)))) * bin_energies
    
    data_check(bin_energies)
    alpha = 0.2
    smoothed_energies = alpha * np.array(bin_energies) + (1 - alpha) * smoothed_energies
    bin_energies = smoothed_energies
    

    # Create the frame (initially small)
    frame = np.zeros((100, num_bins, 3), dtype=np.uint8)
    
    
    for i, energy in enumerate(bin_energies):
        intensity = int(255 * energy)
        bintensity = int(255*i/num_bins)
        # insert colors here
#        cv2.rectangle(frame, (i, energy), (i + 1, 0), (intensity, (255-intensity), bintensity), -1)
        cv2.rectangle(frame, (i, 100), (i, 100 - int(energy * 100)), (intensity, (150-intensity/2), bintensity), -1)


    # Scale the frame to 720p
    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_NEAREST)

    # Write the frame
    video_writer.write(frame)

# Release video writer
video_writer.release()



# Combine video and audio using moviepy
video_clip = VideoFileClip("audio_visualization_temp.mp4")
audio_clip = AudioFileClip("your_audio_file_cut.wav")

#audio_clip = audio_clip.subclip(start_time, end_time)

final_audio = CompositeAudioClip([audio_clip.set_duration(total_audio_duration)])
final_video = video_clip.set_audio(final_audio)
final_video.write_videofile("audio_visualization.mp4", codec="libx264")

# Cleanup
video_clip.close()
audio_clip.close()
