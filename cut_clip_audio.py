from scipy.io import wavfile

def cut_audio(input_file, output_file, start_time, end_time):
    # Read the input WAV file
    sample_rate, data = wavfile.read(input_file)
    
    # If stereo, use only one channel
    if len(data.shape) > 1:
        data = data[:, 0]

    # Calculate start and end sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Cut the audio
    cut_data = data[start_sample:end_sample]

    # Write to the output WAV file
    wavfile.write(output_file, sample_rate, cut_data)

# Usage
input_file = "/home/damian/Downloads/Wiguez & Vizzen Ft. Maestro Chives - Running Wild (EH!DE Remix) [NCS Release].wav"
output_file = "your_audio_file_cut.wav"
start_time = 50  # start time in seconds
end_time = 80 # end time in seconds

cut_audio(input_file, output_file, start_time, end_time)
