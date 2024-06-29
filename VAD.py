import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import random
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torchaudio.transforms import Spectrogram
from scipy.signal import spectrogram

"""A script for Voice activity detection implementation check"""
torch.set_num_threads(1)

librispeech_data_dir = "/home/prsh7458/work/speech_data/raw_speech"

directory_path=librispeech_data_dir



files = [file for file in os.listdir(directory_path) if file.endswith(".flac")]


random_file = random.choice(files) #choosing random file for analysis

random_wav_path = os.path.join(directory_path, random_file)

waveform, sample_rate = torchaudio.load(random_wav_path)

print("Length of waveform is ",(waveform.shape)," and sampling rate is ",sample_rate)

waveform = waveform[0,:]


# Define the directory where you want to save the model
model_dir = "./silero_vadsilero_vad"

# Check if the model directory exists, and create it if not
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Define the model path
model_path = os.path.join(model_dir, "silero_vad.pth")




if not os.path.exists(model_path):
    print("Downloading the VAD model...")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False)
    torch.save(model.state_dict(), model_path)
else:
    print("Using the cached VAD model.")

# Load the model (note the correct arguments)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')


# Unpack utilities
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils




# wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)
# get speech timestamps from full audio file



def select_and_plot_longest_speech_window(wav, model, sampling_rate):
    # Convert window size from seconds to samples
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
    hop_size = None # 128
    window_size =None # 256
    # Convert window size from seconds to samples
    window_size_samples = int(1.024 * sampling_rate)
    print(f'Window size in samples: {window_size_samples}')

    # Identify the longest non-silent period
    valid_windows = [ts for ts in speech_timestamps if ts['end'] - ts['start'] > window_size_samples]
    selected_window = random.choice(valid_windows)

    # longest_duration = longest_window['end'] - longest_window['start']

    # Check if the longest window is at least 1 second long
    # if longest_duration < window_size_samples:
    #     print("The longest non-silent period is less than 1 second.")
    #     return

    # Choose a random start index within the longest non-silent period
    # start_index = random.randint(longest_window['start'], longest_window['end'] - window_size_samples)
    start_index  = int(selected_window['start'])

    # Extract the 1-second window of audio from the longest non-silent period
    speech_window = wav[start_index:start_index + window_size_samples]

     # Create a time axis for the waveform
    time_axis = np.linspace(0, len(speech_window) / sample_rate, len(speech_window))

    # Calculate the spectrogram for the speech window
    frequencies, times, Sxx = spectrogram(speech_window, fs=sampling_rate)

    # Plot the audio waveform and spectrogram
    plt.figure(figsize=(12, 6))

    # Subplot 1: Waveform
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, speech_window)
    plt.title('Random 1-second Window from Longest Speech Segment (Waveform)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Subplot 2: Spectrogram
    plt.subplot(2, 1, 2)
    im = plt.imshow(10 * np.log10(Sxx), cmap='inferno', aspect='auto', origin='lower',
                    extent=[time_axis.min(), time_axis.max(), frequencies.min(), frequencies.max()])
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # Add colorbar below the spectrogram
    cbar = plt.colorbar(im, ax=plt.gca(), pad=0.1, orientation='horizontal')
    cbar.set_label('dB')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
# Example usage:

    for i in range(5):
        random_file = random.choice(files) #choosing random file for analysis

        random_wav_path = os.path.join(directory_path, random_file)

        waveform, sample_rate = torchaudio.load(random_wav_path)

        print("Length of waveform is ",(waveform.shape)," and sampling rate is ",sample_rate)

        waveform = waveform[0,:]
        select_and_plot_longest_speech_window(waveform, model, sample_rate)
