import os
import wave
import random
import numpy as np
import matplotlib.pyplot as plt


"""A script to plot misclaneous plot"""


# Define the path where the .wav files are stored
path = "/home/prsh7458/Desktop/scratch4/RIR_dataset_room/IR_DATA_HL00W"

# List all .wav files in the directory
wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]

# Select a random .wav file from the list
random_wav_file = random.choice(wav_files)

# Construct the full file path
full_wav_path = os.path.join(path, random_wav_file)

# Open the .wav file
with wave.open(full_wav_path, 'r') as wav_file:
    # Extract Raw Audio from Wav File
    signal = wav_file.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int16)
    
    # Get the frame rate
    frame_rate = wav_file.getframerate()
    
    # Find the number of frames in the audio clip
    n_frames = wav_file.getnframes()
    
    # Calculate the duration of the audio file
    duration = n_frames / float(frame_rate)
    
    # Create a time axis for the audio file
    time_axis = np.linspace(0, duration, num=n_frames)

signal = signal/(np.max(np.abs(signal)))
# Plot the audio file waveform
plt.figure(figsize=(12, 6))
plt.plot(time_axis, signal, label='Room Impulse Response')
plt.legend(loc='upper right', fontsize=28)
plt.title('Time Domain Room Impulse Response', fontsize=28)
plt.xlabel('Time [s]', fontsize=28)
plt.ylabel('Amplitude', fontsize=28)
plt.tick_params(axis='both', which='major', labelsize=28)
plt.tight_layout()
plt.show()

