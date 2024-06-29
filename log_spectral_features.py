""" Implementation of paper H. Gamper and I. J. Tashev, "Blind Reverberation Time Estimation Using a Convolutional Neural Network," 2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC), Tokyo, Japan, 2018, pp. 136-140, doi: 10.1109/IWAENC.2018.8521241. keywords: {Training;Estimation;Neural networks;Noise measurement;Training data;Reverberation;T60;energy decay rate;deep neural networks},

"""
import torch
import torchaudio
import numpy as np
from scipy.signal import lfilter, butter
import scipy.signal as signal
import librosa
import matplotlib.pylab as  plt
from scipy.signal import bilinear
import random
import torch
import torch.nn as nn
import torch.nn.functional as F




def read_audio(filename):
    waveform, samplerate = torchaudio.load(filename)
    return waveform.numpy(), samplerate  # Convert to numpy array immediately

def chunk_audio(data, samplerate):
    time = 1.024
    chunk_size = int(time * samplerate)  # 4 seconds chunks
    overlap = int(0.5 * samplerate)  # 0.5 seconds overlap
    chunks = []
    start = 0
    while start + chunk_size <= data.shape[1]:  # Ensure the chunk does not exceed the data length
        chunks.append(data[:, start:start + chunk_size])
        start += chunk_size - overlap  # Move start forward by the chunk size minus overlap
    return chunks

def filter_chunks(chunks):
    full_rms = 10 * np.log10(np.mean([np.square(chunk).mean() for chunk in chunks]))
    threshold = full_rms - 20  # 20 dB lower than the full RMS
    filtered_chunks = [chunk for chunk in chunks if 10 * np.log10(np.square(chunk).mean()) > threshold]
    return filtered_chunks

def apply_a_weighting(chunks, fs):
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997
    pi = np.pi

    NUMs = [(2 * pi * f4)**2 * (10**(A1000 / 20)), 0, 0, 0, 0]
    DENs = np.polymul([1, 4 * pi * f4, (2 * pi * f4)**2],
                      [1, 4 * pi * f1, (2 * pi * f1)**2])
    DENs = np.polymul(np.polymul(DENs, [1, 2 * pi * f3]),
                      [1, 2 * pi * f2])

    # Bilinear transformation to get the digital filter coefficients
    b, a = bilinear(NUMs, DENs, fs)

    # Apply the A-weighting filter to each chunk
    weighted_chunks = [lfilter(b, a, chunk) for chunk in chunks]

    return weighted_chunks

def erb_space(low_freq, high_freq, num_bands):
    # Constants
    ear_q = 9.26449
    min_bw = 24.7
    # Compute frequencies
    erb = [((i * (24.7 * (4.37e-3 * high_freq - 1.0)) / (num_bands - 1)) + 24.7 * (4.37e-3 * low_freq - 1.0)) / ear_q for i in range(num_bands)]
    return np.array(erb)

# def erb_space(low_freq, high_freq, num_bands):
#     # Calculate the Equivalent Rectangular Bandwidth (ERB) spaced frequencies
#     # This uses the formula from Moore and Glasberg (1983)
#     ear_q = 9.26449  # Glasberg and Moore Parameters
#     min_bw = 24.7
#     order = 1

#     # Calculate ERB at each frequency
#     erb = ((np.linspace(1, num_bands, num=num_bands) - 1) * (high_freq ** order - low_freq ** order) / (num_bands - 1) + low_freq ** order) ** (1 / order)
#     return erb / ear_q + min_bw

def process_gammatone(chunks, samplerate):
    num_bands = 21  # Number of frequency bands
    low_freq = 400  # Lowest band edge of filters (Hz)
    high_freq = 6000  # Highest band edge of filters (Hz)
    
    # Calculate the center frequencies for the gammatone filterbank
    freqs = erb_space(low_freq, high_freq, num_bands)
    
    # Initialize the list to hold the feature matrices
    features = []

    # Process each chunk with the gammatone filterbank
    for chunk in chunks:
        chunk_features = []
        for f in freqs:
            # Create the gammatone filter
            b, a = signal.gammatone(f, 'fir', fs=samplerate, numtaps=1024)
            # Filter the chunk
            filtered = signal.lfilter(b, [1.0], chunk)
            # Compute the energy of the filtered signal in frames
            frame_len = 64  # Frame size (samples)
            frame_step = 32  # Step size (samples)
            frames = librosa.util.frame(filtered, frame_length=frame_len, hop_length=frame_step).T
            energy = np.log(np.sum(frames**2, axis=1) + np.finfo(float).eps)  # Log energy, avoid log(0)
            chunk_features.append(energy)
        # Stack the energies for all frequency bands to form the feature matrix
        features.append(np.stack(chunk_features, axis=0))
    
    return np.array(features)

def normalize_features(features):
    medians = np.median(features, axis=2)
    features -= medians[:, :, None]
    means = np.mean(features, axis=2)
    stds = np.std(features, axis=2)
    normalized_features = (features - means[:, :, None]) / stds[:, :, None]
    return normalized_features

def plot_features_as_image(final_features, samplerate, num_bands, low_freq, high_freq):
    num_time_frames = final_features.shape[1]
    
    # Define the time axis extending from 0 to the length of the audio in seconds
    time_axis = np.linspace(0, num_time_frames / samplerate, num=num_time_frames)
    
    # Define the frequency axis extending from low_freq to high_freq
    frequency_axis = np.linspace(low_freq, high_freq, num=num_bands)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(final_features, aspect='auto', origin='lower', 
               extent=[time_axis.min(), time_axis.max(), frequency_axis.min(), frequency_axis.max()])
    
    # plt.colorbar(label='Log Energy')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Pre-processed input sample')
    plt.show()

# You would call the function as follows:
# Ensure you provide the correct samplerate, num_bands, and frequency range.

# Example Usage
audio_file = f'/home/prsh7458/Desktop/scratch4/speech_data/motus_saved_noise_reverb_edc/reverberant_speech/reverberant_speech_{random.randint(0,200)}.wav'

data, samplerate = read_audio(audio_file)
print(f"sample rate is : {samplerate}")
print(f"Data shape: {data.shape}")

chunks = chunk_audio(data, samplerate)
print(f"Chunks shape: {np.array(chunks).shape}")

filtered_chunks = filter_chunks(chunks)
print(f"Filtered chunks shape: {np.array(filtered_chunks).shape}")

weighted_chunks = apply_a_weighting(filtered_chunks, samplerate)
print(f"Weighted chunks shape: {np.array(weighted_chunks).shape}")

gammatone_features = process_gammatone(weighted_chunks, samplerate)
print(f"Gammatone features shape: {gammatone_features.shape}")

final_features = normalize_features(gammatone_features)
print(f"Final features shape: {np.squeeze(final_features).shape}")

# plot_features_as_image(np.squeeze(final_features), samplerate, 21, 400, 6000)

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 5, kernel_size=(1, 10), stride=(1, 2), padding=(0, 1))  # 1x10 kernel, stride (1, 2), padding (0, 1)
        # self.conv2 = nn.Conv2d(5, 5, kernel_size=(1, 10), stride=(1, 3), padding=(0, 1))  # 1x10 kernel, stride (1, 3), padding (0, 1)
        # self.conv3 = nn.Conv2d(5, 5, kernel_size=(1, 11), stride=(1, 3), padding=(0, 1))  # 1x11 kernel, stride (1, 3), padding (0, 1)
        self.conv4 = nn.Conv2d(5, 5, kernel_size=(1, 11), stride=(1, 2), padding=(0, 1))  # 1x11 kernel, stride (1, 2), padding (0, 1)
        self.conv5 = nn.Conv2d(5, 5, kernel_size=(3, 8), stride=(2, 2), padding=(0, 1))  # 3x8 kernel, stride (2, 2), padding (0, 1)
        # self.conv6 = nn.Conv2d(5, 5, kernel_size=(4, 7), stride=(2, 1), padding=(0, 1))  # 4x7 kernel, stride (2, 1), padding (0, 1)

        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 2x2 pooling kernel, stride (2, 2)

        # Define the dropout layer
        self.dropout = nn.Dropout(0.5)

        # Define the fully connected layers
        self.fc1 = nn.Linear(30, 512)  # Adjust the input size based on the flattened output shape
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # Apply the convolutional layers with ReLU activations and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        # x = self.pool(F.relu(self.conv6(x)))

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)

        print(x.shape)

        # Apply dropout
        x = self.dropout(x)

        # Apply the fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Apply the final fully connected layer
        x = self.fc3(x)
        return x


# Create the network
cnn = CustomCNN()

# Forward pass to get the output
input_feature = torch.from_numpy(np.squeeze(final_features)).float()
sample_batch = torch.randn(5,1, 21, 511)
output = cnn(input_feature.unsqueeze(0).unsqueeze(0))
# output = cnn(sample_batch)
print(output.shape)  



