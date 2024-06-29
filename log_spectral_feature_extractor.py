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


class Log_spectral_feature_extraction():

    def __init__(self,audio_data,samplerate) :
        self.audio_data = audio_data
        self.samplerate = samplerate

    def chunk_audio(self, data):
        time = 1.024
        chunk_size = int(time * self.samplerate)  # 4 seconds chunks
        overlap = int(0.5 * self.samplerate)  # 0.5 seconds overlap
        chunks = []
        start = 0
        while start + chunk_size <= 16384:  # Ensure the chunk does not exceed the data length
            chunks.append(data[:, start:start + chunk_size])
            start += chunk_size - overlap  # Move start forward by the chunk size minus overlap
        return chunks

    def filter_chunks(self,chunks):
        full_rms = 10 * np.log10(np.mean([np.square(chunk).mean() for chunk in chunks]))
        threshold = full_rms - 20  # 20 dB lower than the full RMS
        filtered_chunks = [chunk for chunk in chunks if 10 * np.log10(np.square(chunk).mean()) > threshold]
        return filtered_chunks

    def apply_a_weighting(self,chunks):
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
        b, a = bilinear(NUMs, DENs, self.samplerate)

        # Apply the A-weighting filter to each chunk
        weighted_chunks = [lfilter(b, a, chunk) for chunk in chunks]

        return weighted_chunks

    def erb_space(self,low_freq, high_freq, num_bands):
        # Constants
        ear_q = 9.26449
        min_bw = 24.7
        # Compute frequencies
        erb = [((i * (24.7 * (4.37e-3 * high_freq - 1.0)) / (num_bands - 1)) + 24.7 * (4.37e-3 * low_freq - 1.0)) / ear_q for i in range(num_bands)]
        return np.array(erb)

    def process_gammatone(self,chunks):
        num_bands = 21  # Number of frequency bands
        low_freq = 400  # Lowest band edge of filters (Hz)
        high_freq = 6000  # Highest band edge of filters (Hz)
        
        # Calculate the center frequencies for the gammatone filterbank
        freqs = self.erb_space(low_freq, high_freq, num_bands)
        
        # Initialize the list to hold the feature matrices
        features = []

        # Process each chunk with the gammatone filterbank
        for chunk in chunks:
            chunk_features = []
            for f in freqs:
                # Create the gammatone filter
                b, a = signal.gammatone(f, 'fir', fs=self.samplerate, numtaps=1024)
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

    def normalize_features(self,features):
        medians = np.median(features, axis=2)
        features -= medians[:, :, None]
        means = np.mean(features, axis=2)
        stds = np.std(features, axis=2)
        normalized_features = (features - means[:, :, None]) / stds[:, :, None]
        return normalized_features
    
    def main_process(self):
        chunks = self.chunk_audio(self.audio_data)
        filtered_chunks = self.filter_chunks(chunks)
        weighted_chunks = self.apply_a_weighting(filtered_chunks)
        gammatone_features = self.process_gammatone(weighted_chunks)
        final_features = self.normalize_features(gammatone_features)
        return  final_features


    def plot_features_as_image(self,final_features, num_bands=21, low_freq=400, high_freq=6000):
        num_time_frames = final_features.shape[1]
        
        # Define the time axis extending from 0 to the length of the audio in seconds
        time_axis = np.linspace(0, num_time_frames / self.samplerate, num=num_time_frames)
        
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



if __name__ =='__main__':

# Example Usage
     
    def read_audio(filename):
        waveform, samplerate = torchaudio.load(filename)
        return waveform.numpy(), samplerate  # Convert to numpy array immediately
        
    audio_file = f'/home/prsh7458/Desktop/scratch4/speech_data/motus_saved_noise_reverb_edc/reverberant_speech/reverberant_speech_{random.randint(0,200)}.wav'

    data, samplerate = read_audio(audio_file)
    print(data.shape)

    feature_extractor = Log_spectral_feature_extraction(data,samplerate=samplerate)
    final_features = feature_extractor.main_process()

