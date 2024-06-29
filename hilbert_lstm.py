import librosa
import librosa.display
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy.signal import butter, lfilter, freqz, hilbert, filtfilt
import os
import numpy as np 
from random import shuffle
import matplotlib
import matplotlib.pyplot as plt 
from scipy.io import wavfile
import random


"""A script demonstrating Hilbert transform/power envelope application"""

audio_path =     "/data2/hpcuser/prsh7458/Backupdata/Data/reverberant_speech_ilm"

wav_files = [file for file in os.listdir(audio_path) if file.endswith('.wav')]
random_file = random.choice(wav_files)
file_path = os.path.join(audio_path, random_file)
fs, Rs = wavfile.read(file_path)
Rs = Rs / np.max(np.abs(Rs))



def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


import scipy as sp
# f,t,Sxx = sp.signal.spectrogram(Rs,16000)
# plt.pcolormesh(t,f,10*np.log10(Sxx))
# plt.title('Spectrogram plot')
# plt.show()




fcentre = 125
fd = np.power(2,1/2)
highcut = fcentre *fd
lowcut = fcentre/fd
Fs = fs

b, a = butter_bandpass(lowcut, highcut, Fs, order=3)
y_f125 = sp.signal.filtfilt(b,a,Rs)

fcentre = 250
fd = np.power(2,1/2)
highcut = fcentre *fd
lowcut = fcentre/fd
b, a = butter_bandpass(lowcut, highcut, Fs, order=3)
y_f250 = sp.signal.filtfilt(b,a,Rs)




fcentre = 500
fd = np.power(2,1/2)
highcut = fcentre *fd
lowcut = fcentre/fd
Fs = 16000

b, a = butter_bandpass(lowcut, highcut, Fs, order=3)
y_f500 = sp.signal.filtfilt(b,a,Rs)

fcentre = 1000
fd = np.power(2,1/2)
highcut = fcentre *fd
lowcut = fcentre/fd
Fs = 16000

b, a = butter_bandpass(lowcut, highcut, Fs, order=3)
y_f1k = sp.signal.filtfilt(b,a,Rs)

fcentre = 2000
fd = np.power(2,1/2)
highcut = fcentre *fd
lowcut = fcentre/fd
Fs = 16000

b, a = butter_bandpass(lowcut, highcut, Fs, order=3)
y_f2k = sp.signal.filtfilt(b,a,Rs)

fcentre = 4000
fd = np.power(2,1/2)
highcut = fcentre *fd
lowcut = fcentre/fd
Fs = 16000

b, a = butter_bandpass(lowcut, highcut, Fs, order=3)
y_f4k = sp.signal.filtfilt(b,a,Rs)





fc = 500 # cutoff frequency of the speech envelope  at 30 Hz
N = 5   #filter order
w_L = 2*fc/16000
#lowpas filter tf
b, a = butter(N,w_L, 'low')

e_yt_1 = sp.signal.filtfilt(b,a,np.abs(hilbert(y_f125)))
e_yt_2 = sp.signal.filtfilt(b,a,np.abs(hilbert(y_f250)))
e_yt_3 = sp.signal.filtfilt(b,a,np.abs(hilbert(y_f500)))
e_yt_4 = sp.signal.filtfilt(b,a,np.abs(hilbert(y_f1k)))
e_yt_5 = sp.signal.filtfilt(b,a,np.abs(hilbert(y_f2k)))
e_yt_6 = sp.signal.filtfilt(b,a,np.abs(hilbert(y_f4k)))
env = sp.signal.filtfilt(b,a,np.abs(hilbert(Rs)))


nn = np.arange(0,len(Rs),1)
n = nn[2500:2500+10000]
env1_sl = env[2500:2500+10000]
# fig = plt.figure()
# ax0 = fig.add_subplot(111)
# ax0.plot(timeaxis2,y_resampled, label='interpolated envelope')
# ax0.plot(timeaxis1,y, label='envelope')
fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.plot(n, Rs[2500:2500+10000],'b-', label='signal')
ax0.plot(n,env1_sl,'r',marker = "o",ms=0.5,ls="")
ax0.grid(True)
plt.title('Power envelope of the reverberant speech')
plt.xlabel('Sampling (n)')
plt.ylabel('Gain')
plt.show()




# import scipy as sp
# import torch

# fc = 30# cutoff frequency of the speech envelope  at 30 Hz
# N = 6   #filter order
# w_L = 2*fc/fs
# #lowpas filter tf
# b, a = butter(N,w_L, 'low')


# # fcentre = 500 #Centre Frequency at 125 Hz
# # fd = np.power(2,1/2)
# # highcut = fcentre *fd
# # lowcut = fcentre/fd
# # b, a = butter_bandpass(lowcut, highcut, fs, order=6)
# y = filtfilt(b,a,np.abs(hilbert(Rs)))


# from scipy import signal

# # Assuming `y` is your original signal sampled at 16000 Hz
# original_rate = 16000
# new_rate = 100
# num_original_samples = len(y)

# # Duration of your signal in seconds
# duration = num_original_samples / original_rate

# # Number of samples in the resampled signal
# num_new_samples = int(duration * new_rate)

# # Resampling
# y_resampled = signal.resample(y, num_new_samples)

# timeaxis1  = np.linspace(0,2,32000)
# timeaxis2  = np.linspace(0,2,200)



# fig = plt.figure()
# ax0 = fig.add_subplot(111)
# ax0.plot(timeaxis2,y_resampled, label='interpolated envelope')
# ax0.plot(timeaxis1,y, label='envelope')
# # plt.plot(y)
# ax0.legend()
# plt.show()


# new_Fs = (fs*(size/num_samples))

# print(y_resampled.shape)

# plt.specgram(y_resampled, Fs=100, NFFT=1024, noverlap=5, detrend='mean', mode='psd')
# plt.xlabel('time')
# plt.ylabel('frequency')
# plt.colorbar(format='%+2.0f dB')
# plt.show()


# from scipy.signal import stft
# n_fft = min(100, len(y_resampled))  # n_fft cannot exceed the length of y_resampled
# noverlap = n_fft // 2  # Typically half of n_fft

# # Perform STFT
# frequencies, times, Zxx = stft(y_resampled, fs=new_rate, nperseg=n_fft, noverlap=noverlap)

# # Compute the magnitude of the STFT result
# magnitude = np.abs(Zxx)

# # Convert to dBs
# magnitude_dB = 20 * np.log10(magnitude + 1e-6)  # Adding a small value to avoid log of zero

# # Plot the spectrogram
# plt.figure(figsize=(12, 6))
# plt.pcolormesh(times, frequencies, magnitude_dB, shading='gouraud')
# plt.colorbar(format='%+2.0f dB')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.title('Modulation Spectrogram')
# plt.show()


