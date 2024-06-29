import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
import h5py 
import scipy
import os
import random
from scipy.io import wavfile

"""A script demonstrating simple Hilbert transform application"""


duration = 1.0
fs = 400.0
samples = int(fs*duration)
t = np.arange(32000) / 16000



signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )



analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) /(2.0*np.pi) * fs)


# audio_path =     "/data2/hpcuser/prsh7458/Backupdata/Data/reverberant_speech_ilm"


audio_path =    "/home/prsh7458/Desktop/scratch4/validation_dataset/r3vival_location/matlab_data/all_data/reverb_speech_data.mat" 


def read_mat_file(file_path, variable_name):
    with h5py.File(file_path, 'r') as file:
        data = np.array(file[variable_name])
    return data




# wav_files = [file for file in os.listdir(audio_path) if file.endswith('.wav')]

wav_files = read_mat_file(audio_path, 'allAudioData')
idx = random.randrange(wav_files.shape[1])
# random_file = random.choice(wav_files)
# file_path = os.path.join(audio_path, random_file)
# sample_rate, Rs = wavfile.read(file_path)
Rs = wav_files[:, idx] / np.max(np.abs(wav_files[:, idx]))  # Added to give normalized RIRs as an input but already normalized


# Apply Hilbert transform using SciPy and convert back to tensor
Rs_hilbert_numpy = scipy.signal.hilbert(Rs)  # Apply Hilbert transform
Rs_hilbert = (np.abs(Rs_hilbert_numpy))
Rs_hilbert = Rs_hilbert / np.max(np.abs(Rs_hilbert))




instantaneous_phase = np.unwrap(np.angle(Rs_hilbert_numpy))
instantaneous_frequency = (np.diff(instantaneous_phase) /(2.0*np.pi) * 16000)

samples = 2500
fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.plot(t[500:500+samples],Rs[500:500+samples], label='signal')
ax0.plot(t[500:500+samples], Rs_hilbert[500:500+samples], label='envelope')
ax0.set_xlabel("Time in seconds",fontsize=28)
ax0.set_ylabel("Normalized Amplitude",fontsize=28)
plt.ylim([-1,1])
ax0.tick_params(axis='x', labelsize=24) 
ax0.tick_params(axis='y', labelsize=24) 
ax0.legend(fontsize=28)
# ax1 = fig.add_subplot(212)
# ax1.plot( t[1:], instantaneous_frequency)
# ax1.set_xlabel("time in seconds")
plt.show()
