import numpy as np
import os
from pathlib import Path
import torchaudio
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import random
import torch

"""A script to preprocess RIRs and make EDCs with an option to implement lundeby truncation"""

def load_and_preprocess_audio(self, directory, file_extension):
    waveform, sample_rate = self.load_random_file(directory, file_extension)
    waveform = waveform[0]  # Ensure single channel (mono)
    return waveform, sample_rate

def load_random_file(self, directory, file_extension):
    # Load all file names with the specified extension
    all_files = [file for file in os.listdir(directory) if file.endswith(file_extension)]
    if not all_files:
        raise ValueError(f"No files found in {directory} with extension {file_extension}")

    # Select a file using a random index
    # random_index = random.randint(0, len(all_files) - 1)
    file = all_files[self.index]
    file_path = os.path.join(directory, file)
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def resample_waveform(self, waveform, orig_sr, new_sr):
    if orig_sr != new_sr:
        resampler = T.Resample(orig_sr, new_sr)
        waveform = resampler(waveform)
    return waveform





def intlinear(x, y):
    """
    It computes the linear regression coefficients A and B for data x and y.
    """
    mx = np.mean(x)
    my = np.mean(y)
    mx2 = np.mean(x**2)
    my2 = np.mean(y**2)
    mxy = np.mean(x * y)
    A = (mx2 * my - mx * mxy) / (mx2 - mx**2)
    B = (mxy - mx * my) / (mx2 - mx**2)

    return A, B


def lundeby(impulse_response, fs, flag=0):
    """
    Translated MATLAB function 'lundeby' to Python.
    It implements the Lundeby method to determine the truncation point.
    """

    # Calculate the energy of the impulse response
    energy_impulse = impulse_response**2

    # Calculate the noise level of the last 10% of the signal
    rms_db = 10 * np.log10(np.mean(energy_impulse[int(0.9 * len(energy_impulse)):]) / np.max(energy_impulse))

    # Divide into intervals and get the mean
    t = int(np.floor(len(energy_impulse) / (fs * 0.01)))
    v = int(np.floor(len(energy_impulse) / t))
    media = []
    eixo_tempo = []
    for n in range(1, t + 1):
        media.append(np.mean(energy_impulse[(n - 1) * v:n * v]))
        eixo_tempo.append(np.ceil(v / 2) + (n - 1) * v)
    media_array = np.array(media)
    # Replace zero or near-zero values with a small positive value
    media_array[media_array < 1e-10] = 1e-10

    media_db = 10 * np.log10(media_array / (np.max(energy_impulse) + 1e-10))

    # Obtain the linear regression for the interval from 0 dB to the mean nearest to rms_db + 10 dB
    r = np.max(np.where(media_db > rms_db + 10))
    if any(media_db[0:r] < rms_db + 10):
        r = np.min(np.where(media_db[0:r] < rms_db + 10))
    if r < 10:
        r = 10

    A, B = intlinear(np.array(eixo_tempo[:r]), media_db[:r])
    crossing = (rms_db - A) / B

    # Start the iterative process
    erro = 1
    INTMAX = 50
    vezes = 1
    while erro > 0.0001 and vezes <= INTMAX:
        # Recalculate the intervals
        p = 5
        delta = abs(10 / B)
        v = int(np.floor(delta / p))
        t = int(np.floor(len(energy_impulse[0:int(crossing - delta)]) / v))
        t = max(t, 2)

        media = [np.mean(energy_impulse[(n - 1) * v:n * v]) for n in range(1, t + 1)]
        eixo_tempo = [np.ceil(v / 2) + (n - 1) * v for n in range(1, t + 1)]
        media_db = 10 * np.log10(np.array(media) / np.max(energy_impulse))

        # Recalculate linear regression
        A, B = intlinear(np.array(eixo_tempo), media_db)

        # Calculate new noise mean
        noise = energy_impulse[int(crossing + delta):] if len(energy_impulse[int(crossing + delta):]) >= int(0.1 * len(energy_impulse)) else energy_impulse[int(0.9 * len(energy_impulse)):]
        rms_db = 10 * np.log10(np.mean(noise) / np.max(energy_impulse))

        # Recalculate the crossing point
        new_crossing = (rms_db - A) / B
        erro = abs(crossing - new_crossing) / crossing
        crossing = new_crossing
        vezes += 1

    # Final calculations
    if crossing > len(energy_impulse):
        ponto = len(energy_impulse)
    else:
        ponto = int(crossing)
    
    C = np.max(energy_impulse) * 10**(A / 10) * np.exp(B / 10 / np.log10(np.exp(1)) * ponto) / (-B / 10 / np.log10(np.exp(1)))
    decay_line = A + np.arange(1, ponto + 1001) * B
    noise_val = 10 * np.log10(np.mean(noise))

    if flag == 1:
        plt.figure()
        energy_impulse = np.where(energy_impulse > 1e-10, energy_impulse, 1e-10)
        plt.plot(np.arange(len(energy_impulse)) / fs, 10 * np.log10(energy_impulse / np.max(energy_impulse)), label='ETC')
        plt.plot(np.array(eixo_tempo) / fs, media_db, 'r', drawstyle='steps', label='Median dB',linewidth=3)
        plt.plot(np.arange(1, crossing + 1001) / fs, A + np.arange(1, crossing + 1001) * B, 'g', label='Linear Regression',linewidth=3)
        plt.axhline(y=rms_db, color='.4', linestyle='-', xmin=(crossing - 1000) / len(energy_impulse), xmax=1.0, label='RMS dB Line',linewidth=3)
        plt.plot(crossing / fs, rms_db, 'yo', markersize=10, label='Crossing Point')
        plt.xlabel('t in s',fontsize=28)
        plt.ylabel('ETC in dB',fontsize=28)
        plt.tick_params(axis='x', labelsize=24)  
        plt.tick_params(axis='y', labelsize=24)  
        plt.legend(fontsize=24)
        plt.show()


    return ponto, C, decay_line, noise_val



def schroeder(x):
    """
    Translates the MATLAB schroeder function to Python.
    It calculates the energy decay curve of an audio signal.
    """
    # Square the signal
    x_squared = x**2

    # Cumulatively integrate the flipped squared signal and flip it back
    cum_int = np.flip(np.cumsum(np.flip(x_squared, axis=0), axis=0), axis=0)

    # Ensure no zero values
    cum_int[cum_int < 1e-10] = 1e-10

    # Convert to decibel scale
    y = 10 * np.log10(cum_int)

    # Normalize each column by its first value, handle both 1D and 2D cases
    if y.ndim > 1:
        for k in range(y.shape[1]):
            y[:, k] -= y[0, k]
    else:
        y -= y[0]

    return y




if __name__ == "__main__":
    import os
    os.system('clear')
    rir_data_dir = "/home/prsh7458/Desktop/scratch4/RIR_dataset_room_test/IR_DATA_HL02WP"
    RIR_files = [file for file in os.listdir(rir_data_dir) if file.endswith(".wav")]

    max_number = (len(RIR_files))

    random_index = random.randint(0, max_number-1)

    file = RIR_files[1]
    file_path = os.path.join(rir_data_dir, file)

    rir, fs = torchaudio.load(file_path)
    rir = rir.numpy()  # Convert tensor to numpy array
    # Use only omni channel
    if len(rir.shape) > 1:
        rir = rir[0:1, :]


    rir_normalized = rir / np.max(np.abs(rir))
    rir_normalized = rir_normalized[0, :]
    print(rir_normalized.shape)
   
    # max_val = np.max(np.abs(rir))
    # print("Maximum absolute value of rir:", max_val)
    # if max_val == 0:
    #     print("Warning: Maximum value is zero, normalization will result in NaN values")
    # if np.isnan(max_val):
    #     print("Warning: rir contains NaN values")





    # Estimate noisefloor
    mpd = fs * 0.001  # MinPeakDistance
    mph = 0.7  # MinPeakHeight

    # Find peaks
    peaks, _ = find_peaks(np.abs(rir_normalized), distance=mpd, height=mph)
    if len(peaks) == 0:
        print('Warning: idxD_peak not found')
    print(peaks[0])
    # Apply the lundeby method (assuming it's already defined)
    idxTruncPoint, SC, decay_line, noise = lundeby(rir_normalized[peaks[0]:], fs, 1)

    # Sample offset calculation
    sample_offset = round(10 / (decay_line[1] - decay_line[0]))
    end_i = idxTruncPoint + peaks[0] - 1 + sample_offset

    # Ensure end index doesn't exceed the length of rir
    if end_i > len(rir_normalized):
        end_i = len(rir_normalized)

    # Find the index of the beginning of the direct sound
    mph_begin = 0.03
    idxD_begin = np.argmax(np.abs(rir_normalized) > mph_begin * np.max(np.abs(rir_normalized)))

    if idxD_begin >= peaks[0]:
        print('Warning: idxD index error')

    # Calculate Energy Decay Curves (assuming schroeder function is defined)
    EDC_all = schroeder(rir_normalized)
    EDC_begin_End = schroeder(rir_normalized[idxD_begin:end_i])
    EDC_End = schroeder(rir_normalized[0:end_i])

    # Plotting
    time_axis = np.linspace(0,len(EDC_all)/fs,len(EDC_all))
    time_axis_EDC_begin_End = np.linspace(0,len(EDC_begin_End)/fs,len(EDC_begin_End))
    time_axis_EDC_End = np.linspace(0,len(EDC_End)/fs,len(EDC_End))
    plt.figure()
    plt.plot(time_axis,EDC_all, label='EDC_all',linewidth=3)
    plt.plot(time_axis_EDC_begin_End,EDC_begin_End, label='EDC_begin_End',linewidth=3)
    plt.plot(time_axis_EDC_End,EDC_End, label='EDC_End',linewidth=3)
    plt.xlabel('Time (secs)', fontsize=28)
    plt.ylabel('Amplitude (dB)', fontsize=28)
    plt.tick_params(axis='x', labelsize=24)  
    plt.tick_params(axis='y', labelsize=24)  
    plt.legend(fontsize=24)
    plt.show()
