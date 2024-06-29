import numpy as np
import os
import re
from pathlib import Path
import torchaudio
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import random
import torch
from scipy.integrate import cumtrapz
from scipy.signal import spectrogram

"""Implementatino of lundeby truncation method"""
class Truncated_RIR_EDC:


    def __init__(self,directory,file_extension,view_EDC=False) :
        self.directory =  directory
        self.file_extension =file_extension
        self.flag = view_EDC


    def load_and_preprocess_audio(self,index,pos_bool = False):

        if pos_bool:
            waveform, sample_rate,cord = self.load_file(index,pos_bool)
        else:
            waveform, sample_rate,room = self.load_file(index,pos_bool)
            
        waveform = waveform[0]  # Ensure single channel (mono)
        if pos_bool:
            return waveform, sample_rate,cord
        else:
            return waveform, sample_rate,room
    
    def sort_numerically(self,file):
        """
        Extracts leading numbers from the filename and returns it for sorting.
        If the filename does not start with a number, returns -1.
        """
        numbers = re.findall('^\d+', file)
        return int(numbers[0]) if numbers else -1
    
    def load_file(self,index,pos_bool):
        # Load all file names with the specified extension
        all_files = [file for file in os.listdir(self.directory) if file.endswith(self.file_extension)]
        if not all_files:
            raise ValueError(f"No files found in {self.directory} with extension {self.file_extension}")
        all_files.sort(key=self.sort_numerically)
        # Select a file using a random index
        # random_index = random.randint(0, len(all_files) - 1)
        file = all_files[index]
        # print("the index is ", self.index)
        file_path = os.path.join(self.directory, file)
        waveform, sample_rate = torchaudio.load(file_path)
        

        if not pos_bool:
            room = self.extract_room_number_from_filename(index)
            return waveform, sample_rate,room
        else:
            cord = self.extract_xyz_from_filename(index)
            return waveform, sample_rate, cord
    

    def extract_xyz_from_filename(self, index):
        """
        Extracts the x, y, z coordinates from the filename at the given index.
        """
        all_files = [file for file in os.listdir(self.directory) if file.endswith(self.file_extension)]
        if not all_files:
            raise ValueError(f"No files found in {self.directory} with extension {self.file_extension}")
        
        all_files.sort(key=self.sort_numerically)
        file = all_files[index]
        # Extract x, y, z coordinates from the filename
        pattern = r'-?\d+\.\d+'
        coords = re.findall(pattern, file)
        if len(coords) >= 3:
            x, y, z = map(float, coords[-3:])  # Get the last three elements as they represent x, y, z
            return [x, y, z]
        else:
            raise ValueError("Filename does not contain valid x, y, z coordinates.")

    def extract_room_number_from_filename(self, index):
        """
        Extracts a number directly from the room name in the filename.
        """
        # List all files in the directory with the specified file extension
        all_files = [file for file in os.listdir(self.directory) if file.endswith(self.file_extension)]
        if not all_files:
            raise ValueError(f"No files found in {self.directory} with extension {self.file_extension}")

        # Ensure the files are sorted numerically if needed
        all_files.sort(key=self.sort_numerically)
        file = all_files[index]

        # Extract the room name from the filename
        room_name = file.split('_')[-1].replace('.wav', '')

        # Extract the numeric part from the room name and convert it to an integer
        # Assuming the format is always HLXXW or HLXXWL where XX is a number
        numeric_part = ''.join(filter(str.isdigit, room_name))
        try:
            return int(numeric_part)
        except ValueError:
            return "Invalid room name format"


    # def resample_waveform(self, waveform, orig_sr, new_sr):
    #     if orig_sr != new_sr:
    #         resampler = T.Resample(orig_sr, new_sr)
    #         waveform = resampler(waveform)
    #     return waveform





    def intlinear(self,x, y):
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


    def lundeby(self,impulse_response, fs):
        """
        Translated MATLAB function 'lundeby' to Python.
        It implements the Lundeby method to determine the truncation point.
        """

        # Calculate the energy of the impulse response
        energy_impulse = impulse_response**2

        # Calculate the noise level of the last 10% of the signal
        rms_db = 10 * np.log10(np.mean(energy_impulse[int(np.round(0.9 * len(energy_impulse))):]) / np.max(energy_impulse))

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
        media_array[media_array < 1e-9] = 1e-9

        media_db = 10 * np.log10(media_array / (np.max(energy_impulse)))

        # Obtain the linear regression for the interval from 0 dB to the mean nearest to rms_db + 10 dB
        r = np.max(np.where(media_db > rms_db + 10))
        if any(media_db[0:r] < rms_db + 10):
            r = np.min(np.where(media_db[0:r] < rms_db + 10))
        if r < 10:
            r = 10

        A, B = self.intlinear(np.array(eixo_tempo[:r]), media_db[:r])
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
            A, B = self.intlinear(np.array(eixo_tempo), media_db)

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

        if self.flag:
            plt.figure()
            energy_impulse = np.where(energy_impulse > 1e-9, energy_impulse, 1e-9)
            plt.plot(np.arange(len(energy_impulse)) / fs, 10 * np.log10(energy_impulse / np.max(energy_impulse)), 
                    label='ETC', linewidth=3)
            plt.plot(np.array(eixo_tempo) / fs, media_db, 'r', drawstyle='steps', label='Median dB', linewidth=3)
            plt.plot(np.arange(1, crossing + 1001) / fs, A + np.arange(1, crossing + 1001) * B, 
                    'g', label='Linear Regression', linewidth=3)
            plt.axhline(y=rms_db, color='.4', linestyle='-', 
                        xmin=(crossing - 1000) / len(energy_impulse), xmax=1.0, label='RMS dB Line', linewidth=3)
            plt.plot(crossing / fs, rms_db, 'yo', markersize=10, label='Crossing Point', linewidth=3)

            # Setting ylabel, xlabel, and title with specific font sizes
            plt.ylabel('ETC in dB', fontsize=20)
            plt.xlabel('t in s', fontsize=20)
            plt.title('Energy Time Curve', fontsize=20)

            # Setting legend with a specific font size
            plt.legend(fontsize=16)

            # Setting the tick params for both axes
            plt.tick_params(axis='x', labelsize=16)
            plt.tick_params(axis='y', labelsize=16)

            plt.show()



        return ponto, C, decay_line, noise_val



    def schroeder(self,x,onset_detection = False):
        """
        Translates the MATLAB schroeder function to Python.
        It calculates the energy decay curve of an audio signal.
        """
        # Square the signal
        if onset_detection:
            onset = self.rir_onset(x)
            x = x[..., onset:]

        # x = self.discard_trailing_zeros(x)
        # x = self.discard_last_n_percent(x, 0.5)
        x_squared = x**2

        # Cumulatively integrate the flipped squared signal and flip it back
        cum_int = np.flip(cumtrapz(np.flip(x_squared, axis=0), axis=0), axis=0)

        # Ensure no zero values
        # cum_int[cum_int < 1e-20] = 1e-20

        # Convert to decibel scale
        y = 10 * np.log10(cum_int+1e-16)

        # Normalize each column by its first value, handle both 1D and 2D cases
        if y.ndim > 1:
            for k in range(y.shape[1]):
                y[:, k] -= y[0, k]
        else:
            y -= y[0]

        # y = self.discard_last_n_percent(y, 0.05)
        return y




    def process(self,index,pos_bool=False):

        if pos_bool:
            rir, fs ,cord = self.load_and_preprocess_audio(index,pos_bool)
        else:
            rir, fs,room = self.load_and_preprocess_audio(index,pos_bool)

        # print("cord is", cord)
        rir = rir.numpy()   # Convert tensor to numpy array
        rir_normalized = rir / np.max(np.abs(rir))
        # print(rir_normalized.shape)
    

        # Estimate noisefloor
        mpd = fs * 0.001  # MinPeakDistance
        mph = 0.7  # MinPeakHeight

        # Find peaks
        peaks, _ = find_peaks(np.abs(rir_normalized), distance=mpd, height=mph)
        if len(peaks) == 0:
            print('Warning: idxD_peak not found')
        # print(peaks[0])
        # Apply the lundeby method (assuming it's already defined)
        idxTruncPoint, SC, decay_line, noise = self.lundeby(rir_normalized[peaks[0]:], fs)
        # print("idxTruncPoint",idxTruncPoint, "SC",SC, "decay_line.shape",decay_line.shape,"decay_line[0] - decay_line[1]",(decay_line[0] - decay_line[1]),"noise",noise)

        # Sample offset calculation
        sample_offset = round(10 / (decay_line[0] - decay_line[1]))
        # print("sample offset",sample_offset)
        end_i = idxTruncPoint + peaks[0] - 1 + (sample_offset)
        # print("end_i",end_i)
        # Ensure end index doesn't exceed the length of rir
        if end_i > len(rir_normalized):
            end_i = len(rir_normalized)

        # Find the index of the beginning of the direct sound
        mph_begin = 0.03
        idxD_begin = np.argmax(np.abs(rir_normalized) > mph_begin * np.max(np.abs(rir_normalized)))

        if idxD_begin >= peaks[0]:
            print('Warning: idxD index error')

        # Calculate Energy Decay Curves (assuming schroeder function is defined)
        # print(rir_normalized.shape)
        EDC_all = self.schroeder(rir_normalized)
        # print(EDC_all.shape)
        EDC_begin_End = self.schroeder(rir_normalized[idxD_begin:end_i])
        EDC_End = self.schroeder(rir_normalized[:end_i])

        # Plotting
        if self.flag:

                    
                # Time axis
                # Here we are using 16000 because later on we resampled the RIR to 16000 fs 
                t = np.arange(len(EDC_all)) / 48000
                t_b_e = np.arange(len(EDC_begin_End)) / 48000
                t_e = np.arange(len(EDC_End)) / 48000

                # Mask for specified dB range
                dbmax = 0
                dbmin = -20
                mask = (EDC_all >= dbmin) & (EDC_all <= dbmax)
                mask_b_e = (EDC_begin_End >= dbmin) & (EDC_begin_End <= dbmax)
                mask_e = (EDC_End >= dbmin) & (EDC_End <= dbmax)

                # Filtered time and EDC values for the fitting range
                x = t[mask]
                x_b_e = t_b_e[mask_b_e]
                x_e = t_e[mask_e]

                y = EDC_all[mask]
                y_begin_end = EDC_begin_End[mask_b_e]
                y_end = EDC_End[mask_e]

                # Perform linear regression using NumPy
                # Adding a column of ones to x for the intercept calculation
                X = np.vstack([x, np.ones(len(x))]).T
                X_b_e = np.vstack([x_b_e, np.ones(len(x_b_e))]).T
                X_e = np.vstack([x_e, np.ones(len(x_e))]).T
                # Linear least squares fitting
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                beta_b_e, _, _, _ = np.linalg.lstsq(X_b_e, y_begin_end, rcond=None)
                beta_e, _, _, _ = np.linalg.lstsq(X_e, y_end, rcond=None)

                slope, intercept = beta
                slope_b_e, intercept_b_e = beta_b_e
                slope_e, intercept_e = beta_e
                
                # Calculate fitted EDC values across the entire range for visualization
                fitted_y = slope * t + intercept
                fitted_y_b_e = slope_b_e * t + intercept_b_e
                fitted_y_e = slope_e * t + intercept_e
                
                # Calculate T60 using the slope (time to decay 60 dB)
                T60 = -60 / slope

                
                plt.figure()
                fitted_lines_true, T_true = fitted_y, T60
                mask_lines_true = (fitted_lines_true >= -60) 
                selected_EDT_true = fitted_lines_true[mask_lines_true]
                mask_lines_true_b_e = (fitted_y_b_e >= -60) 
                selected_EDT_true_b_e = fitted_y_b_e[mask_lines_true_b_e]
                mask_lines_true_e = (fitted_y_e >= -60) 
                selected_EDT_true_e = fitted_y_e[mask_lines_true_e]

                # Plotting the lines with labels and specific linestyle
                plt.plot(t[mask_lines_true], selected_EDT_true, label='Regression EDC_all', linestyle='--', linewidth=3)
                plt.plot(t[mask_lines_true_b_e], selected_EDT_true_b_e, label='Regression EDC_begin_End', linestyle='--', linewidth=3)
                plt.plot(t[mask_lines_true_e], selected_EDT_true_e, label='Regression EDC_End', linestyle='--', linewidth=3)
                plt.plot(t, EDC_all, label='EDC_all', linewidth=3)
                plt.plot(t_b_e, EDC_begin_End, label='EDC_begin_End', linewidth=3)
                plt.plot(t_e, EDC_End, label='EDC_End', linewidth=3)

                # Setting ylabel, xlabel, title and their font sizes
                plt.ylabel("dB", fontsize=24)
                plt.xlabel("Time", fontsize=24)
                plt.title("Schroeder's Energy Decay Curves", fontsize=28)

                # Setting legend font size
                plt.legend(fontsize=20)

                # Setting the tick params for both axes
                plt.tick_params(axis='x', labelsize=24)
                plt.tick_params(axis='y', labelsize=24)

                plt.show()


        if pos_bool:
            return EDC_all,EDC_begin_End,noise,rir_normalized,fs,cord  #Here I have used Full RIR instead of cropped one
        else:
            return EDC_all,EDC_begin_End,noise,rir_normalized,fs ,room
        # return EDC_all,EDC_begin_End,noise,rir_normalized[idxD_begin:end_i],fs  #Make sure to use apt RIR FOR THE EDC

    def resample_waveform(self, waveform, orig_sr, new_sr):
        if orig_sr != new_sr:
            resampler = T.Resample(orig_sr, new_sr)
            waveform = resampler(waveform)
        return waveform


    def discard_last_n_percent(self,edc: torch.Tensor, n_percent: float) -> torch.Tensor:
        # Discard last n%
        last_id = int(np.round((1 - n_percent / 100) * edc.shape[-1]))
        out = edc[..., 0:last_id]

        return out
    
    def discard_trailing_zeros(self,rir: np.ndarray) -> np.ndarray:
        # find first non-zero element from back
        last_above_thres = rir.shape[-1] - np.argmax((np.flip(rir, axis=-1) != 0).squeeze().astype(int))

        # discard from that sample onwards
        out = rir[..., :last_above_thres]

        return out

    def rir_onset(self,rir):
        # Calculate the spectrogram
        f, t, Sxx = spectrogram(rir, nperseg=64, noverlap=60)

        # Sum across the frequency bins to get the windowed energy
        windowed_energy = np.sum(Sxx, axis=0)

        # Compute the delta energy
        delta_energy = windowed_energy[1:] / (windowed_energy[:-1] + 1e-16)

        # Find the index of the highest energy change
        highest_energy_change_window_idx = np.argmax(delta_energy)

        # Calculate the onset
        onset = int((highest_energy_change_window_idx - 2) * 4 + 64)

        return onset

if __name__ == "__main__":
    
    import os
    os.system('clear')
    rir_data_dir = "/home/prsh7458/Desktop/scratch4/speech_data/loc/position/HL00W/BC"
    extension = ".wav"
    RIR_files = [file for file in os.listdir(rir_data_dir) if file.endswith(".wav")]
    RIR_files.sort()
    # print(RIR_files[index])


    def generate_synthetic_edc_torch(t_vals, a_vals, noise_level, time_axis, device='cpu', compensate_uli=True) -> torch.Tensor:
        """Generates an EDC from the estimated parameters."""
        # Ensure t_vals, a_vals, and noise_level are properly shaped


        # Calculate decay rates based on the requirement that after T60 seconds, the level must drop to -60dB
        tau_vals = torch.log(torch.tensor([1e6], device=device)) / t_vals

        # Calculate exponentials from decay rates
        time_vals = -time_axis * tau_vals  # batchsize x 16000 which is the length of tim axis
        exponentials = torch.exp(time_vals)

        # # Account for limited upper limit of integration, if needed

        exp_offset = 0

        # # Multiply exponentials with their amplitudes (a_vals)
        edcs = a_vals * (exponentials - exp_offset)

        # # Add noise (scaled linearly over time)

        noise = noise_level * torch.linspace(1, 0, time_axis.shape[0], device=device)
        print(noise.shape)
        edc = torch.tensor(edcs).clone().detach()+ torch.tensor(noise).clone().detach()


        # plt.plot(edc[1,:].cpu().detach().numpy())
        # plt.show()

        return edc 




    for i in range (1):
        # index = random.randint(0,len(RIR_files)-1)
        #multiple of 
        
        ## 5i is SiL , 5*i+1 is fc, +2 is FR, +3 is SiR , +4 is BC 
        i = 2
        index = 5*i+1
        # print(RIR_files[index])
        truncation_process = Truncated_RIR_EDC(rir_data_dir,extension,view_EDC=True)
        RIR_files.sort(key=truncation_process.sort_numerically)
        print(RIR_files[index])


        EDC_all, EDC_begin_End, noise, rir_normalized, fs, __ = truncation_process.process(index)

        # if EDC_begin_End.shape[0] < 103854:
        #         padding_length = 103854 - EDC_begin_End.shape[0]
        #         print("Before Padding",EDC_begin_End.shape[0])
        #         EDC_begin_End_padded = np.pad(EDC_begin_End, (0, padding_length), 'constant', constant_values=(EDC_begin_End[-1]))
        #         print("After Padding",EDC_begin_End_padded.shape[0])


        time_axis = torch.linspace(0, (EDC_all.shape[0] - 1) / fs, EDC_all.shape[0])


        # time_axis_padded = torch.linspace(0, (EDC_begin_End_padded.shape[0] - 1) / fs, EDC_begin_End_padded.shape[0])

        print(" EDC: ",EDC_all.shape,"and time axis shape",EDC_all.shape)


        desired_output = 16000

        schroeder_decays_db = torch.nn.functional.interpolate(torch.from_numpy(EDC_begin_End).unsqueeze(0).unsqueeze(0), size=desired_output,
                                                                    mode='linear', align_corners=True)

        fs_new = int((desired_output*fs)/(EDC_all.shape[0]))
        time_axis_interpolated = torch.linspace(0, (schroeder_decays_db.shape[2] - 1) / fs_new, schroeder_decays_db.shape[2])


        # t_vals = torch.ones(2,1)*2.6
        # a_vals = torch.ones(2,1)*1.5
        # noise_level = torch.ones(2,1)*9.7e-9

        # # random_t_vals = torch.rand_like(t_vals)
        # random_t_vals = (t_vals)
        # random_a_vals = (a_vals)
        # random_noise_level = torch.rand_like(noise_level)

        # print(random_t_vals,random_a_vals,random_noise_level)



        # edc = generate_synthetic_edc_torch(random_t_vals, random_a_vals, random_noise_level, time_axis)

#       posprocessing given time_axis_interpolated 



        # def post_process(schroeder_decays_db,time_axis_interpolated,fs):
        #     total_duration = time_axis_interpolated[-1] - time_axis_interpolated[0]

        #     # Count the number of samples
        #     num_samples = len(time_axis_interpolated)

        #     # Calculate the new sampling frequency
        #     fs_new = num_samples / total_duration

        #     original_size = len(time_axis_interpolated) * fs / fs_new

        #     schroeder_decays_db_post = torch.nn.functional.interpolate(schroeder_decays_db, size=int(original_size), # error
        #                                                                 mode='linear', align_corners=True)

        #     time_axis_interpolated_post = torch.linspace(0, (schroeder_decays_db_post.shape[2] - 1) / fs, schroeder_decays_db_post.shape[2])

        #     return  time_axis_interpolated_post, schroeder_decays_db_post
    



        # # time_axis_interpolated_post, schroeder_decays_db_post = post_process(schroeder_decays_db,time_axis_interpolated,fs)
        # time_axis_all = torch.linspace(0, (EDC_all.shape[0] - 1) / fs, EDC_all.shape[0])
        # time_b_e = torch.linspace(0, (EDC_begin_End.shape[0] - 1) / fs, EDC_begin_End.shape[0])
        




        # fig, ax = plt.subplots()
        # # plt.plot(time_axis_padded,EDC_begin_End_padded, label='EDC_with padding')
        # # plt.legend()
        # # plt.show()

        # ax.plot(time_axis_all,EDC_all, label='EDC_full')
        # ax.plot(time_b_e,EDC_begin_End, label='EDC_lundeby')
        # plt.legend()
        # plt.show()


        # print(schroeder_decays_db.shape)
        # # plt.figure()
        # # plt.plot(time_axis,edc[1,:].cpu().detach().numpy())
        # # plt.plot(time_axis_interpolated,schroeder_decays_db.squeeze(0).squeeze(0), label='EDC_with interpolation')
        # # plt.legend()
        

        # # plt.plot(time_axis_interpolated_post,schroeder_decays_db_post.squeeze(0).squeeze(0), label='EDC post processing')
        # # plt.legend()
        # plt.show()



    # # Initialize a list to store the EDC sizes
    # edc_sizes = []

    # # Iterate over all RIR files
    # for file in range(len(RIR_files)):
    #     EDC_all, EDC_begin_End, noise, rir_normalized, fs = truncation_process.process(file)

    #     rir_normalized = torch.from_numpy(rir_normalized)
    #     # Resample RIR if necessary
    #     rir_waveform = truncation_process.resample_waveform(rir_normalized, 48000, 16000)
    #     rir_waveform.numpy()
    #     edc_sizes.append(EDC_begin_End.shape[0])

    # # Calculate the maximum EDC size
    # max_edc_size = np.max(edc_sizes)

    # print("Max samples of EDC are:", max_edc_size) #103854
    # print("Max RT is:", max_edc_size / int(48000))  #2.16 for motus dataset


