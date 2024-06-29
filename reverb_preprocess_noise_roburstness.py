import os
import re
import random
import torchaudio
import scipy.signal
import torch
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np
import torchaudio.transforms as T
from scipy.signal import butter, lfilter
from truncation import Truncated_RIR_EDC
import scipy.io  # For saving MATLAB files

"""A script to prepare dataset and save in mat files"""

class ReverbSpeech:

    """
    Process the audio files to create a fixed-length reverberant speech segment.

    This method performs several steps:
    1. Loads and preprocesses audio files (speech).
    2. Loads truncated RIRs, EDCs using lundeby method.
    3. Ensure they are mono and have matching sample rates.
    4. Creates reverberant speech by convolving the speech waveform with the RIR waveform.
    5. Extracts a fixed-length segment from the reverberant speech using a VAD model.

    Returns:
        np.ndarray: A numpy array containing the fixed-length speech segment with reverberation added. 
        np.ndarray: A numpy array containing the room impulse response (RIR) waveform. 
        np.ndarray: A numpy array containing the full Energy Decay Curve (EDC). 
        np.ndarray: A numpy array containing the truncated Energy Decay Curve (EDC) computed using the Lundeby method.     
        np.ndarray: A numpy array containing the noise floor estimation by Lundeby method.

    Note:
        The plot_waveform_and_spectrogram method should be called separately for visualization.
        There is a flag as well to see the plots for EDC as well 
    """


    def __init__(self, librispeech_data_dir, rir_data_dir, model_dir,view_EDC=False):
        self.librispeech_data_dir = librispeech_data_dir
        self.rir_data_dir = rir_data_dir
        self.model_dir = model_dir
        self.view_EDC = view_EDC
        self.model_path = os.path.join(model_dir, "silero_vad.pth")
        self.model, self.utils = self.setup_vad_model()

    def load_and_preprocess_audio(self, directory, file_extension,index):
        waveform, sample_rate = self.load_file(directory, file_extension,index)
        waveform = waveform[0]  # Ensure single channel (mono)
        return waveform, sample_rate
    

    
    def sort_numerically(self,file):
        """
        Extracts leading numbers from the filename and returns it for sorting.
        If the filename does not start with a number, returns -1.
        """
        numbers = re.findall('^\d+', file)
        return int(numbers[0]) if numbers else -1

    def load_file(self, directory, file_extension,index):
        # Load all file names with the specified extension
        all_files = [file for file in os.listdir(directory) if file.endswith(file_extension)]
        if not all_files:
            raise ValueError(f"No files found in {directory} with extension {file_extension}")
        
        all_files.sort(key=self.sort_numerically)


        file = all_files[index]
        file_path = os.path.join(directory, file)
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate

    def resample_waveform(self, waveform, orig_sr, new_sr):
        if orig_sr != new_sr:
            resampler = T.Resample(orig_sr, new_sr)
            waveform = resampler(waveform)
        return waveform
    def bandpass_filter(self, signal, lowcut, highcut, sample_rate, order=5):
        # Ensure cutoff frequencies are less than the Nyquist frequency
        nyquist = 0.5 * sample_rate


        # Normalize the frequencies
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = butter(order, [low, high], btype='band')
        filtered_signal = lfilter(b, a, signal)
        return filtered_signal


    def create_reverberant_speech(self, speech_waveform, rir_waveform):
        return scipy.signal.fftconvolve(speech_waveform, rir_waveform, mode='full')

    def setup_vad_model(self):
        if not os.path.exists(self.model_path):
            # print("Downloading the VAD model...")
            model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False)
            torch.save(model.state_dict(), self.model_path)
        else:
            # print("Using the cached VAD model.")
            model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        return model, utils

    def extract_fixed_length_segment(self, reverberant_speech, sample_rate, segment_length=1.024):  #1.024 earlier
        (get_speech_timestamps, _, _, _, _) = self.utils
        speech_timestamps = get_speech_timestamps(reverberant_speech, self.model, sampling_rate=sample_rate)
        window_size_samples = int(segment_length * sample_rate)
        valid_windows = [ts for ts in speech_timestamps if ts['end'] - ts['start'] > window_size_samples]

        if not valid_windows:
            print("No relevant window found in this audio, hence selecting longest window")
            # Find the longest window in speech_timestamps
            longest_window = max(speech_timestamps, key=lambda ts: ts['end'] - ts['start'])
            start_index = int(longest_window['start'])
            end_index = min(start_index + window_size_samples, int(longest_window['end']))
            
            return reverberant_speech[start_index:end_index], False

        selected_window = random.choice(valid_windows)
        start_index = int(selected_window['start'])
        end_index = start_index + window_size_samples
        return reverberant_speech[start_index:end_index], True

    def plot_waveform_and_spectrogram(self, speech_window, sample_rate=16000):
        time_axis = np.linspace(0, len(speech_window) / sample_rate, len(speech_window))
        frequencies, times, Sxx = spectrogram(speech_window, fs=sample_rate)

        plt.figure(figsize=(12, 8))  # Increased figure size to provide more room vertically

        # Plot the waveform
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, speech_window)
        plt.title('Waveform', fontsize=20)
        plt.ylabel('Amplitude', fontsize=20)
        plt.tick_params(axis='x', labelsize=16)  # Set font size for x-axis
        plt.tick_params(axis='y', labelsize=16)  # Set font size for y-axis

        # Hide x-axis labels for the top plot to prevent overlap with the bottom plot
        plt.gca().xaxis.set_ticklabels([])

        # Plot the spectrogram
        plt.subplot(2, 1, 2)
        im = plt.imshow(10 * np.log10(Sxx), aspect='auto', origin='lower',
                        extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
        plt.title('Spectrogram', fontsize=20)
        plt.xlabel('Time (s)', fontsize=20)
        plt.ylabel('Frequency (Hz)', fontsize=20)
        plt.tick_params(axis='x', labelsize=16)  # Set font size for x-axis
        plt.tick_params(axis='y', labelsize=16)  # Set font size for y-axis

        # Add the colorbar with a bit of padding to prevent overlap
        cbar = plt.colorbar(im, orientation='horizontal', pad=0.35)
        cbar.set_label('dB', fontsize=12)

        # Adjust layout to prevent overlap and ensure everything fits well
        plt.tight_layout(pad=2.0, h_pad=2.0)  # Add padding between figures and around

        plt.show()

    def generate_bandpassed_noise(self, duration, sr, dB_level):
        # Generate white noise
        noise = np.random.normal(0, 1, int(sr * duration))
        
        # Apply a bandpass filter
        b, a = butter(5, [0.01, 0.99], btype='band')
        filtered_noise = lfilter(b, a, noise)
        
        # Calculate scaling factor for the desired dB level
        # dB_level should be a negative number to reduce the noise level
        # scaling_factor = 10**(dB_level / 20)
        
        # Scale the filtered noise
        # scaled_noise = filtered_noise * scaling_factor
        
        return filtered_noise # scaled_noise
    

    def process(self,speech_index,rir_index,dB_val,pos_bool= False):
        # Load and preprocess audio files
        # speech_waveform, speech_sample_rate = self.load_and_preprocess_audio(self.librispeech_data_dir, ".flac",speech_index)
        # rir_waveform, rir_sample_rate = self.load_and_preprocess_audio(self.rir_data_dir, ".wav")

        guassian_noise = self.generate_bandpassed_noise(3, 16000, dB_val)
        

        truncation_process = Truncated_RIR_EDC(self.rir_data_dir, ".wav")
        if pos_bool:
            EDC_all,EDC_begin_End,noise,rir_normalized,rir_sample_rate,cord = truncation_process.process(rir_index,pos_bool) #change it back to index
        else:
            EDC_all,EDC_begin_End,noise,rir_normalized,rir_sample_rate,room = truncation_process.process(rir_index,pos_bool)


        rir_normalized = torch.from_numpy(rir_normalized)
        EDC_begin_End = torch.from_numpy(EDC_begin_End)
        noise = torch.from_numpy(np.array(noise))

        # Resample RIR if necessary
        rir_waveform = self.resample_waveform(rir_normalized, rir_sample_rate, 16000)

        # rir_waveform = self.bandpass_filter(rir_waveform, 10, 7200, speech_sample_rate)
        # speech_waveform = self.bandpass_filter(speech_waveform, 10, 7200, speech_sample_rate)

        # Create reverberant speech
        reverberant_guassian_noise= self.create_reverberant_speech(guassian_noise, rir_waveform)

        # Extract fixed-length segment from the reverberant speech
        # speech_window,exist = self.extract_fixed_length_segment(reverberant_speech, speech_sample_rate)
        reverberant_guassian_noise = reverberant_guassian_noise[0:16384]
        reverberant_guassian_noise = reverberant_guassian_noise/(max(abs(reverberant_guassian_noise)))
        # plt.plot(reverberant_guassian_noise)
        # plt.show()

        if pos_bool:
            return reverberant_guassian_noise,False,rir_waveform,EDC_all,EDC_begin_End,noise,16000,cord
        else:
            return reverberant_guassian_noise,False,rir_waveform,EDC_all,EDC_begin_End,noise,16000,room

if __name__ == "__main__":
    import os
    os.system('clear')
    
    librispeech_data_dir = "/home/prsh7458/Desktop/scratch4/noise_roburstness_test/speech_files"
    model_dir = "/home/prsh7458/work/R2D/edc-estimation/silero_vadsilero_vad"
    base_rir_dir = "/home/prsh7458/Desktop/scratch4/noise_roburstness_test/room_data"


        # Room locations and rooms
    room_locations = ["BC", "FC", "FR", "SiL", "SiR"]
    # rooms = ["HL02WP", "HL04W"]  # Add more rooms as needed
    rooms = ["HL00W", "HL01W", "HL02WL","HL02WP", "HL03W", "HL04W", "HL05W", "HL06W", "HL08W"]  # Add more rooms as needed

    for i,dB_val in enumerate([-50,-40,-30,-20,-10]):
        base_output_dir = os.path.join("/home/prsh7458/Desktop/scratch4/noise_roburstness_test/noisy_dataset",str(dB_val)) 
        os.makedirs(base_output_dir, exist_ok=True)
        for room in rooms:
            for loc in room_locations:
                # Update directories for each room and location
                rir_data_dir = os.path.join(base_rir_dir, room, loc)
                output_dir_speech = os.path.join(base_output_dir, room, loc, "reverberant_noise_ilm")
                output_dir_edc = os.path.join(base_output_dir, room, loc, "EDC_ilm")
                output_dir_noise = os.path.join(base_output_dir, room, loc, "noise_ilm")
                os.makedirs(output_dir_speech, exist_ok=True)
                os.makedirs(output_dir_edc, exist_ok=True)
                os.makedirs(output_dir_noise, exist_ok=True)
                

                # File listing
                speech_files = [file for file in os.listdir(librispeech_data_dir) if file.endswith(".flac")]
                RIR_files = [file for file in os.listdir(rir_data_dir) if file.endswith(".wav")]

                print(f"Processing for room {room}, location {loc}:")
                print(f"Total number of speech files: {len(speech_files)}, RIR files: {len(RIR_files)}")

                # Create an instance of ReverbSpeech for processing
                reverb_speech_processor = ReverbSpeech(librispeech_data_dir, rir_data_dir, model_dir, view_EDC=False)

                # Process files
                num_speech_files = len(speech_files)
                num_rir_files = len(RIR_files)

                for rir_index in range(num_rir_files):
                    for speech_index in range(num_speech_files):

                        # Process the files
                        reverberant_noise, exist, rir_waveform, EDC_all, EDC_begin_End, noise, sample_rate,var  = reverb_speech_processor.process(speech_index, rir_index,dB_val = dB_val,pos_bool=False)  # Adjust this call as per your process function

                        # Calculate the unique index for naming
                        unique_index = rir_index * num_speech_files + speech_index

                        # Define the file paths using the unique index
                        output_file_path_rs = os.path.join(output_dir_speech, f"reverberant_noise_{unique_index}.wav")
                        output_file_path_edc_true = os.path.join(output_dir_edc, f"EDC_begin_end_{unique_index}.mat")
                        output_file_path_noise = os.path.join(output_dir_noise, f"noise_{unique_index}.mat")


                        # Save the reverberant speech to a WAV file
                        torchaudio.save(output_file_path_rs, torch.from_numpy(reverberant_noise).float().unsqueeze(0), sample_rate)
                        scipy.io.savemat(output_file_path_edc_true, {'EDC_begin_end': np.array(EDC_all)}, do_compression=True)  #changed to full EDDC
                        scipy.io.savemat(output_file_path_noise, {'noise': np.array(noise)}, do_compression=True)


                        # Print messages using the unique index
                        print(f"Saved EDC data for unique index {unique_index} to {output_file_path_edc_true}")
                        print(f"Saved reverberant speech for unique index {unique_index} to {output_file_path_rs}")
                        print(f"Saved noise for unique index {unique_index} to {output_file_path_noise}")
                        # print(f"Saved room for unique index {unique_index} to {output_file_path_room}")

