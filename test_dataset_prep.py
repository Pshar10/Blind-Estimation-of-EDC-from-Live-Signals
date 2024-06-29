import os
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
import re

"""Prepare dataset for testing"""
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

    def extract_fixed_length_segment(self, reverberant_speech, sample_rate, segment_length=1.024):
        (get_speech_timestamps, _, _, _, _) = self.utils
        speech_timestamps = get_speech_timestamps(reverberant_speech, self.model, sampling_rate=sample_rate)
        window_size_samples = int(segment_length * sample_rate)
        valid_windows = [ts for ts in speech_timestamps if ts['end'] - ts['start'] > window_size_samples]

        if not valid_windows:
            print("No relevant window found in this audio")
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

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, speech_window)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(2, 1, 2)
        im = plt.imshow(10 * np.log10(Sxx), aspect='auto', origin='lower', 
                        extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        cbar = plt.colorbar(im, orientation='horizontal')
        cbar.set_label('dB')
        plt.tight_layout()
        plt.show()

    def process(self,speech_index,rir_index):
        # Load and preprocess audio files
        speech_waveform, speech_sample_rate = self.load_and_preprocess_audio(self.librispeech_data_dir, ".flac",speech_index)
        # rir_waveform, rir_sample_rate = self.load_and_preprocess_audio(self.rir_data_dir, ".wav")

        truncation_process = Truncated_RIR_EDC(self.rir_data_dir, ".wav")
        EDC_all,EDC_begin_End,noise,rir_normalized,rir_sample_rate = truncation_process.process(rir_index) #change it back to index


        rir_normalized = torch.from_numpy(rir_normalized)
        EDC_begin_End = torch.from_numpy(EDC_begin_End)
        noise = torch.from_numpy(np.array(noise))

        # Resample RIR if necessary
        rir_waveform = self.resample_waveform(rir_normalized, rir_sample_rate, speech_sample_rate)

        # Bandpass filter the RIR
        rir_waveform = self.bandpass_filter(rir_waveform, 10, 7200, speech_sample_rate)
        # speech_waveform = self.bandpass_filter(speech_waveform, 10, 7200, speech_sample_rate)

        # Create reverberant speech
        reverberant_speech = self.create_reverberant_speech(speech_waveform, rir_waveform)

        # Extract fixed-length segment from the reverberant speech
        speech_window,exist = self.extract_fixed_length_segment(reverberant_speech, speech_sample_rate)

        speech_window = speech_window/(max(abs(speech_window)))

        return speech_window,exist,rir_waveform,EDC_all,EDC_begin_End,noise,speech_sample_rate

if __name__ == "__main__":
    import os
    os.system('clear')


    librispeech_data_dir = "/home/prsh7458/Desktop/scratch4/speech_data/raw_test_speech"
    rir_data_dir = "/home/prsh7458/Desktop/scratch4/speech_data/Ilmenau_test_IR"
    model_dir = "/home/prsh7458/work/R2D/edc-estimation/silero_vadsilero_vad"

    # output directories
    output_dir_speech = "/home/prsh7458/Desktop/scratch4/test_data/reverberant_speech" 
    output_dir_edc = "/home/prsh7458/Desktop/scratch4/test_data/EDC" 
    output_dir_noise = "/home/prsh7458/Desktop/scratch4/test_data/noise" 



    speech_files = [file for file in os.listdir(librispeech_data_dir) if file.endswith(".flac")]
    RIR_files = [file for file in os.listdir(rir_data_dir) if file.endswith(".wav")]

    print("Total number of speech files are: ", len(speech_files), "and the RIR files are: ", len(RIR_files))


    reverb_speech_processor = ReverbSpeech(librispeech_data_dir, rir_data_dir, model_dir, view_EDC=False)


    num_speech_files =  len(speech_files)
    num_rir_files = len(RIR_files)


    for rir_index in range(num_rir_files):
        for speech_index in range(num_speech_files):
            # Process the files
            reverberant_speech, exist, rir_waveform, EDC_all, EDC_begin_End, noise, speech_sample_rate  = reverb_speech_processor.process(speech_index, rir_index)  # Adjust this call as per your process function

            # Calculate the unique index for naming
            unique_index = (rir_index) * num_speech_files + (speech_index)

            # Define the file paths using the unique index
            output_file_path_rs = os.path.join(output_dir_speech, f"reverberant_speech_{unique_index}.wav")
            output_file_path_edc_true = os.path.join(output_dir_edc, f"EDC_begin_end_{unique_index}.mat")
            output_file_path_noise = os.path.join(output_dir_noise, f"noise_{unique_index}.mat")

            # Save the reverberant speech to a WAV file
            torchaudio.save(output_file_path_rs, torch.from_numpy(reverberant_speech).float().unsqueeze(0), speech_sample_rate)
            scipy.io.savemat(output_file_path_edc_true, {'EDC_begin_end': np.array(EDC_all)}, do_compression=True)
            scipy.io.savemat(output_file_path_noise, {'noise': np.array(noise)}, do_compression=True)

            # Print messages using the unique index
            print(f"Saved EDC data for unique index {unique_index} to {output_file_path_edc_true}")
            print(f"Saved reverberant speech for unique index {unique_index} to {output_file_path_rs}")
            print(f"Saved noise for unique index {unique_index} to {output_file_path_noise}")