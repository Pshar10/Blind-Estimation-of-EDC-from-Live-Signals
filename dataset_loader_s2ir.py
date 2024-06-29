import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from reverbspeech_preprocess import ReverbSpeech
from torch.utils.data import DataLoader
import h5py
import os

"""A script to load dataset in pytorch for S2IR model only"""


class R2DDataset(Dataset):
    """Decay dataset."""

    def __init__(self,train_flag=True):
        """
        Args:
        """
        self.train_flag = train_flag

        if train_flag:
            self.audio_path =     "/home/prsh7458/Desktop/scratch4/matlab_data/reverb_speech_data.mat" #/home/prsh7458/Desktop/scratch4/matlab_data
            self.noise_path =   "/home/prsh7458/Desktop/scratch4/matlab_data/noise_data.mat"
            self.EDC_path =     "/home/prsh7458/Desktop/scratch4/matlab_data/EDC_data.mat"
            output_dir_speech =  "/home/prsh7458/Desktop/scratch4/speech_data/reverberant_speech_ilm"
        else:

            self.audio_path =     "/home/prsh7458/Desktop/scratch4/matlab_test_data/reverb_speech_data.mat"                             
            self.noise_path =   "/home/prsh7458/Desktop/scratch4/matlab_test_data/noise_data.mat"                                 
            self.EDC_path =     "/home/prsh7458/Desktop/scratch4/matlab_test_data/EDC_data.mat"   
            output_dir_speech =  "/home/prsh7458/Desktop/scratch4/test_data/reverberant_speech"                               # "/home/prsh7458/work/speech_data/reverberant_speech"                                   


        # as of now it gets the number from the saved file directory

        speech_files = [file for file in os.listdir(output_dir_speech) if file.endswith(".wav")]
        self.max_number = len(speech_files)

        # Reading data
        audio_data = self.read_mat_file(self.audio_path, 'allAudioData')
        noise_data = self.read_mat_file(self.noise_path, 'allNoiseData')
        EDC_data = self.read_cell_array(self.EDC_path, 'allEDCData')

        # Convert the list of numpy arrays (EDC data) to a list of tensors
        self.EDC_true = [torch.from_numpy(edc.astype(np.float32)).float() for edc in EDC_data]

        self.noise_data = torch.from_numpy(noise_data).float()
        self.audio_data = torch.from_numpy(audio_data).float()

        self.desired_output = 16384

    def read_mat_file(self, file_path, variable_name):
        with h5py.File(file_path, 'r') as file:
            data = np.array(file[variable_name])
        return data

    def read_cell_array(self, file_path, variable_name):
        with h5py.File(file_path, 'r') as file:
            cell_array_refs = file[variable_name][()]
            cell_array_data = []
            for ref in cell_array_refs:
                dereferenced_data = file[ref[0]]
                cell_array_data.append(np.array(dereferenced_data).flatten())
        return cell_array_data

    def __len__(self):
        return self.max_number

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        Rs = self.audio_data[:, idx]
        Rs = Rs / torch.max(torch.abs(Rs))  # Added to give normalized RIRs as an input but already normalized
        EDC = self.EDC_true[idx] #Already in db
        noise_val = self.noise_data[idx] # Already in db

        schroeder_decays_db = torch.nn.functional.interpolate(EDC.unsqueeze(0).unsqueeze(0), size=self.desired_output,
                                                              mode='linear', align_corners=True)

        fs_new = int((self.desired_output * 16000) / EDC.shape[0])
        time_axis_interpolated = torch.linspace(0, (schroeder_decays_db.shape[2] - 1) / fs_new, schroeder_decays_db.shape[2])
        EDC_length = torch.from_numpy(np.array(EDC.shape[0])).float()

        return Rs, schroeder_decays_db.squeeze(0).squeeze(0), time_axis_interpolated, noise_val,EDC_length

if __name__ == "__main__":


    import os
    os.system('clear')

    dataset_synthdecays = R2DDataset()


    trainloader = DataLoader(dataset_synthdecays, batch_size=2, shuffle=True)

    for batch_idx, data_frame in enumerate(trainloader):

        reverberant_speech,schroeder_decays_db,time_axis_interpolated,noise,EDC_length = data_frame
        reverberant_speech,schroeder_decays_db,time_axis_interpolated,noise,EDC_length = reverberant_speech.to("cuda"),schroeder_decays_db.to("cuda").requires_grad_(True),time_axis_interpolated.to("cuda"),noise.to("cuda"),EDC_length.to("cuda")
        # EDC_begin_End = data_frame

        print("............................................................................................................................")
        print(batch_idx)
        print(reverberant_speech.shape)
        print(schroeder_decays_db.shape[1])
        print(time_axis_interpolated.shape)
        print(noise.shape)
        print(EDC_length.shape)
        print("............................................................................................................................")


