import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from reverbspeech_preprocess import ReverbSpeech
from torch.utils.data import DataLoader
import h5py
import os
import scipy.signal
from log_spectral_feature_extractor import Log_spectral_feature_extraction

"""This scripts handles the data loading of whole ML pipeline"""

class R2DDataset(Dataset):
    """Decay dataset."""

    def __init__(self,train_flag=True,analysis=False,room_idx=0,loc_idx=0,SNR_level = 50,room_idx_noise=0,loc_idx_noise=1, position_bool=False, validation = False, noise_test=False):
        """
        Args:
        """
        self.train_flag = train_flag
        self.analysis = analysis
        self.position_bool = position_bool
        self.validation = validation
        self.SNR_level = SNR_level
        self.noise_test = noise_test

        if train_flag:

            self.audio_path =     "/home/prsh7458/Desktop/scratch4/matlab_data/reverb_speech_data.mat" #/home/prsh7458/Desktop/scratch4/matlab_data
            self.noise_path =   "/home/prsh7458/Desktop/scratch4/matlab_data/noise_data.mat"
            self.EDC_path =     "/home/prsh7458/Desktop/scratch4/matlab_data/EDC_data.mat"
            # output_dir_speech =  "/home/prsh7458/Desktop/scratch4/speech_data/reverberant_speech_ilm"

        elif validation:


            self.audio_path =   "/home/prsh7458/Desktop/scratch4/validation_dataset/r3vival_location/matlab_data/all_data/reverb_speech_data.mat" #/home/prsh7458/Desktop/scratch4/matlab_data
            self.noise_path =   "/home/prsh7458/Desktop/scratch4/validation_dataset/r3vival_location/matlab_data/all_data/noise_data.mat"
            self.EDC_path =     "/home/prsh7458/Desktop/scratch4/validation_dataset/r3vival_location/matlab_data/all_data/EDC_data.mat"

        elif noise_test:

            # print("!!!!!!!!!!!!!!!!!!!!!!!!!ICH BIN HIER!!!!!!!!!!!!!!!!!!!!")
            base_dir = "/home/prsh7458/Desktop/scratch4/noise_roburstness_test/dataset"
            # base_dir_noise = "/home/prsh7458/Desktop/scratch4/noise_roburstness_test/noisy_dataset"

            rooms = ["HL00W", "HL01W", "HL02WL", "HL02WP", "HL03W", "HL04W", "HL05W", "HL06W", "HL08W"] # do not change this order
            room_locations = ["BC", "FC", "FR", "SiL", "SiR"] # do not change this order

            data_dir = os.path.join(base_dir, rooms[room_idx], room_locations[loc_idx])

            self.audio_path = os.path.join(data_dir, "all_data/reverb_speech_data.mat")
            # self.noisy_audio_path = os.path.join(base_dir_noise,str(self.SNR_level), rooms[room_idx_noise], room_locations[loc_idx_noise],"all_data/reverb_noise_data.mat")
            self.noise_path = os.path.join(data_dir, "all_data/noise_data.mat")
            self.EDC_path = os.path.join(data_dir, "all_data/EDC_data.mat")
            
        else:
            # self.audio_path =     "/home/prsh7458/Desktop/scratch4/matlab_test_data/reverb_speech_data.mat"                             
            # self.noise_path =   "/home/prsh7458/Desktop/scratch4/matlab_test_data/noise_data.mat"                                 
            # self.EDC_path =     "/home/prsh7458/Desktop/scratch4/matlab_test_data/EDC_data.mat"   
            # output_dir_speech =  "/home/prsh7458/Desktop/scratch4/test_data/reverberant_speech"  
                        # "/home/prsh7458/desktop/scratch4/speech_data/reverberant_speech"                                   
            if not analysis:
                #/home/prsh7458/Desktop/scratch4/speech_data/loc/dataset/consolidate_data #/home/prsh7458/Desktop/scratch4/matlab_test_data
                self.audio_path =     "/home/prsh7458/Desktop/scratch4/speech_data/loc/dataset/consolidate_data/reverb_speech_data.mat"                             
                self.noise_path =   "/home/prsh7458/Desktop/scratch4/speech_data/loc/dataset/consolidate_data/noise_data.mat"                                 
                self.EDC_path =     "/home/prsh7458/Desktop/scratch4/speech_data/loc/dataset/consolidate_data/EDC_data.mat"  
                # output_dir_speech =  "/home/prsh7458/Desktop/scratch4/test_data/reverberant_speech"  

            else:

                # for individual testing
                # base_dir = "/home/prsh7458/Desktop/scratch4/speech_data/loc/dataset"
                if self.position_bool:
                    base_dir = "/home/prsh7458/Desktop/scratch4/speech_data/loc/dataset_position"
                else:
                    base_dir = "/home/prsh7458/Desktop/scratch4/speech_data/loc/dataset"

                rooms = ["HL00W", "HL01W", "HL02WL", "HL02WP", "HL03W", "HL04W", "HL05W", "HL06W", "HL08W"] # do not change this order
                room_locations = ["BC", "FC", "FR", "SiL", "SiR"] # do not change this order

                data_dir = os.path.join(base_dir, rooms[room_idx], room_locations[loc_idx])

                self.audio_path = os.path.join(data_dir, "all_data/reverb_speech_data.mat")
                self.noise_path = os.path.join(data_dir, "all_data/noise_data.mat")
                if self.position_bool:
                    self.position_path = os.path.join(data_dir, "all_data/position_data.mat")

                self.EDC_path = os.path.join(data_dir, "all_data/EDC_data.mat")
                
                # output_dir_speech = os.path.join(data_dir, "reverberant_speech_ilm")


        # as of now it gets the number from the saved file directory

        # self.speech_files = [file for file in os.listdir(output_dir_speech) if file.endswith(".wav")]
        

        # Reading data
        audio_data = self.read_mat_file(self.audio_path, 'allAudioData')
        noise_data = self.read_mat_file(self.noise_path, 'allNoiseData')
        EDC_data = self.read_cell_array(self.EDC_path, 'allEDCData')

        if (analysis and self.position_bool):
            position_data = self.read_mat_file(self.position_path, 'allPositionData')
            # print("Shape of position data:", position_data.shape)
            self.position_data = torch.from_numpy(position_data).float()
        
        # if noise_test: 
            # self.noisy_audio_data = self.read_mat_file(self.noisy_audio_path, 'allAudioData')
            # self.noisy_audio_data = torch.from_numpy(self.noisy_audio_data).float()



        self.max_number = audio_data.shape[1] 
        # print("Number of files are: ", (audio_data.shape))

        # Convert the list of numpy arrays (EDC data) to a list of tensors

        ###also cropped the direct sound via visual inspection

        
        Direct_sound_sample = 1096 if not self.validation else 500
        self.EDC_true = [torch.from_numpy((edc[Direct_sound_sample:] - edc[Direct_sound_sample]).astype(np.float32)).float() for edc in EDC_data]

        self.noise_data = torch.from_numpy(noise_data).float()
        self.audio_data = torch.from_numpy(audio_data).float()

        self.desired_output = 94375  if not self.validation else 47500    #8000

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


    # def adjust_edc(self,edc, samples_to_crop):

    #     if samples_to_crop >= len(edc):
    #         print("Number of samples to crop is too large.")

    #     # Crop the EDC
    #     cropped_edc = edc[samples_to_crop:]

    #     # Shift the EDC so it starts from zero
    #     shift_value = cropped_edc[0]
    #     adjusted_edc = cropped_edc - shift_value

    #     return adjusted_edc
    def inverse_schroeder_batch(self,y):
        # Convert from dB scale back to linear scale

        y = (y.unsqueeze(1))
        y_lin = 10 ** (y / 10)

        # Flip the EDC (since Schroeder integration involves flipping)
        # Flip along the last dimension
        y_lin_flipped = torch.flip(y_lin, dims=[-1])

        # Approximate the reverse of cumulative integration using differential
        zeros = torch.zeros(y_lin_flipped.shape[0], 1).to(y.device)
        y_lin_flipped_with_zero = torch.cat([zeros, y_lin_flipped], dim=-1)

        # Perform differential along the last dimension
        x_approx = torch.diff(y_lin_flipped_with_zero, dim=-1)

        # Flip back to original orientation
        energy_impulse = torch.flip(x_approx, dims=[-1])

        # Square root to approximate the original squared impulse response
        rms_db = 10 * torch.log10(torch.mean(energy_impulse[int(np.round(0.9 * len(energy_impulse))):]) / torch.max(energy_impulse))

        return rms_db.float()
    

    def __len__(self):
        return self.max_number


    def generate_bandpassed_noise(self, duration, sr):
        # Generate white noise
        from scipy.signal import butter, lfilter
        noise = np.random.normal(0, 1, int(sr * duration))
        
        # Apply a bandpass filter
        b, a = butter(5, [0.01, 0.99], btype='band')
        filtered_noise = lfilter(b, a, noise)
        
        # Calculate scaling factor for the desired dB level
        # dB_level should be a negative number to reduce the noise level
        # scaling_factor = 10**(dB_level / 20)
        
        # Scale the filtered noise
        # scaled_noise = filtered_noise * scaling_factor
        
        return filtered_noise
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.speech_files = f"Reverberant_speech_{idx}"

        Rs = self.audio_data[:, idx]
          # Added to give normalized RIRs as an input but already normalized

        if self.noise_test:
            Rs_rms = torch.sqrt(torch.mean(Rs**2))
            # reverb_noise = self.noisy_audio_data[:, idx]  # if want to use change dry noise with reverb_noise
            dry_noise = self.generate_bandpassed_noise(3, 16000)
            dry_noise = torch.from_numpy(dry_noise).float()
            dry_noise = dry_noise[0:16384]
            # normalized_noise = reverb_noise / torch.max(torch.abs(reverb_noise))
            noise_rms = torch.sqrt(torch.mean(dry_noise**2))  #HERE NORMLIZED NOISE IS NOT USED
            # print(noise_rms,Rs_rms)
            # initial_snr = 10 * torch.log10(Rs_rms**2 / noise_rms**2)
            # print("Initial SNR (dB):", initial_snr)
            desired_noise_rms = Rs_rms / (10 ** (self.SNR_level / 20))
            scaled_noise = dry_noise * (desired_noise_rms / noise_rms)
            Rs = Rs + scaled_noise

            # # Recalculate RMS values after noise addition
            # final_Rs_rms = torch.sqrt(torch.mean(Rs**2))
            # final_noise_rms = torch.sqrt(torch.mean(scaled_noise**2))
            
            # # Calculate final SNR (in dB)
            # final_snr = 10 * torch.log10(final_Rs_rms**2 / final_noise_rms**2)
            # print("Final SNR (dB):", final_snr)
            # reverb_noise = self.noisy_audio_data[:, idx]
            # n = reverb_noise / torch.max(torch.abs(reverb_noise))
            # Rs = Rs + (10**(self.noise_level/10))* n

        Rs = Rs / torch.max(torch.abs(Rs))
        # Apply Hilbert transform using SciPy and convert back to tensor


    #################################   EXTRACT Log Spectral Features    ######################################
        # feature_extractor = Log_spectral_feature_extraction(Rs.unsqueeze(0),samplerate=16000)
        # features = feature_extractor.main_process()
        # final_features = torch.from_numpy(np.squeeze(features)).float()
        # final_features = final_features.unsqueeze(0)  #Pass this as an output instead of Rs when used


    ##################################################################################################################    
        # feature_extractor.plot_features_as_image(final_features)


        # fc = 30 # cutoff frequency of the speech envelope  at 30 Hz
        # N = 5   #filter order
        # w_L = 2*fc/16000
        #lowpas filter tf
        # b, a = scipy.signal.butter(N,w_L, 'low')
        # Rs_numpy = Rs.numpy()  # Convert to numpy array
        # Rs_hilbert_numpy = scipy.signal.filtfilt(b,a,np.abs(scipy.signal.hilbert(Rs_numpy)))




        # # # Rs_hilbert_numpy = scipy.signal.hilbert(Rs_numpy)  # Apply Hilbert transform
        # # # Rs_hilbert = torch.from_numpy(np.abs(Rs_hilbert_numpy)).float()


        # Rs_hilbert = torch.from_numpy(Rs_hilbert_numpy.copy()).float()
        # Rs_hilbert = Rs_hilbert / torch.max(torch.abs(Rs_hilbert))
        

        EDC = self.EDC_true[idx] #Already in db

        # print(EDC.shape)  #94375
        # EDC= self.adjust_edc(EDC,1096)
        # EDC  = EDC[:]
        noise_val = self.noise_data[idx] # Already in db

        # noise_val = self.inverse_schroeder_batch(EDC) 

        if (self.analysis and self.position_bool):
                position_x = self.position_data[0, idx]
                position_y = self.position_data[1, idx]
                position_z = self.position_data[2, idx]
                position = torch.tensor([position_x, position_y, position_z])

        # noise_val = self.noise_data[idx] # Already in db

        schroeder_decays_db = torch.nn.functional.interpolate(EDC.unsqueeze(0).unsqueeze(0), size=self.desired_output,
                                                              mode='linear', align_corners=True)
        


        # schroeder_decays_db = torch.nn.functional.interpolate(EDC.unsqueeze(0).unsqueeze(0), size=self.desired_output,
        #                                                       mode='linear', align_corners=True)
        

        fs_new = int((self.desired_output * 48000) / EDC.shape[0])
        # print(fs_new)
        time_axis_interpolated = torch.linspace(0, (schroeder_decays_db.shape[2] - 1) / fs_new, schroeder_decays_db.shape[2])

        EDC_length = torch.from_numpy(np.array(EDC.shape[0])).float()
        # print("EDC_length",EDC_length) #EDC_length tensor(94375.)

        schroeder_decays_db = schroeder_decays_db.squeeze(0).squeeze(0)

        if self.analysis and self.position_bool:
            return Rs, schroeder_decays_db, time_axis_interpolated, noise_val,EDC_length,self.speech_files,position
        else: 
            return Rs, schroeder_decays_db, time_axis_interpolated, noise_val,EDC_length,self.speech_files 



if __name__ == "__main__":


    import os
    os.system('clear')

    pos_bool =True
    analysis_bool = True
    validation = False
    noise_roburstness_test = False
    SNR_level = 10
    log_spectral = False

    dataset_synthdecays = R2DDataset(train_flag=False,analysis=analysis_bool,
                                     SNR_level=SNR_level, room_idx_noise= 0, loc_idx_noise= 1,
                                     position_bool=pos_bool,
                                     validation=validation,
                                     noise_test=noise_roburstness_test)


    trainloader = DataLoader(dataset_synthdecays, batch_size=2, shuffle=True)

    def inverse_schroeder_batch(y):
        # Convert from dB scale back to linear scale
        y_lin = 10 ** (y / 10)

        # Flip the EDC (since Schroeder integration involves flipping)
        # Flip along the last dimension
        y_lin_flipped = torch.flip(y_lin, dims=[-1])

        # Approximate the reverse of cumulative integration using differential
        zeros = torch.zeros(y_lin_flipped.shape[0], 1).to(y.device)
        y_lin_flipped_with_zero = torch.cat([zeros, y_lin_flipped], dim=-1)

        # Perform differential along the last dimension
        x_approx = torch.diff(y_lin_flipped_with_zero, dim=-1)

        # Flip back to original orientation
        x_approx_flipped = torch.flip(x_approx, dims=[-1])

        # Square root to approximate the original squared impulse response
        x_recovered = torch.sqrt(x_approx_flipped)

        return x_recovered


    def calculate_ere_batch(impulse_responses, direct_sound_end_ms, sampling_rate):
        drr_values = []
        direct_sound_end_sample = int(direct_sound_end_ms / 1000 * sampling_rate)

        for impulse_response in impulse_responses:
            direct_sound = impulse_response[:direct_sound_end_sample]
            reverberant_sound = impulse_response[direct_sound_end_sample:]

            direct_energy = torch.sum(direct_sound ** 2)
            reverberant_energy = torch.sum(reverberant_sound ** 2)

            drr_db = 10 * torch.log10(direct_energy / reverberant_energy)
            drr_values.append(drr_db)

        return torch.tensor(drr_values).to(impulse_responses.device)


    for batch_idx, data_frame in enumerate(trainloader):

        if pos_bool:
            reverberant_speech,schroeder_decays_db,time_axis_interpolated,noise,EDC_length,speech_files,cord = data_frame
            cord=cord.to("cuda")
        else: reverberant_speech,schroeder_decays_db,time_axis_interpolated,noise,EDC_length,speech_files = data_frame
        reverberant_speech,schroeder_decays_db,time_axis_interpolated,noise,EDC_length = reverberant_speech.to("cuda"),schroeder_decays_db.to("cuda").requires_grad_(True),time_axis_interpolated.to("cuda"),noise.to("cuda"),EDC_length.to("cuda")
        
        # EDC_begin_End = data_frame

        print("............................................................................................................................")

        if not log_spectral:
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Adjust the figure size as needed
            # First subplot for schroeder_decays_db
            axs[1].plot(time_axis_interpolated[1,:].cpu().detach().numpy(),schroeder_decays_db[1,:].cpu().detach().numpy())
            axs[1].set_title("Ground Truth: Energy Decay Curve",fontsize=24)
            axs[1].set_xlabel("Time",fontsize=24)
            axs[1].set_ylabel("Amplitude (dB)",fontsize=24)
            axs[1].tick_params(axis='x', labelsize=24) 
            axs[1].tick_params(axis='y', labelsize=24) 
            axs[1].legend(fontsize=24)

        #     # Apply inverse Schroeder and plot in the second subplot
        #     y = inverse_schroeder_batch(schroeder_decays_db)

        #     direct_sound_end_ms = 80
        #     sampling_rate = 8137

        #     # Calculate the boundary in samples (convert time in ms to number of samples)
        #     direct_reverberant_boundary_samples = int(direct_sound_end_ms / 1000 * sampling_rate)
        # # Add vertical line to indicate the direct/reverberant boundary
        #     axs[0].axvline(x=direct_reverberant_boundary_samples, color='r', linestyle='--')
        #     axs[0].legend(['Schroeder Decay', 'Direct/Reverberant Boundary'])

        #     # Calculate DRR for the entire batch
        #     drr = calculate_ere_batch(y, direct_sound_end_ms, sampling_rate) 

        #     # print("drr shape",drr.shape)

        #     axs[1].plot(y[1,:].cpu().detach().numpy())
        #     axs[1].set_title("Inverse Schroeder Transformed Data")
        #     axs[1].set_xlabel("Time")
        #     axs[1].set_ylabel("Amplitude")
        #     axs[1].axvline(x=direct_reverberant_boundary_samples, color='r', linestyle='--')
        #     axs[1].legend(['Schroeder Decay', 'Direct/Reverberant Boundary'])
            # Add a main title to the figure
            # plt.suptitle(f"DRR = {np.round((drr[1].detach().cpu().numpy()),2)}")

            # Adjust the layout so that titles and labels do not overlap
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            time_axis_reverberant_speech= torch.linspace(0, (reverberant_speech.shape[1] - 1) / 16000, reverberant_speech.shape[1]).cpu().detach().numpy()
            axs[0].plot(time_axis_reverberant_speech,reverberant_speech[1,:].cpu().detach().numpy())
            axs[0].set_title("Reverberant speech",fontsize=24)
            axs[0].set_xlabel("Time",fontsize=24)
            axs[0].set_ylabel("Amplitude",fontsize=24)
            axs[0].tick_params(axis='x', labelsize=24) 
            axs[0].tick_params(axis='y', labelsize=24) 
            axs[0].legend(fontsize=24)
            # Show the plot
            plt.show()

        print("schroeder_decays_db shape: ",schroeder_decays_db.shape)
        print("Reverb_speech shape: ",reverberant_speech.shape)
        print("time_axis_interpolated shape: ",time_axis_interpolated.shape)
        print("noise shape: ",noise.shape)
        print("EDC_length shape: ",EDC_length.shape)
        if pos_bool:
            print(cord.shape)
            print(cord)
        print("............................................................................................................................")


