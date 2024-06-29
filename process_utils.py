import torch
import torch.nn as nn
import torchaudio.functional
from torch.utils.data import Dataset
import scipy
import scipy.stats
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
from lundeby import lundeby


def edc_loss(t_vals_prediction, a_vals_prediction, n_exp_prediction, edcs_true, device, time_axis_interpolated, s_prediction, edcs, speech_files, room=None, location=None, mse_flag =False,
             plot_analysis=False, apply_mean=True,figure=False,plot_save=False,plt_show=False):
    # fs = 10
    # l_edc = 10

    #  #print("Inside loss - t: {}, a: {}, n: {}".format(t_vals_prediction.requires_grad, a_vals_prediction.requires_grad, n_exp_prediction.requires_grad))

    # Generate the t values that would be discarded (last 5%) as well, otherwise the models do not match.
    # t = (torch.linspace(0, l_edc * fs - 1, round((1 / 0.95) * l_edc * fs)) / fs).to(device)

    # Clamp noise to reasonable values to avoid numerical problems and go from exponent to actual noise value
    # n_exp_prediction = torch.clamp(n_exp_prediction, -32, 32)
    n_vals_prediction = torch.pow(10, n_exp_prediction/10)
    # print("Inside loss - n_exp_prediction: {}, n_vals_prediction: {}".format(n_exp_prediction.requires_grad, n_vals_prediction.requires_grad))

    loss_l1 = nn.L1Loss(reduction='none')
    loss_mse = nn.MSELoss(reduction='none')

    if not mse_flag:
        # loss_fn = nn.L1Loss(reduction='none')
        # loss_fn = nn.SmoothL1Loss(reduction='none')
        loss_fn = nn.L1Loss(reduction='none')
    else:
        loss_fn = nn.MSELoss(reduction='none')

    # Use predicted values to generate an EDC

    # fs = 16000 #sampling frequency

    # batch_size = s_prediction.shape[0]  # Assuming s_prediction has shape [batch_size]
    # time_axes_pred = []

    # for i in range(batch_size):
    #     fs_new = (fs * (edcs_true.shape[1] / s_prediction[i])).item()
    #     time_axis = torch.linspace(0, (edcs_true.shape[1] - 1) / fs_new, edcs_true.shape[1])
    #     time_axes_pred.append(time_axis)

    # # Convert the list of tensors into a single tensor
    # time_axes_pred = torch.stack(time_axes_pred, dim=0).to("cuda")

    edc_prediction,noise,edc_pred_wo_noise = generate_synthetic_edc_torch(t_vals_prediction, a_vals_prediction, n_vals_prediction, time_axis_interpolated, device)
    edc_prediction_db = 10 * torch.log10(edc_prediction + 1e-16)                    #ADDED NEW ____NEDDS TO BE CHECKED


    if apply_mean:
        loss = torch.mean(loss_fn(edcs_true, edc_prediction_db))
    else:
        loss = loss_fn(edcs_true, edc_prediction_db)



    # up_to_miliseconds = 50

    # y_pred = inverse_schroeder_batch(edc_prediction_db)
    # ere_pred = calculate_ere_batch(y_pred, up_to_miliseconds, 16000) 


    # y_true = inverse_schroeder_batch(edcs_true)
    # ere_true = calculate_ere_batch(y_true, up_to_miliseconds, 16000) 

    # loss_ere = torch.mean(loss_l1(ere_true, ere_pred)) ## added new to be checked
        
    

    # Define the dB range
    db_min = -15
    db_max = 0

    # Select samples within the dB range
    mask = (edcs_true >= db_min) & (edcs_true <= db_max)
    selected_edc_true = edcs_true[mask]
    selected_edc_pred = edc_prediction_db[mask]


    # loss_EDT = torch.mean(loss_mse(selected_edc_true, selected_edc_pred))
    loss_EDT = torch.mean(loss_l1(selected_edc_true, selected_edc_pred))

    # print("loss_EDT:" , loss_EDT)

##################  CAN BE ADDED WHILE TRAINING ###################################################################
    
    # d20_mean_true = EDC2DT_batch(edcs_true)
    # d20_mean_pred = EDC2DT_batch(edc_prediction_db)
    # loss_DT = torch.mean(loss_l1(d20_mean_true, d20_mean_pred))


    # print(d20_mean_true,d20_mean_pred)
        


    if plot_analysis:

        fitted_lines_true,T_true , floor_true ,DT20m_true= EDC2T60(edcs_true)
        fitted_lines_pred,T_pred , floor_pred,DT20m_pred = EDC2T60(edc_prediction_db)
        T_true = torch.tensor([T_true], device='cuda:0') 
        T_pred = torch.tensor([T_pred], device='cuda:0')

        mask_lines_true = (fitted_lines_true >= -60) 
        mask_lines_pred = (fitted_lines_pred >= -60) 
        selected_EDT_true = fitted_lines_true[mask_lines_true]
        selected_EDT_pred = fitted_lines_pred[mask_lines_pred]
    

        time = time_axis_interpolated[0,:]
        time_true = time[mask_lines_true]
        time_pred = time[mask_lines_pred]

        # index_true = selected_EDT_true.size(0) - 1
        # index_pred = selected_EDT_pred.size(0) - 1

        # Calculate the time at which they reach -60 dB, considering a sampling rate of 16000 Hz
        # Indexing starts at 0, so we add 1 to get the correct sample number
        # T60_true_float = (index_true + 1) / 2712  #2712 is new sampling freq as per initial adjustments
        # T60_pred_float = (index_pred + 1) / 2712
        # T60_true = torch.tensor([T60_true_float], device='cuda:0') 
        # T60_pred = torch.tensor([T60_pred_float], device='cuda:0')
        pad_size = max(selected_EDT_true.size(0), selected_EDT_pred.size(0)) #2712
        regression_EDT_true = torch.nn.functional.pad(selected_EDT_true, (0, pad_size - selected_EDT_true.size(0)))
        regression_EDT_pred = torch.nn.functional.pad(selected_EDT_pred, (0, pad_size - selected_EDT_pred.size(0)))


        # T60_loss = loss_mse(T_true, T_pred)
        T60_loss = loss_mse(T_true, t_vals_prediction.squeeze(0))  # the one with multiexponential decay RT
        DT_loss = loss_mse(DT20m_true, DT20m_pred)  # the one with multiexponential decay RT

        # Regression_loss = torch.mean(loss_l1(regression_EDT_true, regression_EDT_pred))
        # print("Regression_loss L1 loss:    " ,Regression_loss)
        # print("T60 orig : " ,T_true,"T60 pred : " ,T_pred, "T60_loss: ",(loss_l1(T_true, T_pred)))


        if figure:
            dir_path = os.path.join('/home/prsh7458/Desktop/scratch4/R2D/curves/curves_loc', f'room_{room}_loc_{location}')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for idx in range(edc_prediction.shape[0]):
                # Calculate the loss for the current prediction
                if (abs(T_true-t_vals_prediction.squeeze(0))/T_true)*100<5 and loss<1.5:
                # if True:

                    fig, ax = plt.subplots()  # Create a new figure and axes

                    ax.plot(time_axis_interpolated[idx,:].cpu().numpy(), 10 * np.log10(edc_prediction[idx, :].cpu().numpy()), label='Predicted Curve')
                    # ax.plot(time_axis_interpolated[idx,:].cpu().numpy(), 10 * np.log10(edc_pred_wo_noise[idx, :].cpu().numpy()), label='Predicted Curve without noise')
                    # ax.plot(time_axis_interpolated[idx,:].cpu().numpy(), 10 * np.log10(noise[idx, :].cpu().numpy()), label='Predicted noise')
                    ax.plot(time_axis_interpolated[idx,:].cpu().numpy(), edcs[idx,:].cpu().numpy(), label='Original Curve')
                    # ax.plot(time_axis_interpolated[idx,:].cpu().numpy(), edcs[idx,:].cpu().numpy(), label="Schroeder's EDC", linewidth = 3)


                    # ax.plot(time_true.cpu().numpy(), selected_EDT_true.cpu().numpy(), label='Regression', linestyle='--', linewidth = 3) #uncommet this
                    ax.plot(time_true.cpu().numpy(), selected_EDT_true.cpu().numpy(), label='EDT True', linestyle='--') #uncommet this
                    ax.plot(time_pred.cpu().numpy(), selected_EDT_pred.cpu().numpy(), label='EDT Pred', linestyle='--') #uncomment this


                    # ax.plot(time_axis_interpolated[idx,:].cpu().numpy(), 10 * np.log10(noise[idx, :].cpu().numpy()))
                    # ax.plot(time_axis_interpolated[idx,:].cpu().numpy(), 10 * np.log10(noise[idx,:].cpu().numpy()))

                    ax.set_ylim([-62, 5])
                    ax.set_xlim([0, round(len(edcs[idx,:].cpu().numpy())/48000)])
                    ax.tick_params(axis='x', labelsize=14)  
                    ax.tick_params(axis='y', labelsize=14)  

                    plt.xlabel('Time', fontsize=14)
                    plt.ylabel('Amplitude (dB)', fontsize=14)
                    ax.legend(fontsize=14)
                    plt.title(f"{speech_files[0]} Room: {room} Loc: {location} ", fontsize=14)
                    # plt.title("Schroeder's EDC with Regression till -60 dB", fontsize=28)

                    save_filename = os.path.join(dir_path, f"{speech_files[0]}.png")

                    if plot_save:
                        fig.savefig(save_filename)
                        plt.close(fig)
                    if plt_show:
                        T60_mae_loss = loss_l1(T_true, T_pred) 
                        print("DT20m_true: ",round(DT20m_true.item(),2),"DT20m_pred: ",round(DT20m_pred.item(),2))
                        print("T_true: ",round(T_true.item(),2),"T_pred: ",round(T_pred.item(),2),"T_loss: ",round(T60_mae_loss.item(),2))
                        print("floor_true: ",round(floor_true.item(),2),"floor_pred: ",round(floor_pred.item(),2))
                        plt.show()
                    


        return loss , T60_loss, T_true, T_pred , DT20m_true, DT20m_pred, DT_loss      #,loss_EDT to be changed

    return loss , loss_EDT, 0, 0 ,0 ,0, 0




def calcDTfromSC(sc, fs, mdB_start, mdB_end):
    # Normalize Schroeder curve so that it starts from 0 dB
    # if sc[0] != 0:
    #     sc = sc - sc[0]

    # Find the indices in the Schroeder curve corresponding to the start and end dB levels
    mRTdB_start = (sc <= mdB_start).nonzero(as_tuple=True)[0][0]
    mRTdB_end = (sc <= mdB_end).nonzero(as_tuple=True)[0][0]

    # If there's no point below the end dB level, use the minimum point as the end
    if mRTdB_end.numel() == 0:
        mRTdB_end = sc.argmin().item()
        mdB_end = sc[mRTdB_end].item()
    
    # Time points for the linear fit
    time = torch.linspace(start=(mRTdB_start/fs).item(), end=(mRTdB_end/fs).item(), steps=mRTdB_end-mRTdB_start+1)
    
    # Prepare the design matrix for linear regression: y = ax + b
    # First column is time, second column is ones for intercept
    X = torch.vstack((time, torch.ones_like(time))).T.to(sc.device)
    y = sc[mRTdB_start:mRTdB_end + 1].unsqueeze(1)

    # Solve the least squares problem
    beta = torch.linalg.lstsq(X, y).solution
    slope, intercept = beta.squeeze().to('cuda')  # Ensure slope and intercept are on CUDA

    # Calculate the decay time (DT) using the slope of the line
    DT = ((-60 - intercept) / slope).item()

    return DT,intercept

def EDC2T60(edc, fs=48000,plot= False): # 94375 samples and fs is  48000,  2034 for 48000fs and 4000 samples, 4068 for 48000fs and 8000 samples ,8137 for 48000 fs and 16000 samples  and 2712 for 16000fs and 16000 samples
    # Convert edc to a PyTorch tensor if it's not already
    if not isinstance(edc, torch.Tensor):
        edc = torch.tensor(edc, dtype=torch.float32)
    edc = torch.squeeze(edc).to('cuda')  # Ensure it's on CUDA
    if plot:
        # Assuming rir_approx is your data to plot
        edc_curve = edc.cpu().numpy()
        time_axis_edc_curve = np.linspace(0, len(edc_curve) / 48000, len(edc_curve))
        plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
        plt.plot(time_axis_edc_curve, edc_curve)
        plt.title('EDC', fontsize=16)
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('Amplitude(dB)', fontsize=16)
        plt.show()

    # last_index = int(0.5 * edc.size(0))
    # first_index = int(0.25 * edc.size(0))
    # noise_floor_level = torch.mean(edc[first_index:last_index]).to('cuda')  # Calculated and moved to CUDA
    safety_threshold = 3
    rir_approx = inverse_schroeder(edc)

    if plot:
        # Assuming rir_approx is your data to plot
        rir_data = rir_approx.cpu().numpy()
        time_axis = np.linspace(0, len(rir_data) / 48000, len(rir_data))
        plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
        plt.plot(time_axis, rir_data)
        plt.title('Approximate RIR', fontsize=16)
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.show()

    ponto, __, __, __ = lundeby(rir_approx.cpu().numpy(),fs,plot)

    if plot:
        plot_edc_with_crossing_point(edc,ponto) ##here we can plot


    noise_floor_level = edc[ponto]
    dB_end = noise_floor_level + safety_threshold
    # dB_end = -15


    # Decay time (DT) calculation
    dB_start = 0
    dB_step = 1
    dB_wide = 10
    mdBw = torch.arange(dB_start, dB_end + dB_wide, -dB_step)


    DT20 = []
    DT20_intercept = []
    for idx, mdB in enumerate(mdBw):
        dt,intecept = calcDTfromSC(edc, fs, mdB.item(), (mdB - dB_wide).item())
        DT20.append(dt)
        DT20_intercept.append(intecept)
    
    # Mean and slope of DT20
    DT20_tensor = torch.tensor(DT20).to('cuda')  # Ensure conversion to tensor and move to CUDA if necessary
    DT20_intercept_tensor = torch.tensor(DT20_intercept).to('cuda')  # Ensure conversion to tensor and move to CUDA if necessary

    DT20m = torch.mean(DT20_tensor)
    DT20_intercept_tensorm = torch.mean(DT20_intercept_tensor)

    # DT20s = (DT20[1] - DT20[0]) / (mdBw[1] - mdBw[0])



    # Time axis
    t = torch.arange(edc.size(0), dtype=torch.float32) / fs
    t = t.to('cuda')  # Move to CUDA

    mask = (edc >= -15) & (edc <= dB_start)

    # Filtered time and EDC values
    x = t[mask]
    y = edc[mask]

    # Perform linear regression using PyTorch
    X = torch.vstack((x, torch.ones_like(x))).T.to('cuda')  # Ensure X is on CUDA
    Y = y[:, None].to('cuda')  # Ensure Y is on CUDA

    # Calculate (X^T X)^(-1) X^T Y
    beta = torch.linalg.lstsq(X, Y).solution
    slope, intercept = beta.squeeze().to('cuda')  # Ensure slope and intercept are on CUDA
    slope = slope if slope != 0 else 0.01  # Avoid division by zero




    fitted_y = slope * t + intercept
    T = (-60 - intercept) / slope


    DT_slope = (-60-intercept)/DT20m
    fitted_y_DT = DT_slope * t + intercept

    return fitted_y, T, noise_floor_level, DT20m



def EDC2DT_batch(edc, fs=48000):
    if not isinstance(edc, torch.Tensor):
        edc = torch.tensor(edc, dtype=torch.float32)
    edc = edc.to('cuda')  # Convert the entire batch to CUDA at once

    batch_size, _ = edc.shape
    DT20_means = []

    for i in range(batch_size):
        single_edc = torch.squeeze(edc[i])

        # Noise floor calculation
        last_index = int(0.5 * single_edc.size(0))
        first_index = int(0.25 * single_edc.size(0))
        safety_threshold = 3

        rir_approx = inverse_schroeder(single_edc)
        ponto, __, __, __ = lundeby(rir_approx.cpu().numpy(),fs,1)
        noise_floor_level = single_edc[ponto] + safety_threshold

        # safety_threshold = 3
        # dB_end = noise_floor_level + safety_threshold
        dB_end = noise_floor_level

        # Decay time calculation
        dB_start = 0
        dB_step = 1
        dB_wide = 10
        mdBw = torch.arange(dB_start, dB_end + dB_wide, -dB_step).to('cuda')

        DT20 = []
        for mdB in mdBw:
            # This function should calculate and return the DT for a single curve at a specific dB level
            dt, _ = calcDTfromSC(single_edc, fs, mdB.item(), (mdB - dB_wide).item())
            DT20.append(dt)
        
        # Mean of DT20 for the current EDC
        DT20_means.append(torch.mean(torch.tensor(DT20)).item())

    # Convert the list of means to a tensor
    DT20_means_tensor = torch.tensor(DT20_means).to('cuda')

    return DT20_means_tensor



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

def inverse_schroeder(y):
    # Convert from dB scale back to linear scale
    y_lin = 10 ** (y / 10)

    # Flip the EDC (since Schroeder integration involves flipping)
    y_lin_flipped = torch.flip(y_lin, dims=[-1])

    # Approximate the reverse of cumulative integration using differential
    # Add a zero at the beginning of the flipped array
    y_lin_flipped_with_zero = torch.cat([torch.zeros(1).to(y.device), y_lin_flipped], dim=-1)

    # Perform differential to approximate the reverse of cumulative integration
    x_approx = torch.diff(y_lin_flipped_with_zero, dim=-1)

    # Flip back to original orientation
    x_approx_flipped = torch.flip(x_approx, dims=[-1])

    # Square root to approximate the original squared impulse response
    x_recovered = torch.sqrt(x_approx_flipped)

    return x_recovered


def calculate_ere_batch(impulse_responses, direct_sound_end_ms, sampling_rate):
    ere_values = []
    direct_sound_end_sample = int(direct_sound_end_ms / 1000 * sampling_rate)

    for impulse_response in impulse_responses:
        direct_sound = impulse_response[:direct_sound_end_sample]
        reverberant_sound = impulse_response[direct_sound_end_sample:]

        direct_energy = torch.sum(direct_sound ** 2)
        reverberant_energy = torch.sum(reverberant_sound ** 2)

        # ere_db = 10 * torch.log10(direct_energy / reverberant_energy)
        ere_db = 10 * torch.log10(direct_energy)
        ere_values.append(ere_db)

    return torch.tensor(ere_values).to(impulse_responses.device)
    


def plot_edc_with_crossing_point(edc, ponto):
    """
    Plot the Energy Decay Curve (EDC) and indicate the crossing point where it meets the noise floor.

    Parameters:
    - edc (numpy.ndarray): The energy decay curve as a numpy array.
    - ponto (int): The index in the EDC array where it crosses the noise floor.
    - noise_floor (float): The estimated noise floor level.
    """
    # Extract the value at 'ponto' for plotting
    edc = edc.cpu().numpy()
    point_value = edc[ponto]

    # Create the plot
    plt.figure(figsize=(10, 6))
    time_axis = np.linspace(0, len(edc) / 48000, len(edc))
    plt.plot(time_axis,edc, label='EDC')  # Plot the EDC curve

    # Highlight the point of crossing with an arrow
    plt.annotate('Noise floor crossing', xy=(ponto, point_value), xytext=(ponto + 10, point_value + 0.2 * point_value),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 )

    # Optionally, highlight the noise floor level as well
    noise_floor = point_value
    plt.axhline(y=noise_floor, color='r', linestyle='--', label='Noise Floor')

    # Adding labels and title for clarity
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.title('Energy Decay Curve (EDC) and Noise Floor Crossing Point')
    plt.legend()

    # Show the plot
    plt.show()


def discard_last_n_percent(edc: torch.Tensor, n_percent: float) -> torch.Tensor:
    # Discard last n%
    last_id = int(np.round((1 - n_percent / 100) * edc.shape[-1]))
    out = edc[..., 0:last_id]

    return out


def _discard_below(edc: torch.Tensor, threshold_val: float) -> torch.Tensor:
    # set all values below minimum to 0
    out = edc.detach().clone()
    out[out < threshold_val] = 0

    out = _discard_trailing_zeros(out)
    return out


def _discard_trailing_zeros(rir: torch.Tensor) -> torch.Tensor:
    # find first non-zero element from back
    last_above_thres = rir.shape[-1] - torch.argmax((rir.flip(-1) != 0).squeeze().int())

    # discard from that sample onwards
    out = rir[..., :last_above_thres]
    return out


def check_format(rir):
    rir = torch.as_tensor(rir).detach().clone()

    if len(rir.shape) == 1:
        rir = rir.reshape(1, -1)

    if rir.shape[0] > rir.shape[1]:
        rir = torch.swapaxes(rir, 0, 1)
         #print(f'Swapped axes to bring rir into the format [{rir.shape[0]} x {rir.shape[1]}]. This should coincide '
            #   f'with [n_channels x rir_length], which is the expected input format to the function you called.')
    return rir


def rir_onset(rir):
    spectrogram_trans = torchaudio.transforms.Spectrogram(n_fft=64, win_length=64, hop_length=4)
    spectrogram = spectrogram_trans(rir)
    windowed_energy = torch.sum(spectrogram, dim=len(spectrogram.shape)-2)
    delta_energy = windowed_energy[..., 1:] / (windowed_energy[..., 0:-1]+1e-16)
    highest_energy_change_window_idx = torch.argmax(delta_energy)
    onset = int((highest_energy_change_window_idx-2) * 4 + 64)
    return onset



def _postprocess_parameters(t_vals, a_vals, n_vals, scale_adjust_factors, exactly_n_slopes_mode):
    # Process the estimated t, a, and n parameters

    # Adjust for downsampling
    n_vals = n_vals / scale_adjust_factors['n_adjust']

    # Only for DecayFitNet: T value predictions have to be adjusted for the time-scale conversion (downsampling)
    t_vals = t_vals / scale_adjust_factors['t_adjust']  # factors are 1 for Bayesian

    # In nSlope estimation mode: get a binary mask to only use the number of slopes that were predicted, zero others
    if not exactly_n_slopes_mode:
        mask = (a_vals == 0)

        # Assign NaN instead of zero for now, to sort inactive slopes to the end
        t_vals[mask] = np.nan
        a_vals[mask] = np.nan

    # Sort T and A values
    sort_idxs = np.argsort(t_vals, 1)
    for band_idx in range(t_vals.shape[0]):
        t_vals[band_idx, :] = t_vals[band_idx, sort_idxs[band_idx, :]]
        a_vals[band_idx, :] = a_vals[band_idx, sort_idxs[band_idx, :]]

    # In nSlope estimation mode: set nans to zero again
    if not exactly_n_slopes_mode:
        t_vals[np.isnan(t_vals)] = 0
        a_vals[np.isnan(a_vals)] = 0

    return t_vals, a_vals, n_vals


def decay_model(t_vals, a_vals, n_val, time_axis, compensate_uli=True, backend='np', device='cpu'):
    # t_vals, a_vals, n_vals can be either given as [n_vals, ] or as [n_batch or n_bands, n_vals]

    # Avoid div by zero for T=0: Write arbitary number (1) into T values that are equal to zero (inactive slope),
    # because their amplitude will be 0 as well (i.e. they don't contribute to the EDC)
    zero_t = (t_vals == 0)
    also_zero_a = (a_vals[zero_t] == 0)
    if backend == 'torch':
        also_zero_a = also_zero_a.numpy()
    assert (np.all(also_zero_a)), "T values equal zero detected, for which A values are nonzero. This " \
                                  "yields division by zero. For inactive slopes, set A to zero."
    t_vals[t_vals == 0] = 1

    if backend == 'np':
        edc_model = generate_synthetic_edc_np(t_vals, a_vals, n_val, time_axis, compensate_uli=compensate_uli)
        return edc_model
    elif backend == 'torch':
        edc_model = generate_synthetic_edc_torch(t_vals, a_vals, n_val, time_axis, device=device,
                                                 compensate_uli=compensate_uli)

        # Output should have the shape [n_bands, n_batches, n_samples]
        edc_model = torch.unsqueeze(edc_model, 1)
        return edc_model
    else:
        raise ValueError("Backend must be either 'np' or 'torch'.")


def generate_synthetic_edc_torch(t_vals, a_vals, noise_level, time_axis, device='cuda', compensate_uli=True) -> torch.Tensor:
    """Generates an EDC from the estimated parameters."""
    # Ensure t_vals, a_vals, and noise_level are properly shaped
    zero_t = (t_vals == 0)
    t_vals[zero_t] = 1  # Avoid division by zero

    # Calculate decay rates based on the requirement that after T60 seconds, the level must drop to -60dB
    tau_vals = torch.log(torch.tensor([1e6], device=device)) / t_vals

    # Calculate exponentials from decay rates
    time_vals = -time_axis * tau_vals  # batchsize x 16000 which is the length of tim axis
    
    exponentials = torch.exp(time_vals)
     #print("Inside generate_synthetic_edc_torch - exponentials: {}, time_vals: {}".format(exponentials.requires_grad, time_vals.requires_grad))


    # Account for limited upper limit of integration, if needed

    exp_offset = 0

    # Multiply exponentials with their amplitudes (a_vals)
    edcs = a_vals * (exponentials - exp_offset)
    # edcs = (exponentials - exp_offset)
    edc = torch.sum(edcs, 1)                                            #ADDED NEW ____NEDDS TO BE CHECKED

    # Add noise (scaled linearly over time)
    noise = noise_level * torch.linspace(time_axis.shape[1], 1, time_axis.shape[1], device=device)
    # print("noise shape",noise.shape)
    # print((noise.cpu().numpy()))
    # print(noise.shape)
    # plt.plot(10*np.log10(noise[1,:].cpu().numpy()))
    # plt.show()
    edc = edcs + noise

    return edc,noise,edcs




def generate_synthetic_edc_np(t_vals, a_vals, noise_level, time_axis, compensate_uli=True) -> np.ndarray:
    value_dim = len(t_vals.shape) - 1

    # get decay rate: decay energy should have decreased by 60 db after T seconds
    zero_a = (a_vals == 0)
    tau_vals = np.log(1e6) / t_vals
    tau_vals[zero_a] = 0

    # calculate decaying exponential terms
    time_vals = - np.tile(time_axis, (*t_vals.shape, 1)) * np.expand_dims(tau_vals, -1)
    exponentials = np.exp(time_vals)

    # account for limited upper limit of integration, see: Xiang, N., Goggans, P. M., Jasa, T. & Kleiner, M.
    # "Evaluation of decay times in coupled spaces: Reliability analysis of Bayeisan decay time estimation."
    # J Acoust Soc Am 117, 3707–3715 (2005).
    if compensate_uli:
        exp_offset = np.expand_dims(exponentials[..., -1], -1)
    else:
        exp_offset = 0

    # calculate final exponential terms
    exponentials = (exponentials - exp_offset) * np.expand_dims(a_vals, -1)

    # zero exponentials where T=A=0 (they are NaN now because div by 0, and NaN*0=NaN in python)
    exponentials[zero_a, :] = 0

    # calculate noise term
    noise = noise_level * np.linspace(len(time_axis), 1, len(time_axis))
    noise = np.expand_dims(noise, value_dim)

    edc_model = np.concatenate((exponentials, noise), value_dim)
    return edc_model
