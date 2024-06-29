import torch
from matplotlib import pyplot as plt
import numpy as np

"""A script to demonstrate how synthetic graphs are plotted and line regression is done"""

def generate_synthetic_edc_torch(t_vals, a_vals, noise_level, time_axis, device='cpu', compensate_uli=True,w_noise=False) -> torch.Tensor:
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

    noise = noise_level * torch.linspace(len(time_axis), 1, len(time_axis)).to(device)
    # print(noise.shape)
    if w_noise:
        edc = torch.tensor(edcs).clone().detach()+ torch.tensor(noise).clone().detach()
    else:
        edc = torch.tensor(edcs).clone().detach()#+ torch.tensor(noise).clone().detach() 


    # plt.plot(edc[1,:].cpu().detach().numpy())
    # plt.show()

    return edc    



def EDC2T60(edc, fs=16000):
    # Convert edc to a PyTorch tensor if it's not already
    if not isinstance(edc, torch.Tensor):
        edc = torch.tensor(edc, dtype=torch.float32)
    edc = torch.squeeze(edc)

    fitted_lines = []
    for row in edc:
        # Time axis
        t = torch.arange(row.size(0), dtype=torch.float32) / fs

        # Mask for -5 dB to -25 dB
        mask = (row >= -25) & (row <= -5)

        # Filtered time and EDC values
        x = t[mask]
        y = row[mask]

        # Perform linear regression using PyTorch
        # Create X matrix for linear regression (with a column of ones for intercept)
        X = torch.vstack((x, torch.ones_like(x))).T
        Y = y[:, None]

        # Calculate (X^T X)^(-1) X^T Y
        beta = torch.linalg.lstsq(X, Y).solution
        slope, intercept = beta[0].item(), beta[1].item()
        fitted_y = slope * t + intercept
        T60 = -60 / slope
        fitted_lines.append(fitted_y)

    return fitted_lines


def detectNoiseFloorLevel(edc, fs=16000):
    # Convert edc to a PyTorch tensor if it's not already
    if not isinstance(edc, torch.Tensor):
        edc = torch.tensor(edc, dtype=torch.float32)
    edc = torch.squeeze(edc)
    
    # Assuming edc is a single curve for simplicity, but you can adapt this code to handle multiple curves.
    
    # Select the last 10% of the EDC samples to approximate the noise floor
    last_10_percent_index = int(0.9 * edc.size(0))
    samples_for_noise_floor = edc[last_10_percent_index:]
    
    # The noise floor level is approximated as the mean value of the last 10% of samples
    noise_floor_level = torch.mean(samples_for_noise_floor).item()
    
    return noise_floor_level

if __name__ == "__main__" :

    t_vals = torch.ones(128,1)
    a_vals = torch.ones(128,1)
    noise_level = torch.ones(128,1)


    # random_t_vals = torch.rand_like(t_vals)
    random_t_vals = (t_vals)*1.6068
    random_a_vals = (a_vals)*1.0560
    noise_level_val1 = (noise_level)*1e-7#e-9


    random_t_vals_diff = (t_vals)*1.6068
    random_a_vals_diff = (a_vals)*1.0560
    noise_level_val2 = (noise_level)*1e-7




    factor1 = 48000*16000/118462
    factor2 = 48000*16000/90000

    # print(factor1,factor2)

    time_axis1 = torch.linspace(0, 16000/factor1, 118462)
    time_axis2 = torch.linspace(0, 16000/factor2, 90000)
    time_axis3 = torch.linspace(0, 16000/factor1, 16000)

    edc_t1_with_noise = generate_synthetic_edc_torch(random_t_vals, random_a_vals, noise_level_val1, time_axis1,w_noise=True)
    edc_t1_without_noise = generate_synthetic_edc_torch(random_t_vals, random_a_vals, noise_level_val1, time_axis1)

    edc_t2_with_noise = generate_synthetic_edc_torch(random_t_vals, random_a_vals, noise_level_val2, time_axis2,w_noise=True)
    edc_t2_without_noise = generate_synthetic_edc_torch(random_t_vals, random_a_vals, noise_level_val2, time_axis2)




    y_t2_without_noise=(edc_t2_without_noise.cpu().detach().numpy())
    ydb_t2_without_noise=((10*np.log10(y_t2_without_noise)))





    y_t1_with_noise=(edc_t1_with_noise.cpu().detach().numpy())
    ydb_t1_with_noise=((10*np.log10(y_t1_with_noise)))
    RT_t1_with_noise = EDC2T60(ydb_t1_with_noise)
    noise_floor_level_t1_with_noise = detectNoiseFloorLevel(ydb_t1_with_noise)
    # print("noise_floor_level",noise_floor_level)


    # RT = torch.from_numpy(np.array(RT)).float()

    y_t2_with_noise=(edc_t2_with_noise.cpu().detach().numpy())
    ydb_t2_with_noise=((10*np.log10(y_t2_with_noise)))



    y_t1_without_noise=(edc_t1_without_noise.cpu().detach().numpy())
    ydb_t1_without_noise=((10*np.log10(y_t1_without_noise)))


    noise_t1= noise_level_val1 * torch.linspace(len(time_axis1), 1, len(time_axis1))
    noise_t2 = noise_level_val1 * torch.linspace(len(time_axis2), 1, len(time_axis2))


    # noise_diff= noise_level_val2 * torch.linspace(len(time_axis1), 1, len(time_axis1))

    # noise_diff= noise_diff[1,:].detach().cpu()
    # noise_diff= 10*np.log10(noise_diff)

    n_t1= noise_t1[1,:].detach().cpu()
    n_t1= 10*np.log10(n_t1)

    n_t2= noise_t2[1,:].detach().cpu()
    n_t2= 10*np.log10(n_t2)





plt.figure()

# Plotting the curves with a linewidth of 3
# plt.plot(np.squeeze(time_axis1), np.squeeze(ydb_diff[1, :]), label='Curve for timeaxis2', linewidth=3)
plt.plot(np.squeeze(time_axis1), np.squeeze(ydb_t1_with_noise[1, :]), label='Curve for timeaxis1', linewidth=3)
plt.plot(np.squeeze(time_axis2), np.squeeze(ydb_t2_with_noise[1, :]), label='Curve for time_axis2', linewidth=3)

plt.plot(np.squeeze(time_axis1), np.squeeze(ydb_t1_without_noise[1, :]), label='Curve without Noise Floor for timeaxis1', linewidth=3)
plt.plot(np.squeeze(time_axis2), np.squeeze(ydb_t2_without_noise[1, :]), label='Curve without Noise Floor for time_axis2', linewidth=3)

plt.plot(np.squeeze(time_axis1), n_t1, label='Noise Floor for timeaxis1', linewidth=3)
plt.plot(np.squeeze(time_axis2), n_t2, label='Noise Floor for timeaxis2', linewidth=3)

# plt.plot(np.squeeze(time_axis1), noise_diff, label='Noise Floor 2 for timeaxis1', linewidth=3)

# Uncommented part
# plt.plot(np.squeeze(time_axis1), RT[1], label='Regression', linewidth=3)    
# plt.plot(np.squeeze(time_axis1), n, label='Noise Floor for timeaxis1', linewidth=3)
# plt.plot(np.squeeze(time_axis1), noise_diff, label='Noise Floor for different values', linewidth=3)
# plt.ylim(-80, 10)

# Display the legend with a font size of 16
plt.legend(fontsize=18)

# Set the label for x and y axis with font size 20
plt.xlabel("Time (secs)", fontsize=28)
plt.ylabel("Amplitude (dB)", fontsize=28)

# Set the tick params for both axes with label size 16
plt.tick_params(axis='x', labelsize=24)
plt.tick_params(axis='y', labelsize=24)

# Show the plot
plt.show()

