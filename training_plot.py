from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import mplcursors
import tensorflow as tf


"""A script to plot training cirves from tensorboard logs"""

def smooth_curve(points, factor=0.98):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)


# Load the TensorBoard log file
# log_dir = '/home/prsh7458/work/R2D/analysis_runs_folder/R2DNet_128_8e-05_80_10-03-2024_11-02-07'  # Update this path\
# log_dir = '/home/prsh7458/work/R2D/histoty_training_logs/R2DNet_05-12-2023_13-47-53'  # sample length loss
s2ir=False
sample_loss_plot=False

if not s2ir and not sample_loss_plot:
    # log_dir = '/home/prsh7458/work/R2D/analysis_runs_folder/R2DNet_128_8e-05_80_09-03-2024_18-48-52'  # 7 layered CNN
    # log_dir = '/home/prsh7458/work/R2D/analysis_runs_folder/R2DNet_128_8e-05_80_12-03-2024_20-41-13'  # 7 layered CNN with TAE
    # log_dir = '/home/prsh7458/work/R2D/analysis_runs_folder/R2DNet_128_8e-05_80_10-03-2024_11-02-07'  # CNN with S2IR
    log_dir = '/home/prsh7458/work/R2D/histoty_training_logs/definite_models/fins/R2DNet_128_8e-05_80_17-02-2024_16-14-15'  # FiNS

elif sample_loss_plot:
    log_dir = '/home/prsh7458/work/R2D/histoty_training_logs/R2DNet_05-12-2023_11-36-53'

# else:
#     log_dir = '/home/prsh7458/work/R2D/S2IR/'  # Update this path


ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()  # Loads the log file

# Extract the scalars
steps = []
total_test_loss = []
ere_test_step = []
test_edc_train = []
ere_train_step = []
edc_train_step = []
edc_test_step = []
loss_D =[]
loss_G =[]
Sample_loss =[]

if (not s2ir) and (not sample_loss_plot):

    for event in ea.Scalars('Loss/Total_test_step'):
        steps.append(event.step)
        total_test_loss.append(event.value)

    for event in ea.Scalars('Loss/ERE_test_step'):
        ere_test_step.append(event.value)


    for event in ea.Scalars('Loss/ERE_train_step'):
        ere_train_step.append(event.value)

    for event in ea.Scalars('Loss/Total_train_step'):
        test_edc_train.append(event.value)

    for event in ea.Scalars('Loss/MAE_step'):
        edc_train_step.append(event.value)

    for event in ea.Scalars('Loss/Test_EDC_step'):
        edc_test_step.append(event.value)


elif sample_loss_plot:
    for event in ea.Scalars('Loss/Sample Length_step'):
        steps.append(event.step)
        Sample_loss.append(np.log10(event.value))

# else:
#     for event in ea.Scalars('Loss/loss_D'):
#         steps.append(event.step)
#         loss_D.append(event.value)

#     for event in ea.Scalars('Loss/loss_G'):
#         loss_G.append(event.value)


plot_edc = 0
plot_ere = 0
plot_total = not (plot_edc ^ plot_ere )

plt.figure(figsize=(10, 6))



if plot_total:
    plt.plot(steps, smooth_curve(total_test_loss), label='Total Test Loss',linewidth=3)
    plt.plot(steps, smooth_curve(test_edc_train), label='Total Train Step',linewidth=3)

if plot_ere:
    plt.plot(steps, smooth_curve(ere_test_step), label='EDC Test Loss upto -15dB',linewidth=3)
    plt.plot(steps, smooth_curve(ere_train_step), label='EDC Train Step upto -15dB',linewidth=3)

if plot_edc:
    plt.plot(steps, smooth_curve(edc_train_step), label='EDC Train Loss',linewidth=3)
    plt.plot(steps, smooth_curve(edc_test_step), label='EDC Test Step',linewidth=3)


# plt.plot(steps,smooth_curve(Sample_loss), label='Sample length Loss',linewidth=3)


# plt.plot(steps, smooth_curve(loss_G), label='Generator Loss',linewidth=3)
# plt.plot(steps, smooth_curve(loss_D), label='Discriminator Loss',linewidth=3)
    
plt.xlabel('Number of Steps', fontsize=28)
plt.ylabel('Loss(dB)', fontsize=28)
plt.ylim([0,25])
plt.title('Train and Test Losses Over Steps', fontsize=28) if not s2ir else plt.title('Train Losses Over Steps', fontsize=28)
plt.legend(fontsize=28)
plt.grid(True)
mplcursors.cursor(hover=False)
plt.tick_params(axis='x', labelsize=24)
plt.tick_params(axis='y', labelsize=24)
plt.show()