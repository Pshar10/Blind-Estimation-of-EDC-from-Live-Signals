from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

"""A script to read from tensorboard logs and plot curves"""

# Path to your TensorBoard log directory
log_dir = '/home/prsh7458/work/R2D/S2IR/'

# Initialize an event accumulator
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

# Lists to hold the extracted data
steps = []
loss_G_values = []
loss_D_values = []
MAE_step_values = []

# Extracting the scalar data
if 'Loss/loss_G' in event_acc.Tags()['scalars']:
    for event in event_acc.Scalars('Loss/loss_G'):
        steps.append(event.step)
        loss_G_values.append(event.value)

if 'Loss/loss_D' in event_acc.Tags()['scalars']:
    for event in event_acc.Scalars('Loss/loss_D'):
        loss_D_values.append(event.value)

if 'Loss/MAE_step' in event_acc.Tags()['scalars']:
    for event in event_acc.Scalars('Loss/MAE_step'):
        MAE_step_values.append(event.value)

def smooth_curve(points, factor=0.98):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)

# Convert lists to numpy arrays for plotting
steps = np.array(steps)
loss_G_values = np.array(loss_G_values)
loss_D_values = np.array(loss_D_values)
MAE_step_values = np.array(MAE_step_values)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(steps, smooth_curve(loss_G_values), label='Generator Loss',linewidth=3)
plt.plot(steps, smooth_curve(loss_D_values), label='Discriminator loss',linewidth=3)
# plt.plot(steps, 10 * np.log10(MAE_step_values), label='Loss/MAE_step')

plt.xlabel('Number of Steps', fontsize=20)
plt.ylabel('Loss(dB)', fontsize=20)
plt.title('Training Loss Over Steps', fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.show()
