from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns

"""Script showing simple execution of reading from tensorboard logs and then plotting them"""

def extract_tb_data_for_epoch(logdir, tag, epoch, steps_per_epoch):
    ea = event_accumulator.EventAccumulator(logdir,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    if tag in ea.scalars.Keys():
        scalar_events = ea.scalars.Items(tag)
        # Calculate the step range for the desired epoch
        start_step = (epoch - 1) * steps_per_epoch
        end_step = epoch * steps_per_epoch
        # Filter values for the specific epoch
        values = [s.value for s in scalar_events if start_step <= s.step < end_step]
        return values
    else:
        return []

def plot_violin(data, title):
    plt.figure(figsize=(10, 6))
    
    # Create the violin plot
    sns.violinplot(data=data)  # 'inner=None' to remove the bars inside the violins

    # Add stripplot to show individual data points
    sns.stripplot(data=data, color='black', alpha=0.3, jitter=True)  # 'jitter=True' for a slight horizontal spread

    plt.title(title)
    plt.xlabel('Loss Values')
    plt.show()


# Path to TensorBoard log directory and parameters
logdir = '/home/prsh7458/work/R2D/training_runs/R2DNet_128_8e-05_100_24-12-2023_14-54-12'
tag = 'Loss/Total_test_step'  # Adjust this tag based on what you need
epoch = 100
import math

# Dataset sizes and batch size
train_dataset_size = 98000
test_dataset_size = 21000
batch_size = 128

# Calculating steps per epoch
train_steps_per_epoch = math.ceil(train_dataset_size / batch_size)
test_steps_per_epoch = math.ceil(test_dataset_size / batch_size)

# Extract and plot data for a specific epoch
epoch = 100

# For Test Data
test_data = extract_tb_data_for_epoch(logdir, 'Loss/Total_test_step', epoch, test_steps_per_epoch)
if test_data:
    plot_violin(test_data, f'Test Total Loss at Epoch {epoch}')
else:
    print(f"No test data found for Epoch {epoch}")

# For Train Data (if needed)
train_data = extract_tb_data_for_epoch(logdir, 'Loss/Total_train_step', epoch, train_steps_per_epoch)
if train_data:
    plot_violin(train_data, f'Train Total Loss at Epoch {epoch}')
else:
    print(f"No train data found for Epoch {epoch}")
