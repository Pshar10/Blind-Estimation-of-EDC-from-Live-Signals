from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns
import math

"""This script is to demonstrate plotting of curves for each training and testing steps per epoch"""




def extract_tb_data_for_selected_epochs(logdir, tag, selected_epochs, steps_per_epoch):
    ea = event_accumulator.EventAccumulator(logdir,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    all_data = []
    for epoch in selected_epochs:
        start_step = (epoch - 1) * steps_per_epoch
        end_step = epoch * steps_per_epoch
        epoch_values = [s.value for s in ea.scalars.Items(tag) if start_step <= s.step < end_step]
        all_data.extend([(epoch, value) for value in epoch_values])
    
    return all_data

def plot_violin_all_epochs(data, title):
    plt.figure(figsize=(15, 6))
    
    # Convert list of tuples to DataFrame for easy plotting
    import pandas as pd
    df = pd.DataFrame(data, columns=['Epoch', 'Loss'])

    # Create the violin plot
    sns.violinplot(x='Epoch', y='Loss', data=df, palette="muted", scale='width')

    # Overlay individual data points
    sns.stripplot(x='Epoch', y='Loss', data=df, color='k', alpha=0.15, jitter=True, size=2)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Values')
    plt.show()


# Parameters
logdir1 = '/home/prsh7458/work/R2D/training_runs/R2DNet_128_8e-05_100_24-12-2023_14-54-12'
logdir2 = '/home/prsh7458/work/R2D/training_runs/R2DNet_128_8e-05_100_24-12-2023_19-32-28'
logdir3 = '/home/prsh7458/work/R2D/training_runs/R2DNet_128_8e-05_80_18-01-2024_22-49-17' #flex
    
logdir = logdir3



tag = 'Loss/Total_train_step'
selected_epochs = range(20, 101, 20)  # Epochs 10, 20, ..., 100
batch_size = 128
train_dataset_size = 98000
train_steps_per_epoch = math.ceil(train_dataset_size / batch_size)

# Extract and plot data
epoch_loss_data = extract_tb_data_for_selected_epochs(logdir, tag, selected_epochs, train_steps_per_epoch)
plot_violin_all_epochs(epoch_loss_data, 'Violin Plots of Loss Values Across different Epochs for Train data')



batch_size = 64
test_dataset_size = 21000
test_steps_per_epoch = math.ceil(test_dataset_size / batch_size)
tag_test = 'Loss/Total_test_step'
epoch_loss_data = extract_tb_data_for_selected_epochs(logdir, tag, selected_epochs, test_steps_per_epoch)
#plot_violin_all_epochs(epoch_loss_data, 'Violin Plots of Loss Values Across different Epochs for Test data')



