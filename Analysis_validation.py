from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from numpy.polynomial.polynomial import Polynomial


global edc_loss_analysis

log_dir = f'/home/prsh7458/work/R2D/test_runs_validation/R2DNet_R3vival_dataset'

def extract_tb_data(logdir, tag):
    ea = event_accumulator.EventAccumulator(logdir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    if tag in ea.scalars.Keys():
        scalar_events = ea.scalars.Items(tag)
        values = [s.value for s in scalar_events]
        return values
    else:
        return []
    

def plot_violin(data):
    plt.figure(figsize=(10, 6))

    # Create a DataFrame for plotting
    loss_data = []
    for i, loc_data in enumerate(data):
                if not edc_loss_analysis:
                     loc_data = loc_data**0.5
                loss_data.append({'Location': 'R3VIVAL DATASET', 'Loss': loc_data})

    df = pd.DataFrame(loss_data)

    # Create the violin plot with location names
    sns.violinplot(x='Location', y='Loss', data=df,palette="muted", scale='width')


    # sns.stripplot(x='Location', y='Loss', data=df, color='black', alpha=0.02, jitter=True)

    plt.title(f'L1 EDC Validation Loss for R3vival dataset',fontsize=26)
    plt.xlabel('R3vival dataset',fontsize=21)
    plt.ylabel('Loss Values',fontsize=21)
    plt.ylim([-1,20]) if edc_loss_analysis else plt.ylim([0,1]) 
    
    plt.tick_params(axis='y', labelsize=22)  
    plt.tick_params(axis='x', labelsize=22) 
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_violin_and_box(data):
    plt.figure(figsize=(10, 6))

    # Prepare the data
    loss_data = [{'Location': 'R3VIVAL DATASET', ' ': loc_data} for loc_data in data]
    df = pd.DataFrame(loss_data)

    # Create the violin plot
    sns.violinplot(data=df, palette="muted",width=0.2, scale='width')

    # Overlay the box plot
    sns.boxplot( data=df, color='white', width=0.02, fliersize=0)  # Hide outliers for cleaner visualization

    # Plot customization
    plt.title('Total L1 EDC Loss for R3VIVAL Dataset', fontsize=28)
    # plt.xlabel('R3vival Dataset', fontsize=21)
    plt.ylabel('L1 Loss (dB)', fontsize=28)

    # Conditionally set the y-axis limits
    plt.ylim([-1, 20]) if edc_loss_analysis else plt.ylim([0, 1])

    plt.tick_params(axis='y', labelsize=28)
    plt.tick_params(axis='x', labelsize=28)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_loss_boxplot(data):
    plt.figure(figsize=(10, 6))
    exp = 0.5 if not edc_loss_analysis else 1
    loss_data = [{'Location': 'R3VIVAL DATASET', ' ': loc_data**exp} for loc_data in data]
    df = pd.DataFrame(loss_data)
    
    sns.boxplot(data=df, width=0.4, fliersize=0.5)
    
    plt.title('DT10 Validation Loss for R3VIVAL DATASET', fontsize=28) if plot_DT_analysis else plt.title(r'$T_{0-15}$ Validation Loss for R3VIVAL DATASET', fontsize=28)

    plt.ylabel('MAE Loss (seconds)', fontsize=28)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim([-1, 20]) if edc_loss_analysis else plt.ylim([0, 1])
    
    plt.tick_params(axis='y', labelsize=28)
    plt.tick_params(axis='x', labelsize=28)
    
    plt.show()

def plot_loss_violinplot(data):
    plt.figure(figsize=(4, 6))
    sns.violinplot(data=data, inner="quartile", palette="muted", scale='width', showfliers=True)
    sns.stripplot(data=data, color='black', size=3, jitter=True, alpha=0.05)
    plt.title('Violin Plot of Loss Values in dB', fontsize=20)
    plt.ylabel('Loss in dB', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim([0, 20])
    plt.show()
from ptitprince import RainCloud

def plot_loss_raincloud(data):


    plt.figure(figsize=(4, 6))
    df = pd.DataFrame(data)
    sns.violinplot(data=data,palette="muted", scale='width')
    # sns.stripplot(data=data, color="black", size=3, jitter=True, alpha=0.5)
    
    plt.title('Raincloud Plot of Loss Values in dB', fontsize=20)
    plt.ylabel('Loss in dB', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim([0, 20])
    plt.show()

def plot_loss_swarmplot(data):
    plt.figure(figsize=(6, 6))  # Adjusted figure size for a wider plot
    swarmplot = sns.swarmplot(data=data, color='black', size=3, alpha=0.6)

    # Dynamically adjust the y-limits based on the data to prevent clipping
    ymin, ymax = swarmplot.get_ylim()
    ypadding = (ymax - ymin) * 0.05  # Add 5% padding to the y-limits
    plt.ylim(ymin - ypadding, ymax + ypadding)

    plt.title('Swarm Plot of Loss Values in dB', fontsize=20)
    plt.ylabel('Loss in dB', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_histogram_true_values_for_room(data):
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Histogram of True T60 Values')
        plt.xlabel('True T60 (seconds)')
        plt.ylabel('Frequency')
        plt.xlim([0,1])
        plt.tight_layout()
        plt.show()
def plot_kde_true_values_for_room(data):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data, shade=True, color="blue", alpha=0.7, linewidth=1.5)
    plt.title('KDE of True T60 Values')
    plt.xlabel('True T60 (seconds)')
    plt.ylabel('Density')
    plt.xlim([0, 1])
    plt.tight_layout()
    plt.show()

def plot_error_percentage_boxplot(log_dir, bin_resolution=0.01, save=False, connect_medians=True, filename='error_percentage_boxplot.png', save_base_dir=''):
    # Assuming extract_tb_data is a function you've defined elsewhere to extract your data
    all_true_values = extract_tb_data(log_dir, 'Loss/T_true')
    all_pred_values = extract_tb_data(log_dir, 'Loss/T_pred')

    # Calculate and adjust the percentage error for each prediction
    all_adjusted_percentage_errors = [((abs(pred - true) / true )* 100) if (abs(pred - true) / true)*100 > 5 else 0 for true, pred in zip(all_true_values, all_pred_values)]
    all_adjusted_percentage_errors = np.abs(all_adjusted_percentage_errors)

    # Determine the bins for true values
    max_value = max(all_true_values)
    min_value = 0.3
    bins = np.arange(min_value, max_value + bin_resolution, bin_resolution)
    binned_adjusted_percentage_errors = [[] for _ in range(len(bins)-1)]

    # Sort adjusted percentage errors into bins
    for true_value, adjusted_percentage_error in zip(all_true_values, all_adjusted_percentage_errors):
        bin_index = np.digitize(true_value, bins) - 1  # Find the bin index for each true_value
        if 0 <= bin_index < len(binned_adjusted_percentage_errors):
            binned_adjusted_percentage_errors[bin_index].append(adjusted_percentage_error)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.boxplot(binned_adjusted_percentage_errors, positions=bins[:-1] + bin_resolution/2, widths=bin_resolution * 0.8, showfliers=False)
    plt.xlim(left=bins[0], right=bins[-1])

    # Adjust x-ticks
    skip = max(1, int(len(bins) / 20))
    xtick_positions = bins[::skip] + bin_resolution/2
    xtick_labels = [f"{b:.2f}" for b in bins[::skip]]
    plt.xticks(ticks=xtick_positions, labels=xtick_labels, rotation=90)

    plt.grid(axis='y')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('True Values', fontsize=16)
    plt.ylabel('Absolute Error Percentage', fontsize=16)
    plt.title('Absolute Prediction Error Percentage by True Value', fontsize=16)

    if connect_medians:
        medians = [np.median(adjusted_percentage_err) if adjusted_percentage_err else 0 for adjusted_percentage_err in binned_adjusted_percentage_errors]
        plt.plot(bins[:-1] + bin_resolution/2, medians, 'r-o', markersize=3, linestyle='dashdot')  # Small red dots and dashed line

    if save:
        if not save_base_dir:
            raise ValueError("Save directory not provided.")
        if not os.path.exists(save_base_dir):
            os.makedirs(save_base_dir)
        save_path = os.path.join(save_base_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


global plot_DT_analysis
edc_loss_analysis = False  ####### TRUE if we want to analyse EDC else T60 analysis
DT_analysis = False
plot_DT_analysis = DT_analysis

if not edc_loss_analysis:
    if DT_analysis: 
        tag = 'Loss/total_loss_test_DT' 
    else:
        tag = 'Loss/total_loss_test' 
else:
    tag = 'Loss/total_loss_test_edc'



data = extract_tb_data(log_dir, tag)
true_values = extract_tb_data(log_dir, 'Loss/T_true')

plot_loss_boxplot(data) if not edc_loss_analysis else plot_violin_and_box(data)

# plot_kde_true_values_for_room(true_values)

# plot_error_percentage_boxplot(log_dir, bin_resolution=0.1, save=False, connect_medians=True, filename='error_percentage_boxplot.png', save_base_dir='')