from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import numpy as np
import os
import warnings
import h5py
import ptitprince as pt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)




"""This script is to do individual and aggregated analysis of all the rooms combined"""




global save_base_dir 
global edc_loss_analysis
global DT_analysis

save_base_dir = '/home/prsh7458/Desktop/scratch4/R2D/loss_graphs'  # Base directory to save plots


def read_mat_file(file_path, variable_name):
    with h5py.File(file_path, 'r') as file:
        data = np.array(file[variable_name])
    return data



def extract_tb_data(logdir, tag):
    ea = event_accumulator.EventAccumulator(logdir,
                                            size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    if tag in ea.scalars.Keys():
        scalar_events = ea.scalars.Items(tag)
        values = [s.value for s in scalar_events]
        return values
    else:
        return []


def extract_tb_data_for_T60(logdir, true_tag='T_true', pred_tag='T_pred'):
    ea = event_accumulator.EventAccumulator(logdir,
                                            size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    true_values, pred_values = [], []
    if true_tag in ea.scalars.Keys():
        true_values = [s.value for s in ea.scalars.Items(true_tag)]
    if pred_tag in ea.scalars.Keys():
        pred_values = [s.value for s in ea.scalars.Items(pred_tag)]

    return true_values, pred_values

def plot_confusion_matrix_for_room(data, room, room_locations, save=False):
    # Prepare data for DataFrame
    all_true_values, all_pred_values = [], []

    # Determine min and max values for binning
    bin_size = 0.2
    bins = np.arange(0, 5 + bin_size, bin_size)
    true_binned = [bin_T60_values(values[0], range_min=0, range_max=5) for values in data if values[0]]
    pred_binned = [bin_T60_values(values[1], range_min=0, range_max=5) for values in data if values[1]]



    # Iterate through each location's data
    for i, (true_bin, pred_bin) in enumerate(zip(true_binned, pred_binned)):
        conf_matrix = confusion_matrix(true_bin, pred_bin, labels=range(len(bins)-1))

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[f"{bin*bin_size:.1f}" for bin in range(len(bins)-1)],
                    yticklabels=[f"{bin*bin_size:.1f}" for bin in range(len(bins)-1)])
        plt.title(f'Confusion Matrix for Room {room} - {room_locations[i]}')
        plt.xlabel('Predicted T60 Bin')
        plt.ylabel('True T60 Bin')

        if save:
            save_path = os.path.join(save_base_dir, f"{room}_{room_locations[i]}_confusion_matrix.png")
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def bin_T60_values(values, bin_size=0.1, range_min=0, range_max=1):
    # Binning T60 values within a specified range and resolution
    bins = np.arange(range_min, range_max + bin_size, bin_size)
    binned_values = np.digitize(values, bins, right=False) - 1
    # Adjust bin indices to fall within the range of 0 to 5 seconds
    # binned_values = np.clip(binned_values, 0, int(range_max / bin_size) - 1)
    return binned_values

def plot_scatter_for_room(data, room, room_locations, save=False):
    for i, (true_values, pred_values) in enumerate(data):
        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, pred_values, alpha=0.5)
        plt.plot([0, 5], [0, 5], 'r--')  # Diagonal line for perfect predictions
        plt.xlim(0, 5)
        plt.ylim(0, 4)
        plt.title(f'Scatter Plot for Room {room} - {room_locations[i]}')
        plt.xlabel('True T60')
        plt.ylabel('Predicted T60')

        if save:
            save_path = os.path.join(save_base_dir, f"{room}_{room_locations[i]}_scatter.png")
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


# def plot_confusion_matrix_diag(combined_data_T, room, room_locations, bin_resolution=0.1, save=False ):
#     for i, (true_values, pred_values) in enumerate(combined_data_T):
#         # Determine the maximum T60 value for binning
#         max_t60 = max(max(true_values), max(pred_values))

#         # Define the bins for the histogram based on the bin resolution and the range of T60 values
#         bins = np.arange(0, max_t60 + bin_resolution, bin_resolution)

#         # Bin the true and predicted values
#         true_binned = np.digitize(true_values, bins)
#         pred_binned = np.digitize(pred_values, bins)

#         # Calculate the confusion matrix
#         cm = confusion_matrix(true_binned, pred_binned)

#         # Plot the confusion matrix
#         plt.figure(figsize=(10, 7))
#         sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
#         plt.title(f'Confusion Matrix for Room {room} - {room_locations[i]}')
#         plt.xlabel('Predicted T60 Bin')
#         plt.ylabel('True T60 Bin')

#         if save:
#             if save_base_dir:
#                 save_path = os.path.join(save_base_dir, f"{room}_{room_locations[i]}_confusion_matrix.png")
#                 plt.savefig(save_path)
#             else:
#                 print("Save directory is not specified")
#             plt.close()
#         else:
#             plt.show()


def plot_violin(data, room, room_locations,mae=False,save=False):
    plt.figure(figsize=(12, 8))

    # Create a DataFrame for plotting
    loss_data = []
    for i, loc_data in enumerate(data):
        for value in loc_data:
            if mae:
                loss_data.append({'Location': room_locations[i], 'Loss': value**0.5})
            else:
                loss_data.append({'Location': room_locations[i], 'Loss': value})

    df = pd.DataFrame(loss_data)

    # Create the violin plot with location names
    sns.violinplot(x='Location', y='Loss', data=df,palette="muted", scale='width')


    # sns.stripplot(x='Location', y='Loss', data=df, color='black', alpha=0.02, jitter=True)

    plt.title(f'Total Test Loss for Room {room}',fontsize=28)
    plt.xlabel('Location',fontsize=28)
    plt.ylabel('L1 Loss (dB)',fontsize=28)
    plt.ylim([-1,20]) if edc_loss_analysis else plt.ylim([0,1]) 
    
    plt.tick_params(axis='y', labelsize=28)  
    plt.tick_params(axis='x', labelsize=28) 
    plt.grid(True, linestyle='--', alpha=0.5)
    if save:
        save_path = os.path.join(save_base_dir, f"{room}_violin.png")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()




def plot_scatter(data, room, room_locations,save=False):
    plt.figure(figsize=(10, 6))

    # Create a DataFrame for plotting
    loss_data = []
    for i, loc_data in enumerate(data):
        for value in loc_data:
            loss_data.append({'Location': room_locations[i], 'Loss': value})
    df = pd.DataFrame(loss_data)

    # Create the scatter plot with jitter and a color palette
    sns.stripplot(x='Location', y='Loss', data=df, jitter=0.2, palette='Set2', alpha=0.5, size=4)

    plt.title(f'Total Test Loss for Room {room}', fontsize=16)
    plt.xlabel('Location', fontsize=14)
    plt.ylabel('Loss Values', fontsize=14)

    # Set the ticks and grid lines
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim([-1,4])
    # Adjust legend
    plt.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust the layout to fit all elements
    plt.tight_layout()

    if save:
        save_path = os.path.join(save_base_dir, f"{room}_scatter.png")
        plt.savefig(save_path)
        plt.close()
    else: plt.show()




def plot_histograms(data, room, room_locations,save=False):
    # Prepare data for DataFrame
    loss_data = []
    for i, loc_data in enumerate(data):
        for value in loc_data:
            loss_data.append({'Location': room_locations[i], 'Loss': value})

    # Create DataFrame
    df = pd.DataFrame(loss_data)

    # Define fixed bins
    bins = np.arange(0, 3, 0.1)  # Bins: 0-2, 2-4, ..., 16-18

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Width of each bar and calculate group width
    width = 1 / (len(room_locations) + 1)
    group_width = width * len(room_locations)

    # Plot bars for each location
    for i, location in enumerate(room_locations):
        subset = df[df['Location'] == location]['Loss']
        hist, bin_edges = np.histogram(subset, bins=bins)
        bar_positions = bin_edges[:-1] + i * width
        plt.bar(bar_positions, hist, width=width, alpha=0.7, label=location, edgecolor='black')

    # Set plot title and labels
    plt.title(f'Total Test Loss for Room {room}')
    plt.xlabel('Loss Values')
    plt.ylabel('Frequency')
    plt.ylim([0,2100])

    # Adjust x-ticks
    group_centers = bin_edges[:-1] + group_width / 2
    plt.xticks(ticks=group_centers, labels=[f'{int(b)}-{int(b)+1}' for b in bins[:-1]])
    plt.grid(True, linestyle='--', alpha=0.5)

    # Display legend
    plt.legend()
    if save:
        save_path = os.path.join(save_base_dir, f"{room}_histogram.png")
        plt.savefig(save_path)
        plt.close()
    else: plt.show()
    

import math

def plot_boxplot(data, room, room_locations, save=False, mae=False):
    # Prepare data for DataFrame
    loss_data = []
    for i, loc_data in enumerate(data):
        for value in loc_data:
            # Skip the loop iteration if value is infinity
            if math.isinf(value):
                continue
            if mae:
                loss_data.append({'Location': room_locations[i], 'Loss': value**0.5})
            else:
                loss_data.append({'Location': room_locations[i], 'Loss': value})

    # Create DataFrame
    df = pd.DataFrame(loss_data)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Create box plot
    sns.boxplot(x='Location', y='Loss', data=df, palette='muted', fliersize=0.5)

    # Add stripplot for individual data point visualization
    sns.stripplot(x='Location', y='Loss', data=df, color='black', alpha=0.0005, jitter=True)

    # # Calculate upper whisker value
    # upper_whisker = df.groupby('Location')['Loss'].quantile(0.75) + 1.5 * (
    #     df.groupby('Location')['Loss'].quantile(0.75) - df.groupby('Location')['Loss'].quantile(0.25))

    # # Calculate mean value
    # mean_value = df.groupby('Location')['Loss'].mean()

    # # Annotate upper whisker and mean values on the plot
    # for loc, upper, mean in zip(room_locations, upper_whisker, mean_value):
    #     plt.text(room_locations.index(loc), upper, f'Upper Whisker: {upper:.2f}', horizontalalignment='center',
    #              verticalalignment='bottom')
    #     plt.text(room_locations.index(loc), mean, f'Mean: {mean:.2f}', horizontalalignment='center',
    #              verticalalignment='top', fontsize=16)

    # Set plot title and labels
    if not DT_analysis:
        plt.title(f'Total Test Loss for Room {room}', fontsize=26) if edc_loss_analysis else plt.title(fr'$T_{{0-15}}$ Test Loss for Room {room}', fontsize=26)

    else : plt.title(f'DT10 Loss for Room {room}', fontsize=26)
    plt.xlabel('Location', fontsize=21)
    plt.ylabel('MAE Loss (seconds)', fontsize=21)
    plt.ylim([0, 20]) if edc_loss_analysis else plt.ylim([0,1]) # Adjust ylim based on your data range
    plt.tick_params(axis='y', labelsize=22)  
    plt.tick_params(axis='x', labelsize=22) 

    plt.grid(True, linestyle='--', alpha=0.5)

    # Display the plot or save it
    if save:
        if not DT_analysis:
            name = f"T60_{room}_box.png" if not edc_loss_analysis else f"{room}_box.png"
        else: name = f"DT10_{room}_box.png"
        save_path = os.path.join(save_base_dir, name)
        plt.savefig(save_path)
        plt.close()
    else: 
        plt.show()


def plot_cdf(data, room, room_locations,save=False):
    # Prepare data for DataFrame
    loss_data = []
    for i, loc_data in enumerate(data):
        for value in loc_data:
            loss_data.append({'Location': room_locations[i], 'Loss': value})

    # Create DataFrame
    df = pd.DataFrame(loss_data)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Create CDF plot for each location
    for location in room_locations:
        subset = df[df['Location'] == location]['Loss']
        sns.ecdfplot(subset, label=location)

    # Set plot title and labels
    plt.title(f'Cumulative Distribution of Loss for Room {room}')
    plt.xlabel('Loss Values')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, linestyle='--', alpha=0.5)
    # Display legend
    plt.legend()
    if save:
        save_path = os.path.join(save_base_dir, f"{room}_cumulative_frequency_distribution.png")
        plt.savefig(save_path)
        plt.close()
    else: plt.show()


def plot_beeswarm(data, room, room_locations, save=False):
    # Prepare data for DataFrame
    loss_data = []
    for i, loc_data in enumerate(data):
        for value in loc_data:
            loss_data.append({'Location': room_locations[i], 'Loss': value})

    # Create DataFrame
    df = pd.DataFrame(loss_data)

    # Initialize the plot
    plt.figure(figsize=(12, 8))

    # Create bee swarm plot
    sns.swarmplot(x='Location', y='Loss', data=df, size=2)  # Adjusted marker size

    # Set plot title and labels
    plt.title(f'Bee Swarm Plot for Room {room}')
    plt.xlabel('Location')
    plt.ylabel('Loss Values')
    plt.ylim([-1, 4])
    plt.grid(True, linestyle='--', alpha=0.5)

    # Display or save the plot
    if save:
        save_path = os.path.join(save_base_dir, f"{room}_beeswarm.png")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_jointplot(data, room, room_locations, save=False):
    # Prepare data for DataFrame
    combined_data = []
    for i, loc_data in enumerate(data):
        for value in loc_data:
            combined_data.append({'Location': room_locations[i], 'Loss': value})

    # Create DataFrame
    df = pd.DataFrame(combined_data)

    # Since a jointplot compares two continuous variables and 'Location' is categorical,
    # we'll need to convert 'Location' into a numerical format. Here, we'll just demonstrate
    # using 'Loss' against itself for demonstration purposes.
    # In practice, you would replace 'Loss' with a continuous variable relevant to 'Location'.
    jp = sns.jointplot(x='Loss', y='Loss', data=df, kind='reg')

    # Set plot title
    jp.fig.suptitle(f'Joint Plot for Room {room}', size=15, y=1.02)

    # Display or save the plot
    if save:
        save_path = os.path.join(save_base_dir, f"{room}_jointplot.png")
        jp.fig.savefig(save_path)  # Use jp.fig to access the figure object for saving
        plt.close(jp.fig)
    else:
        plt.show()


def plot_raincloud(data, room, room_locations, save=False):
    # Prepare data for DataFrame
    loss_data = []
    for i, loc_data in enumerate(data):
        for value in loc_data:
            loss_data.append({'Location': room_locations[i], 'Loss': value})

    # Create DataFrame
    df = pd.DataFrame(loss_data)

    # Initialize the plot
    plt.figure(figsize=(12, 8))

    # Create raincloud plot
    pt.RainCloud(x='Location', y='Loss', data=df, palette="Set2",pointplot = False)

    # Set plot title and labels
    plt.title(f'Raincloud Plot for Room {room}')
    plt.xlabel('Location')
    plt.ylabel('Loss Values')
    plt.ylim([0, 20.5])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Display or save the plot
    if save:
        save_path = os.path.join(save_base_dir, f"{room}_raincloud.png")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
def plot_confusion_matrix_for_room(combined_data_T, room, room_locations, bin_resolution=0.1, save=False):
    # Iterate over each set of true and predicted values
    for i, (true_values, pred_values) in enumerate(combined_data_T):
        # Determine the maximum T60 value for binning
        # max_t60 = max(max(true_values), max(pred_values))
        max_t60 = 3.5
        
        # Define the bins for the histogram based on the bin resolution and the range of T60 values
        bins = np.arange(0, max_t60 + bin_resolution, bin_resolution)
        
        # Calculate the 2D histogram for the true and predicted T60 values
        counts, xedges, yedges, Image = plt.hist2d(true_values, pred_values, bins=bins, cmap='plasma')
        
        plt.colorbar(Image)  # Show a color bar indicating the counts in the bins
        plt.plot(bins, bins, 'r--')  # Diagonal line for perfect predictions
        plt.xlim(0, max_t60)
        plt.ylim(0, max_t60)
        plt.title(f'Confusion Matrix for Room {room} - {room_locations[i]}')
        plt.xlabel('True T60 [s]')
        plt.ylabel('Predicted T60 [s]')
        
        if save:
            save_path = os.path.join(save_base_dir, f"{room}_{room_locations[i]}_confusion_matrix.png")
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def plot_combined_confusion_matrix(combined_data_T, bin_resolution=0.1, save=False, filename='combined_confusion_matrix.png'):
    # Initialize lists to store all true and predicted values
    all_true_values = []
    all_pred_values = []

    # Assuming combined_data_T is a list of lists, each inner list corresponding to a room
    # and containing tuples for each location (true_values, pred_values)
    for room_data in combined_data_T:
        for true_values, pred_values in room_data:
            all_true_values.extend(true_values)
            all_pred_values.extend(pred_values)

    # Now we have all the true and predicted values for all rooms and locations
    # We can proceed to plot the combined confusion matrix
    max_t60 = max(max(all_true_values), max(all_pred_values))
    # max_t60 = 3
    bins = np.arange(0, max_t60 + bin_resolution, bin_resolution)
    
    plt.figure(figsize=(10, 8))
    counts, xedges, yedges, Image = plt.hist2d(all_true_values, all_pred_values, bins=bins, cmap='plasma')  #Blues
    plt.colorbar(Image)
    plt.plot(bins, bins, 'r--')
    plt.xlim(0, max_t60)
    plt.ylim(0, max_t60)
    plt.title('Combined Confusion Matrix for All Rooms and Locations')
    plt.xlabel('True T60 [s]')
    plt.ylabel('Predicted T60 [s]')
    
    if save:
        if not save_base_dir:
            raise ValueError("Save directory not provided.")
        if not os.path.exists(save_base_dir):
            os.makedirs(save_base_dir)
        save_path = os.path.join(save_base_dir, filename)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_scatter_combined(combined_data_T, save=False, filename='scatter_with_diagonal.png'):
    all_true_values = []
    all_pred_values = []

    # Extract and combine all true and predicted values
    for room_data in combined_data_T:
        for true_values, pred_values in room_data:
            all_true_values.extend(true_values)
            all_pred_values.extend(pred_values)

    # Determine the overall range for plotting
    # max_value = max(max(all_true_values), max(all_pred_values))
    max_value = 3.5
    plt.figure(figsize=(10, 8))

    # Scatter plot of predicted vs true values with slight transparency and smaller size for predicted
    plt.scatter(all_true_values, all_pred_values, alpha=0.1, color='blue', s=10,marker='x', label='Error')

    # Enhancing true values visibility: plot them with a distinct marker, color, and larger size
    plt.scatter(all_true_values, all_true_values, alpha=0.7, color='red', edgecolor='red', s=1, marker='x', label='True')

    # Diagonal line for perfect predictions
    plt.plot([0, max_value], [0, max_value], 'r--')

    plt.legend()
    plt.title('Scatter Plot of True vs. Predicted T60 Values in seconds')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)

    if save:
        full_save_path = os.path.join(save_base_dir, filename) if save_base_dir else filename
        if not os.path.exists(os.path.dirname(full_save_path)):
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
        plt.savefig(full_save_path)
        plt.close()
    else:
        plt.show()


def plot_scatter_combined_with_approx_boxplot(combined_data_T, save=False, filename='scatter_with_approx_boxplot.png'):
    all_true_values = []
    all_pred_values = []

    # Extract and combine all true and predicted values
    for room_data in combined_data_T:
        for true_values, pred_values in room_data:
            all_true_values.extend(true_values)
            all_pred_values.extend(pred_values)

    # Calculate errors
    errors = np.array(all_pred_values) - np.array(all_true_values)

    # print("Legth of array",errors.shape)

    # Determine the overall range for plotting and binning
    max_value = max(max(all_true_values), max(all_pred_values))
    # max_value = 3.5
    bins = np.arange(0, max_value + 0.1, 0.1)
    digitized_true = np.digitize(all_true_values, bins) - 1

    plt.figure(figsize=(12, 8))

    # Scatter plot of predicted vs true values
    plt.scatter(all_true_values, all_pred_values, alpha=0.5, color='blue', s=10, label='Predicted vs True')

    # Diagonal line for perfect predictions
    plt.plot([0, max_value], [0, max_value], 'r--')

    for i in range(len(bins) - 1):
        indices = digitized_true == i
        errors_in_bin = errors[indices]
        if len(errors_in_bin) > 0:
            # Calculate median and interquartile range
            median_error = np.median(errors_in_bin)
            q1, q3 = np.percentile(errors_in_bin, [25, 75])
            bin_center = bins[i] + 0.05

            # Plot median error as a smaller point on the diagonal
            plt.plot(bin_center, bin_center + median_error, 'ko', color='black',alpha=0.5,markersize=3)  # Smaller circle for median
            
            # Plot IQR range around the median
            # Vertical line for IQR
            plt.plot([bin_center, bin_center], [bin_center + q1, bin_center + q3], 'k-', linewidth=1)
            
            # Optional: Horizontal ticks at Q1 and Q3 for clarity
            tick_length = 0.02  # Adjust as needed for visibility
            plt.plot([bin_center - tick_length, bin_center + tick_length], [bin_center + q1, bin_center + q1], 'k-', linewidth=1)
            plt.plot([bin_center - tick_length, bin_center + tick_length], [bin_center + q3, bin_center + q3], 'k-', linewidth=1)


    plt.title('Scatter Plot of True vs. Predicted Values with Approximate Error Indicators')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    plt.legend()

    if save:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_bland_altman_for_room(data, room, room_locations, save=False):
    for i, (true_values, pred_values) in enumerate(data):
        plt.figure(figsize=(10, 6))
        differences = np.array(pred_values) - np.array(true_values)
        means = (np.array(pred_values) + np.array(true_values)) / 2
        mean_difference = np.mean(differences)
        std_difference = np.std(differences)

        plt.scatter(means, differences, alpha=0.5)
        plt.axhline(mean_difference, color='red', linestyle='--')
        plt.axhline(mean_difference + 1.96 * std_difference, color='gray', linestyle='--')
        plt.axhline(mean_difference - 1.96 * std_difference, color='gray', linestyle='--')
        plt.title(f'Bland-Altman Plot for Room {room} - {room_locations[i]}')
        plt.xlabel('Mean T60 [(True+Predicted)/2]')
        plt.ylabel('Difference T60 (Predicted-True)')

        if save:
            save_path = os.path.join(save_base_dir, f"{room}_{room_locations[i]}_bland_altman.png")
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def plot_error_histogram_for_room(data, room, room_locations, save=False):
    for i, (true_values, pred_values) in enumerate(data):
        errors = np.array(pred_values) - np.array(true_values)
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Error Histogram for Room {room} - {room_locations[i]}')
        plt.xlabel('Prediction Error (Predicted - True T60)')
        plt.ylabel('Frequency')
        plt.tight_layout()

        if save:
            save_path = os.path.join(save_base_dir, f"{room}_{room_locations[i]}_error_histogram.png")
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def plot_histogram_true_values_for_room(data, room, room_locations, save=False):
    for i, (true_values, _) in enumerate(data):  # We're only interested in true values
        plt.figure(figsize=(10, 6))
        plt.hist(true_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Histogram of True T60 Values for Room {room} - {room_locations[i]}')
        plt.xlabel('True T60 (seconds)')
        plt.ylabel('Frequency')
        plt.xlim([0,1])
        plt.tight_layout()

        if save:
            save_path = os.path.join(save_base_dir, f"{room}_{room_locations[i]}_true_histogram.png")
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

###### TRAINING DATASET APPLICABLE FUNCTION ##############

def plot_stacked_histogram_for_rooms(data, room_data, bin_resolution=0.01, save=False, save_base_dir=''):
    room_names = ["HL00W", "HL01W", "HL02WL", "HL03W", "HL05W", "HL06W", "HL08W"]
    
    # Convert room identifiers to strings to ensure they are hashable
    room_data_str = [str(room) for room in room_data]
    unique_rooms_str = np.unique(room_data_str)
    
    # Initialize lists to store all true values
    all_true_values = []
    
    # Populate all_true_values with data from all rooms
    for true_values, _ in data:
        all_true_values.extend(true_values)
    
    # Determine the bins for true T60 values
    max_t60 = max(all_true_values)
    min_t60 = min(all_true_values)
    bins = np.arange(min_t60, max_t60 + bin_resolution, bin_resolution)
    
    plt.figure(figsize=(10, 6))
    # This will hold the cumulative histogram counts
    cumulative_counts = np.zeros(len(bins) - 1)
    
    # Plotting the histogram for each room
    for i, room_str in enumerate(unique_rooms_str):
        # Filter true_values for the current room
        room_true_values = [value for value, r_str in zip(all_true_values, room_data_str) if r_str == room_str]
        
        # Calculate the histogram for the current room
        counts, _ = np.histogram(room_true_values, bins)
        # Plot the histogram using the cumulative counts as the bottom
        plt.bar(bins[:-1], counts, bin_resolution*0.9, bottom=cumulative_counts, label=room_names[i], edgecolor='black')
        # Update the cumulative counts
        cumulative_counts += counts
    # training_data_volume = {'bins': bins[:-1], 'counts': cumulative_counts}
    # with open('training_data_volume.pickle', 'wb') as f:
    #     pickle.dump(training_data_volume, f)

    plt.title(r'Stacked Histogram of True $T_{0-15}$ Values for Each Room', fontsize=26)
    plt.xlabel(r'True $T_{0-15}$ (seconds)', fontsize=26)
    plt.ylabel('Frequency (Hz)', fontsize=26)
    plt.xticks(np.arange(0.1, 1.05, 0.05), rotation=90)
    plt.tick_params(axis='y', labelsize=23)  
    plt.tick_params(axis='x', labelsize=23)
    plt.ylim([0,8000])
    plt.legend(fontsize=21)
    plt.tight_layout()

    if save:
        if not save_base_dir:
            os.makedirs(save_base_dir)
        save_path = os.path.join(save_base_dir, "rooms_true_t60_stacked_histogram.png")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_kde_distribution_for_rooms(data, room_data, save=False, save_base_dir=''):
    # Directly define the room names inside the function
    room_names = ["HL00W", "HL01W", "HL02WL", "HL03W", "HL05W", "HL06W", "HL08W"]
    
    # Convert room identifiers to strings to ensure they are hashable
    room_data_str = np.array([str(room) for room in room_data])
    unique_rooms_str = np.unique(room_data_str)
    
    plt.figure(figsize=(10, 6))
    
    # Initialize data structure to hold values for KDE plot
    values_for_kde = {room: [] for room in unique_rooms_str}
    
    # Populate the data structure with values
    for true_values, _ in data:
        for value, room_str in zip(true_values, room_data_str):
            # Ensure value is treated as a scalar
            value = value.item() if isinstance(value, np.ndarray) and value.size == 1 else value
            values_for_kde[room_str].append(value)
    
    # Plotting KDE for each room
    for i,room_str in enumerate(unique_rooms_str):
        values = values_for_kde[room_str]
        sns.kdeplot(values, bw_adjust=0.5, fill=True, alpha=0.5, label=f'Room {room_names[i]}')
    
    if DT_analysis:
        title = 'Probability Density of True DT10 Values for Each Room'
        x_label = 'True DT10 (seconds)'
    else:
        title = r'Probability Density of True $T_{0-15}$ Values for Each Room'
        x_label = r'True $T_{0-15}$ (seconds)'
    
    plt.title(title,fontsize=24)
    plt.xlabel(x_label,fontsize=24)
    plt.ylabel('Density',fontsize=24)
    plt.tick_params(axis='y', labelsize=24)  
    plt.tick_params(axis='x', labelsize=24) 
    plt.legend(fontsize=24)
    plt.tight_layout()
    
    if save:
        if not os.path.exists(save_base_dir):
            os.makedirs(save_base_dir)
        save_path = os.path.join(save_base_dir, "rooms_true_t60_probability_distribution.png")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_error_boxplot(combined_data_T, bin_resolution=0.01, save=False, connect_medians=True, mae=False, filename='error_boxplot.png'):
    all_true_values = []
    all_pred_values = []
    all_errors = []

    for room_data in combined_data_T:
        for true_values, pred_values in room_data:
            all_true_values.extend(true_values)
            all_pred_values.extend(pred_values)
            errors = [pred - true for true, pred in zip(true_values, pred_values)]
            if not mae:
                all_errors.extend(np.array(errors)**2)
            else:
                all_errors.extend((np.array(errors)**2)**0.5)

    # max_t60 = max(all_true_values)
    # min_t60 = min(all_true_values)
    max_t60 = 1
    min_t60 = 0
    bins = np.arange(min_t60, max_t60 + bin_resolution, bin_resolution)
    binned_errors = [[] for _ in range(len(bins)-1)]

    for true_value, error in zip(all_true_values, all_errors):
        bin_index = np.digitize(true_value, bins) - 1
        if 0 <= bin_index < len(binned_errors):
            binned_errors[bin_index].append(error)

    plt.figure(figsize=(12, 6))
    plt.boxplot(binned_errors, positions=bins[:-1] + bin_resolution/2, widths=bin_resolution * 0.8, showfliers=False)

    # Adjust tick positions and labels to match bin_resolution without repeating
    tick_positions = bins[:-1] + bin_resolution / 2
    # tick_p = bins[:-1] + bin_resolution 
    tick_labels = [f"{b:.02f}" for b in tick_positions]
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=90)

    plt.grid(axis='y')
    plt.xlabel('True T60 [s]')
    plt.ylabel('Error [s]')
    plt.title('Boxplot of Prediction Errors Binned by True T60 Values')

    if connect_medians:
        medians = [np.median(err) if err else 0 for err in binned_errors]
        plt.plot(tick_positions, medians, 'r-o', markersize=3, linestyle='dashdot')

    if save:
        if not os.path.exists(save_base_dir):
            os.makedirs(save_base_dir)
        save_path = os.path.join(save_base_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_percentage_boxplot(combined_data_T, bin_resolution=0.01, save=False, connect_medians=True, filename='error_percentage_boxplot_JND.png'):
    # Initialize lists to store all true values, predicted values, and percentage errors
    all_true_values = []
    all_pred_values = []
    all_adjusted_percentage_errors = []

    # Process each room's data
    for room_data in combined_data_T:
        for true_values, pred_values in room_data:
            all_true_values.extend(true_values)
            all_pred_values.extend(pred_values)
            # Calculate and adjust the percentage error for each prediction
            adjusted_percentage_errors = [((abs(pred - true) / true )* 100) if (abs(pred - true) / true)*100 > 5 else 0 for true, pred in zip(true_values, pred_values)]
            all_adjusted_percentage_errors.extend(np.abs(adjusted_percentage_errors))

    # Determine the bins for true RT values
    max_t60 = max(all_true_values)
    min_t60 = min(all_true_values)
    bins = np.arange(0.1, max_t60 + bin_resolution, bin_resolution)
    binned_adjusted_percentage_errors = [[] for _ in range(len(bins)-1)]

    # Sort adjusted percentage errors into bins
    for true_value, adjusted_percentage_error in zip(all_true_values, all_adjusted_percentage_errors):
        bin_index = np.digitize(true_value, bins) - 1  # Find the bin index for each true_value
        if 0 <= bin_index < len(binned_adjusted_percentage_errors):
            binned_adjusted_percentage_errors[bin_index].append(adjusted_percentage_error)


    # error_data = {'bins': bins[:-1], 'adjusted_percentage_errors': binned_adjusted_percentage_errors}
    # with open('error_data.pickle', 'wb') as f:
    #     pickle.dump(error_data, f)
    # Plotting
    plt.figure(figsize=(12, 6))
    # Create the box plot for binned adjusted percentage errors
    plt.boxplot(binned_adjusted_percentage_errors, positions=bins[:-1] + bin_resolution/2, widths=bin_resolution * 0.8, showfliers=False)

    # Set x-axis limits to start closer to the first bin
    plt.xlim(left=bins[0], right=bins[-1])

    # Adjust x-ticks
    # Calculate a reasonable number of bins to skip based on the total number of bins
    skip = max(1, int(len(bins) / 20))
    xtick_positions = bins[::skip] + bin_resolution/2
    xtick_labels = [f"{b:.2f}" for b in bins[::skip]]
    plt.xticks(ticks=xtick_positions, labels=xtick_labels, rotation=90)

    plt.grid(axis='y')
    plt.tick_params(axis='y', labelsize=19)  
    plt.tick_params(axis='x', labelsize=19) 
    plt.xlabel(r'True $T_{0-15}$ [s]', fontsize=21)
    plt.ylabel('Absolute Error [%]', fontsize=21)
    plt.title(r'Absolute Prediction Error Percentage Binned by True $T_{0-15}$', fontsize=21)


    if connect_medians:
        medians = [np.median(adjusted_percentage_err) if adjusted_percentage_err else 0 for adjusted_percentage_err in binned_adjusted_percentage_errors]
        plt.plot(bins[:-1] + bin_resolution/2, medians, 'r-o', markersize=3, linestyle='dashdot')  # Small red dots and dashed line

    # Save or show the plot
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





def plot_pearson_correlation(combined_data_T, bin_resolution=0.1, save=False, save_base_dir='', filename='pearson_correlation.png'):
    # Initialize lists to store all true values and predicted values
    from scipy.stats import pearsonr
    all_true_values = []
    all_pred_values = []

    # Assuming combined_data_T is a list of lists, each inner list corresponding to a room
    # and containing tuples for each location (true_values, pred_values)
    for room_data in combined_data_T:
        for true_values, pred_values in room_data:
            all_true_values.extend(true_values)
            all_pred_values.extend(pred_values)

    # Determine the bins for true T60 values
    max_t60 = max(all_true_values)
    min_t60 = min(all_true_values)
    bins = np.arange(min_t60, max_t60 + bin_resolution, bin_resolution)
    pearson_correlations = []

    # Calculate Pearson correlation coefficient for each bin
    for bin_start in bins[:-1]:
        bin_end = bin_start + bin_resolution
        # Filter the values that fall into the current bin
        indices = [i for i, value in enumerate(all_true_values) if bin_start <= value < bin_end]
        bin_true_values = [all_true_values[i] for i in indices]
        bin_pred_values = [all_pred_values[i] for i in indices]

        # Calculate the Pearson correlation coefficient for the bin, if there are enough values
        if len(bin_true_values) > 1:
            rho, _ = pearsonr(bin_true_values, bin_pred_values)
            pearson_correlations.append(rho)
        else:
            pearson_correlations.append(None)  # Not enough data to calculate Pearson correlation

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(bins[:-1], pearson_correlations, width=bin_resolution * 0.8, align='edge')
    plt.xticks(ticks=bins, labels=[f"{b:.1f}" for b in bins], rotation=90)
    plt.grid(axis='y')
    plt.xlabel('True T60 [s]')
    plt.ylabel('Pearson Correlation Coefficient (ρ)')
    plt.title('Pearson Correlation Coefficient Binned by True T60 Values')

    # Save or show the plot
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


# def manual_pearson_correlation_coefficient(true_values, pred_values):
#     """
#     Manually calculate the Pearson correlation coefficient.
#     """
#     mean_true = np.mean(true_values)
#     mean_pred = np.mean(pred_values)
    
#     numerator = np.sum((true_values - mean_true) * (pred_values - mean_pred))
#     denominator = np.sqrt(np.sum((true_values - mean_true)**2) * np.sum((pred_values - mean_pred)**2))
    
#     return numerator / denominator if denominator != 0 else 0

# def plot_pearson_correlation(combined_data_T, bin_resolution=0.1, save=False, filename='pearson_correlation.png'):
#     all_true_values = []
#     all_pred_values = []

#     for room_data in combined_data_T:
#         for true_values, pred_values in room_data:
#             all_true_values.extend(true_values)
#             all_pred_values.extend(pred_values)

#     max_t60 = max(all_true_values)
#     min_t60 = min(all_true_values)
#     bins = np.arange(min_t60, max_t60 + bin_resolution, bin_resolution)
#     pearson_correlations = []

#     for bin_start in bins[:-1]:
#         bin_end = bin_start + bin_resolution
#         indices = [i for i, value in enumerate(all_true_values) if bin_start <= value < bin_end]
#         bin_true_values = [all_true_values[i] for i in indices]
#         bin_pred_values = [all_pred_values[i] for i in indices]

#         if len(bin_true_values) > 1:
#             rho = manual_pearson_correlation_coefficient(np.array(bin_true_values), np.array(bin_pred_values))
#             pearson_correlations.append(rho)
#         else:
#             pearson_correlations.append(None)

#     plt.figure(figsize=(12, 6))
#     plt.bar(bins[:-1], pearson_correlations, width=bin_resolution * 0.8, align='edge')
#     plt.xticks(ticks=bins, labels=[f"{b:.1f}" for b in bins], rotation=90)
#     plt.grid(axis='y')
#     plt.xlabel('True T60 [s]')
#     plt.ylabel('Pearson Correlation Coefficient (ρ)')
#     plt.title('Pearson Correlation Coefficient Binned by True T60 Values')

#     if save:
#         if not save_base_dir:
#             raise ValueError("Save directory not provided.")
#         if not os.path.exists(save_base_dir):
#             os.makedirs(save_base_dir)
#         save_path = os.path.join(save_base_dir, filename)
#         plt.savefig(save_path, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()
        

from scipy.stats import spearmanr
def plot_spearman_correlation(combined_data_T, bin_resolution=0.1, save=False, save_base_dir='', filename='spearman_correlation.png'):
    all_true_values = []
    all_pred_values = []

    # Collect all true and predicted values
    for room_data in combined_data_T:
        for true_values, pred_values in room_data:
            all_true_values.extend(true_values)
            all_pred_values.extend(pred_values)

    # Determine bins for true T60 values
    max_t60 = max(all_true_values)
    min_t60 = min(all_true_values)
    bins = np.arange(min_t60, max_t60 + bin_resolution, bin_resolution)
    spearman_correlations = []

    # Calculate Spearman correlation coefficient for each bin
    for bin_start in bins[:-1]:
        bin_end = bin_start + bin_resolution
        indices = [i for i, value in enumerate(all_true_values) if bin_start <= value < bin_end]
        bin_true_values = [all_true_values[i] for i in indices]
        bin_pred_values = [all_pred_values[i] for i in indices]

        # Calculate Spearman correlation if there are enough values
        if len(bin_true_values) > 1:
            rho, _ = spearmanr(bin_true_values, bin_pred_values)
            spearman_correlations.append(rho)
        else:
            spearman_correlations.append(None)  # Not enough data for correlation

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(bins[:-1], spearman_correlations, width=bin_resolution * 0.8, align='edge')
    plt.xticks(ticks=bins, labels=[f"{b:.1f}" for b in bins], rotation=90)
    plt.grid(axis='y')
    plt.xlabel('True T60 [s]')
    plt.ylabel('Spearman Correlation Coefficient (ρ)')
    plt.title('Spearman Correlation Coefficient Binned by True T60 Values')

    # Save or show plot
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




##########################################################################################################################################################
position_analysis =False
edc_loss_analysis = False  ####### TRUE if we want to analyse EDC else T60/DT10 analysis
DT_analysis = False  ####### TRUE if we want to analyse DT10 else T60 analysis
Training_data = False ####### TRUE if we want to analyse Training data, only few plots work with this
save_bool=False #turn this to true in order to save the plots

####################################################################################################################################################
################################################################################################################################################################
# Base directory and parameters

if position_analysis:
    base_logdir = '/home/prsh7458/work/R2D/test_runs_position'
else: 
    # base_logdir = '/home/prsh7458/work/R2D/test_runs'
    base_logdir = '/home/prsh7458/work/R2D/test_runs/others/latest_hyperparameter_trial5'

if not edc_loss_analysis:
    tag = 'Loss/total_loss_test'  
else:
    tag = 'Loss/total_loss_test_edc'

# Rooms and locations

if not Training_data:
    rooms = ["HL00W", "HL01W", "HL02WL", "HL02WP", "HL03W", "HL04W", "HL05W", "HL06W", "HL08W"] 
    # rooms = ["HL02WP","HL04W"] 
    room_locations = ["BC", "FC", "FR", "SiL", "SiR"]


    # a = 0
    # b = 4
    # rooms  = rooms[a:b]
    # room_locations  = room_locations[b:b+1]
else:

    ##########  FOR TRAINING  #############
    base_logdir = '/home/prsh7458/work/R2D/test_runs'
    rooms = ["Training"] 
    room_locations = ["all"]
    room_data = read_mat_file("/home/prsh7458/Desktop/scratch4/databackup/98000 train and 45000 test 32000 length/matlab_data/room_data.mat", 'allroomData') 
    ############################

# rooms = ["HL00W"] 
# room_locations = ["BC"]
# room_locations = ["BC"]

# Iterate over each room



################################################################################################################################################################

####################################################################################################################################################################


all_rooms_data = []
all_rooms_data_DT = []
for room in rooms:
    combined_data = []  # To store data from all locations in the room
    combined_data_DT_loss = []  # To store data from all locations in the room
    combined_true_values = []  # To store true T60 values
    combined_pred_values = []  # To store predicted T60 values
    combined_data_T = []  # To store predicted T60 values
    combined_data_DT = []  # To store predicted T60 values


    # Iterate over each location within the room
    for loc in room_locations:
        logdir = f'{base_logdir}/R2DNet_{room}_{loc}'
        data = extract_tb_data(logdir, tag)
        data_dt_loss = extract_tb_data(logdir, 'Loss/total_loss_test_DT')
        true_values = extract_tb_data(logdir, 'Loss/T_true')
        pred_values = extract_tb_data(logdir, 'Loss/T_pred')
        DT_true_values = extract_tb_data(logdir, 'Loss/DT_true')
        DT_pred_values = extract_tb_data(logdir, 'Loss/DT_pred')
        combined_data_T.append((true_values, pred_values))
        combined_data_DT.append((DT_true_values, DT_pred_values))
        # print(combined_data_T)

        if data:
            combined_data.append(data)
        if data_dt_loss:
            combined_data_DT_loss.append(data_dt_loss)


    all_rooms_data.append(combined_data_T)
    all_rooms_data_DT.append(combined_data_DT)



    if DT_analysis:

        combined_data = combined_data_DT_loss
        combined_data_T = combined_data_DT
        all_rooms_data = all_rooms_data_DT


    # # # # Check if data was found for the room
    # if combined_data:
    #     # plot_violin(combined_data, room, room_locations,mae= not edc_loss_analysis,save=save_bool)
    #     # plot_scatter(combined_data, room, room_locations,save=save_bool)
    #     # # plot_histograms(combined_data, room, room_locations,save=save_bool)
    #     plot_boxplot(combined_data, room, room_locations ,save=save_bool,mae= not edc_loss_analysis)
    #     # plot_beeswarm(combined_data, room, room_locations,save=save_bool)
    #     # plot_cdf(combined_data, room, room_locations,save=save_bool)
    #     # plot_raincloud(combined_data, room, room_locations,save=save_bool)
    # else:
    #     print(f"No data found for Room {room}")

    # # # Plot Confusion Matrix if T60 data found
    # if combined_data_T:
    #     # plot_error_histogram_for_room(combined_data_T, room,room_locations, save=save_bool)
    #     plot_confusion_matrix_for_room(combined_data_T, room, room_locations, bin_resolution=0.01, save=save_bool)
    #     # plot_histogram_true_values_for_room(combined_data_T, room, room_locations, save=save_bool)                # Extract T60 data
    #     # plot_stacked_histogram_for_rooms(combined_data_T, room_data, save=save_bool) ###ONLY USE FOR TRAINING ONE
    #     # plot_kde_distribution_for_rooms(combined_data_T, room_data, save=save_bool) ###ONLY USE FOR TRAINING ONE
    #     # plot_bland_altman_for_room(combined_data_T, room, room_locations, save=save_bool) 
    # else:
    #     print(f"No T60 data found for Room {room}")
        


#################################### some aggragated aplot_histogram_true_values_for_roomnalysis ################################################################################

# plot_combined_confusion_matrix(all_rooms_data, bin_resolution=0.05, save=save_bool)
# plot_scatter_combined(all_rooms_data, save=save_bool)
# plot_scatter_combined_with_approx_boxplot(all_rooms_data, save=save_bool)
# plot_error_boxplot(all_rooms_data, bin_resolution=0.04, save=save_bool,mae = True)
plot_error_percentage_boxplot(all_rooms_data, bin_resolution=0.05, save=save_bool)