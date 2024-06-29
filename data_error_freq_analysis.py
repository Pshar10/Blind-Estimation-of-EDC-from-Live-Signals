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

"""This script shows the trend is T0-15 error estimation in different time bins and T0-15 values present in training dataset together"""



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



#########################################################################################################################################################
position_analysis =False
edc_loss_analysis = False  ####### TRUE if we want to analyse EDC else T60/DT10 analysis
DT_analysis = False  ####### TRUE if we want to analyse DT10 else T60 analysis
Training_data = True ####### TRUE if we want to analyse Training data, only few plots work with this
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





all_rooms_data = []
all_rooms_data_error = []
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
        true_values = extract_tb_data(logdir, 'Loss/T_true')
        pred_values = extract_tb_data(logdir, 'Loss/T_pred')
        combined_data_T.append((true_values, pred_values))
    all_rooms_data.append(combined_data_T)



rooms_error = ["HL00W", "HL01W", "HL02WL", "HL02WP", "HL03W", "HL04W", "HL05W", "HL06W", "HL08W"] 
room_locations_error = ["BC", "FC", "FR", "SiL", "SiR"]

all_rooms_data_error = []
for room in rooms_error:
    combined_data = []  # To store data from all locations in the room
    combined_data_DT_loss = []  # To store data from all locations in the room
    combined_true_values = []  # To store true T60 values
    combined_pred_values = []  # To store predicted T60 values
    combined_data_T = []  # To store predicted T60 values
    combined_data_DT = []  # To store predicted T60 values
    combined_data_error = []

    # Iterate over each location within the room
    for loc in room_locations_error:
        
        log_error = '/home/prsh7458/work/R2D/test_runs/others/latest_hyperparameter_trial5'
        logdir = f'{log_error}/R2DNet_{room}_{loc}'
        true_values_error = extract_tb_data(logdir, 'Loss/T_true')
        pred_values_error = extract_tb_data(logdir, 'Loss/T_pred')
        combined_data_error.append((true_values_error, pred_values_error))
    all_rooms_data_error.append(combined_data_error)


def plot_stacked_histogram_with_error_median(room_data, combined_data_T, combined_data_error, bin_resolution=0.01, save=False, save_base_dir=''):
    # Initialize lists to store all true values and errors
    all_true_values = []
    all_true_values_error = []
    all_pred_values_error = []
    all_adjusted_errors = []

    # Process each room's data
    for room_data in combined_data_T:
        for true_values, __ in room_data:
            all_true_values.extend(true_values)

    for room_data in combined_data_error:
        for true_values, pred_values in room_data:
            all_true_values_error.extend(true_values)
            all_pred_values_error.extend(pred_values)
            adjusted_errors = [((abs(pred - true) / true)) if (abs(pred - true) / true)*100 > 5 else 0 for true, pred in zip(true_values, pred_values)]
            all_adjusted_errors.extend(np.abs(adjusted_errors))
            
    # Determine the bins for true T60 values
    max_t60 = max(all_true_values)
    min_t60 = min(all_true_values)
    bins = np.arange(min_t60, max_t60 + bin_resolution, bin_resolution)
    bin_centers = bins[:-1] + np.diff(bins)/2
    
    plt.figure(figsize=(15, 8))
    
    # Normalize histogram
    counts, _ = np.histogram(all_true_values, bins)
    normalized_counts = counts / np.max(counts)
    
    # Bar plot for normalized histogram
    # plt.bar(bin_centers, normalized_counts, bin_resolution * 0.9, edgecolor='black', alpha=0.9)

    # Adjusted errors for boxplot
    binned_adjusted_errors = [[] for _ in range(len(bins)-1)]
    for true_value, adjusted_error in zip(all_true_values_error, all_adjusted_errors):
        bin_index = np.digitize(true_value, bins) - 1
        if 0 <= bin_index < len(binned_adjusted_errors):
            binned_adjusted_errors[bin_index].append(adjusted_error)

    # Boxplot on top of histogram
    plt.boxplot(binned_adjusted_errors, positions=bin_centers, widths=bin_resolution * 0.8, showfliers=False, medianprops=dict(color="red"), patch_artist=True)
    medians = [np.median(errors) if errors else 0 for errors in binned_adjusted_errors]
    # plt.plot(bin_centers, medians, 'r-')

    plt.title('Absolute Error Boxplot', fontsize=24)
    plt.xlabel('True T60 (seconds)', fontsize=24)
    plt.ylabel('Binned Error', fontsize=24)
    plt.xticks(bin_centers, labels=[f'{b:.2f}' for b in bin_centers], rotation=90)
    plt.tick_params(axis='y', labelsize=24)  
    plt.tick_params(axis='x', labelsize=24)
    plt.ylim([0, 2])
    plt.xlim([min(bin_centers) - bin_resolution, 0.9])
    plt.legend(['Absolute Error'], fontsize=24)
    plt.tight_layout()

    if save:
        if not os.path.exists(save_base_dir):
            os.makedirs(save_base_dir)
        save_path = os.path.join(save_base_dir, "combined_histogram_error_boxplot.png")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# def plot_stacked_histogram_with_error_median(room_data, combined_data_T, combined_data_error, bin_resolution=0.01, save=False, save_base_dir=''):
#     all_true_values = []
#     all_true_values_error = []
#     all_pred_values_error = []
#     all_adjusted_percentage_errors = []

#     # Collecting true values and errors
#     for room_data in combined_data_T:
#         for true_values, __ in room_data:
#             all_true_values.extend(true_values)

#     for room_data in combined_data_error:
#         for true_values, pred_values in room_data:
#             all_true_values_error.extend(true_values)
#             all_pred_values_error.extend(pred_values)
#             adjusted_percentage_errors = [(abs(pred - true) / true) * 100 for true, pred in zip(true_values, pred_values)]
#             all_adjusted_percentage_errors.extend(np.abs(adjusted_percentage_errors))

#     # Define bins
#     max_t60 = max(all_true_values)
#     min_t60 = min(all_true_values)
#     bins = np.arange(min_t60, max_t60 + bin_resolution, bin_resolution)

#     # Convert all_true_values to a histogram of values
#     counts, _ = np.histogram(all_true_values, bins)

#     # Calculate median errors for each bin
#     binned_adjusted_percentage_errors = [[] for _ in range(len(bins)-1)]
#     for true_value, adjusted_percentage_error in zip(all_true_values_error, all_adjusted_percentage_errors):
#         bin_index = np.digitize(true_value, bins) - 1
#         if 0 <= bin_index < len(binned_adjusted_percentage_errors):
#             binned_adjusted_percentage_errors[bin_index].append(adjusted_percentage_error)
#     medians = [np.median(errors) if errors else 0 for errors in binned_adjusted_percentage_errors]

#     # Create figure and first axis
#     fig, ax1 = plt.subplots(figsize=(10, 8))
#     ax1.set_xlabel('True T60 (seconds)')
#     ax1.set_ylabel('Frequency', color='tab:blue')
#     ax1.plot(bins[:-1], counts, 'b-')  # Blue line for frequency
#     ax1.tick_params(axis='y', labelcolor='tab:blue')

#     # Create second axis
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Median Error Percentage', color='tab:red')
#     ax2.plot(bins[:-1], medians, 'r-')  # Red line for median error percentage
#     ax2.tick_params(axis='y', labelcolor='tab:red')

#     plt.title('Line Plot of True T60 Values with Error Percentage')
#     plt.xticks(np.arange(min_t60, max_t60, 0.05), rotation=90)
#     fig.tight_layout()

#     if save:
#         if not os.path.exists(save_base_dir):
#             os.makedirs(save_base_dir)
#         save_path = os.path.join(save_base_dir, "combined_lineplot_error_percentage.png")
#         plt.savefig(save_path)
#         plt.close()
#     else:
#         plt.show()

def plot_combined_heatmap(room_data, combined_data_T, combined_data_error, bin_resolution=0.01, save=False, save_base_dir=''):
    all_true_values = []
    all_true_values_error = []
    all_pred_values_error = []
    all_adjusted_percentage_errors = []

    # Collect data
    for room_data in combined_data_T:
        for true_values, __ in room_data:
            all_true_values.extend(true_values)

    for room_data in combined_data_error:
        for true_values, pred_values in room_data:
            all_true_values_error.extend(true_values)
            all_pred_values_error.extend(pred_values)
            adjusted_percentage_errors = [(abs(pred - true) / true) * 100 if true != 0 else 0 for true, pred in zip(true_values, pred_values)]
            all_adjusted_percentage_errors.extend(np.abs(adjusted_percentage_errors))

    # Define bins and calculate histograms
    max_t60 = max(all_true_values)
    min_t60 = min(all_true_values)
    bins = np.arange(min_t60, max_t60 + bin_resolution, bin_resolution)
    counts, _ = np.histogram(all_true_values, bins=bins)

    # Calculate median errors for each bin
    binned_adjusted_percentage_errors = [[] for _ in range(len(bins)-1)]
    for true_value, adjusted_percentage_error in zip(all_true_values_error, all_adjusted_percentage_errors):
        bin_index = np.digitize(true_value, bins) - 1
        if 0 <= bin_index < len(binned_adjusted_percentage_errors):
            binned_adjusted_percentage_errors[bin_index].append(adjusted_percentage_error)
    medians = [np.median(errors) if errors else 0 for errors in binned_adjusted_percentage_errors]

    # Normalize data
    counts_norm = (counts - np.min(counts)) / (np.max(counts) - np.min(counts))
    medians_norm = (medians - np.min(medians)) / (np.max(medians) - np.min(medians))

    # Combine normalized data
    combined_data = np.vstack((counts_norm, medians_norm))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap = ax.imshow(combined_data, cmap='coolwarm', aspect='auto')
    fig.colorbar(heatmap, ax=ax)
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels([f"{bin:.2f}" for bin in bins], rotation=90)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Normalized Frequency', 'Normalized Median Error'])
    ax.set_title('Combined Heatmap of Frequency and Error Percentage')

    if save:
        if not os.path.exists(save_base_dir):
            os.makedirs(save_base_dir)
        save_path = os.path.join(save_base_dir, "combined_heatmap.png")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
plot_stacked_histogram_with_error_median(room_data, all_rooms_data,all_rooms_data_error, bin_resolution=0.05)
# plot_combined_heatmap(room_data, all_rooms_data,all_rooms_data_error, bin_resolution=0.01)

