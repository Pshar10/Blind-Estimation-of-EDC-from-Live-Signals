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

"""This script is to do aggregated analysis of all the rooms combined"""


global edc_loss_analysis


######################################################################################################################################################
# Base directory and parameters

pos_bool = False

if pos_bool:
    base_logdir = '/home/prsh7458/work/R2D/test_runs_position'
else:
    base_logdir = '/home/prsh7458/work/R2D/test_runs/others/latest_hyperparameter_trial5'
    # base_logdir = '/home/prsh7458/work/R2D/test_runs'   

save_base_dir = '/home/prsh7458/work/R2D/Loss_Graphs/loss_loc'  # Directory to save plots


edc_loss_analysis = False  ####### TRUE if we want to analyse EDC else T60 analysis
DT_analysis = False


if not edc_loss_analysis:
    if DT_analysis: 
        tag = 'Loss/total_loss_test_DT' 
    else:
        tag = 'Loss/total_loss_test' 
else:
    tag = 'Loss/total_loss_test_edc'


# rooms = ["HL00W", "HL01W", "HL02WL", "HL02WP", "HL03W", "HL04W", "HL05W", "HL06W", "HL08W"] 
# room_locations = ["BC", "FC", "FR", "SiL", "SiR"]


rooms = ["HL00W", "HL01W", "HL02WL", "HL02WP", "HL03W", "HL04W", "HL05W", "HL06W", "HL08W"] 
room_locations = ["BC", "FC", "FR", "SiL", "SiR"]
# room_locations = room_locations[:1]

# rooms = ["Training"] 
# room_locations = ["all"]

##########################################################################################################################################################3




def extract_tb_data(logdir, tag):
    ea = event_accumulator.EventAccumulator(logdir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    if tag in ea.scalars.Keys():
        scalar_events = ea.scalars.Items(tag)
        values = [s.value for s in scalar_events]
        return values
    else:
        return []


def plot_aggregated_violin(data, room_locations, save=False):
    plt.figure(figsize=(10, 6))
    loss_data = []
    for loc in room_locations:
        for room in rooms:
            room_data = data[loc][room]
            for value in room_data:
                loss_data.append({'Location': loc, 'Loss': value})
    df = pd.DataFrame(loss_data)
    sns.violinplot(x='Location', y='Loss', data=df, palette="muted")
    sns.stripplot(x='Location', y='Loss', data=df, color='black', alpha=0.000002, jitter=True)
    plt.title('Aggregated Loss Distribution Across All Rooms for Each Location')
    plt.xlabel('Location')
    plt.ylabel('Loss Values')
    plt.grid(True, linestyle='--', alpha=0.5)
    if save:
        plt.savefig(f'{save_base_dir}/aggregated_violin_plot.png')
        plt.close()
    else:
        plt.show()

import math  # Import the math module to use math.isinf()

def plot_aggregated_boxplot(data, room_locations, save=False, mae = False):
    plt.figure(figsize=(10, 6))
    loss_data = []
    for loc in room_locations:
        for room in rooms:
            room_data = data[loc][room]
            for value in room_data:
                # loss_data.append({'Location': loc, 'Loss': value**2}) ####DO CHANGE OR REMOVE THE SQUARE AFTER VALUE
                if math.isinf(value):
                    continue
                if mae:
                    loss_data.append({'Location': loc, 'Loss': value**0.5}) 
                    
                else:   
                    
                    loss_data.append({'Location': loc, 'Loss': value}) 
                    

    df = pd.DataFrame(loss_data)
    sns.boxplot(x='Location', y='Loss', data=df, palette='muted', fliersize=0)
    plt.title('Aggregated Loss Distribution Across All Rooms for Each Location')
    plt.xlabel('Location')
    plt.ylabel('Loss Values')
    plt.grid(True, linestyle='--', alpha=0.5)
    if edc_loss_analysis:
        plt.ylim([0, 20])
    else: 
        plt.ylim([0, 1])
    if save:
        plt.savefig(f'{save_base_dir}/aggregated_boxplot.png')
        plt.close()
    else:
        plt.show()

def plot_aggregated_histogram(data, room_locations, save=False):
    plt.figure(figsize=(12, 8))
    bins = np.arange(0, 6, 2)  # Bins: 0-2, 2-4, ..., 16-18
    width = 1 / (len(room_locations) + 1)  # Width of each bar
    colors = plt.cm.viridis(np.linspace(0, 1, len(room_locations)))  # Color palette
    for i, bin_start in enumerate(bins[:-1]):
        bin_end = bins[i + 1]
        for j, loc in enumerate(room_locations):
            loc_data = np.concatenate([data[loc][room] for room in rooms])
            count = np.sum((loc_data >= bin_start) & (loc_data < bin_end))
            plt.bar(i + j * width, count, width, alpha=0.7, color=colors[j], label=loc if i == 0 else "")
    plt.title('Aggregated Test Loss Across All Rooms for Each Location')
    plt.xlabel('Loss Value Bins')
    plt.ylabel('Frequency')
    plt.xticks(ticks=np.arange(len(bins) - 1) + width / 2, labels=[f'{int(b)}-{int(b) + 2}' for b in bins[:-1]])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    if save:
        plt.savefig(f'{save_base_dir}/aggregated_histogram.png')
        plt.close()
    else:
        plt.show()


def plot_histogram_true_values_for_room(data, room, room_locations, save=False):
    # for i, (true_values, _) in enumerate(data):  # We're only interested in true values
    #     plt.figure(figsize=(10, 6))
        loss_data = []
        for loc in room_locations:
            for room in rooms:
                room_data = data[loc][room]
                for value in room_data:
                    loss_data.append(value)
        plt.hist(loss_data, bins=100, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Histogram of True T60 ')
        plt.xlabel('True T60 (seconds)')
        plt.ylabel('Frequency')
        plt.tight_layout()

        if save:
            # save_path = os.path.join(save_base_dir, f"{room}_{room_locations[i]}_true_histogram.png")
            # plt.savefig(save_path)
            plt.close()
        else:
            plt.show()





def plot_aggregated_scatter(data, room_locations, save=False):
    plt.figure(figsize=(12, 8))
    loss_data = []
    for loc in room_locations:
        for room in rooms:
            room_data = data[loc][room]
            for value in room_data:
                loss_data.append({'Location': loc, 'Loss': value})
    df = pd.DataFrame(loss_data)
    sns.stripplot(x='Location', y='Loss', data=df, jitter=0.2, palette='Set2', alpha=0.5, size=4)
    plt.title('Aggregated Loss Values Across All Rooms for Each Location')
    plt.xlabel('Location')
    plt.ylabel('Loss Values')
    plt.grid(True, linestyle='--', alpha=0.5)
    if save:
        plt.savefig(f'{save_base_dir}/aggregated_scatter_plot.png')
        plt.close()
    else:
        plt.show()


def generate_stats_table(data, room_locations, rooms, mae=False):
    # Creating a DataFrame to hold statistical data
    stats_columns = ['Location', 'Room', 'Mean', 'Root Mean Square', 'Median', 'Standard Deviation', 'Min', 'Max']
    stats_df = pd.DataFrame(columns=stats_columns)

    for loc in room_locations:
        for room in rooms:
            room_data = np.array(data[loc][room])
            
            # Filter out non-finite values from room_data
            room_data = room_data[np.isfinite(room_data)]

            # Apply square root if mae is True, directly to room_data
            if mae:
                room_data = np.sqrt(room_data)

            if room_data.size > 0:
                stats = {
                    'Location': loc,
                    'Room': room,
                    'Mean': np.round(np.mean(room_data), 3),
                    'Root Mean Square': np.round(np.sqrt(np.mean(room_data**2)), 3),
                    'Median': np.round(np.median(room_data), 3),
                    'Standard Deviation': np.round(np.std(room_data), 3),
                    'Min': np.round(np.min(room_data), 3),
                    'Max': np.round(np.max(room_data), 3)
                }
                stats_df = stats_df.append(stats, ignore_index=True)

    return stats_df

def location_analysis(file_path):
    
    df = pd.read_excel(file_path)

    # Pivot the table to get mean values for each room in each location
    pivot_df = df.pivot(index='Room', columns='Location', values='Mean')

    # Convert pivot table back to DataFrame for easier plotting
    mean_values = pivot_df.reset_index()

    # Extract room labels and location labels
    rooms = mean_values['Room'].tolist()
    locations = mean_values.columns[1:].tolist()  # Excluding the 'Room' column

    # Initialize figure and axes
    fig, axs = plt.subplots(2, 3, figsize=(20, 14), dpi=100)
    axs = axs.flatten()

    # Determine global y-axis limits
    global_min_y = np.inf
    global_max_y = -np.inf

    for location in locations:
        y_values = mean_values[location]
        global_min_y = min(global_min_y, y_values.min())
        global_max_y = max(global_max_y, y_values.max())

    # Add some padding to global min and max
    global_min_y *= 0.9
    global_max_y *= 1.1

    plt.subplots_adjust(
        left=0.1,  # Increased to accommodate the common y-axis label
        bottom=0.093, 
        right=0.987, 
        top=0.936, 
        wspace=0.08, 
        hspace=0.367
    )

    for i, location in enumerate(locations):
        y_values = mean_values[location]
        x_values = np.arange(len(rooms))
        
        # Fit a simple linear polynomial for demonstration
        coefs = Polynomial.fit(x_values, y_values, 10).convert().coef
        poly_fit = np.polyval(coefs[::-1], x_values)
        
        axs[i].scatter(x_values, y_values, color='blue', label='MAE', s=100)
        axs[i].plot(x_values, poly_fit, 'r:', lw=2)
        
        axs[i].set_ylim(global_min_y, global_max_y)
        axs[i].set_title(location, fontsize=15, pad=15)
        axs[i].set_xticks(x_values)
        axs[i].set_xticklabels(rooms, rotation=45, fontsize=10)
        axs[i].tick_params(axis='y', labelsize=10)
        axs[i].legend(prop={'size': 8})

    # Remove individual y-axis labels
    for ax in axs:
        ax.set_ylabel('')

    # Set a common y-axis label
    fig.text(0.04, 0.5, 'Mean L1 Loss for different rooms', va='center', rotation='vertical', fontsize=15)

    # Remove unused axes if necessary
    if len(axs) > len(locations):
        fig.delaxes(axs[-1])

    plt.show()





# Initialize a dictionary to store aggregated data for each location and room
aggregated_data = {loc: {room: [] for room in rooms} for loc in room_locations}
aggregated_data_true = {loc: {room: [] for room in rooms} for loc in room_locations}

# Aggregate data for each location and room
for room in rooms:
    for loc in room_locations:
        logdir = f'{base_logdir}/R2DNet_{room}_{loc}'
        data = extract_tb_data(logdir, tag)
        true_values = extract_tb_data(logdir, 'Loss/T_true')
        if data and true_values:
            aggregated_data[loc][room].extend(data)
            aggregated_data_true[loc][room].extend(true_values)
        else: print("No data found")

def plot_stats_table(stats_table):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.axis('off')  # Turn off the axis

    table_data = []
    for index, row in stats_table.iterrows():
        table_data.append([row['Location'], row['Room'], row['Mean'], row['Root Mean Square'], row['Median'], row['Standard Deviation'], row['Min'], row['Max']])

    table = ax.table(cellText=table_data, colLabels=stats_table.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(True)
    # table.set_fontsize(10)
    table.scale(1, 2.5)  # Adjust the table size as needed

    plt.show()

def style_stats_table(stats_table):
    # Apply styling to the DataFrame
    styled_table = (stats_table.style
        .set_table_styles([{
            'selector': '',
            'props': [
                ('border-collapse', 'collapse'),
                ('border', '2px solid black'),
                ('font-size', '14px'),  # Adjust the font size
            ]
        }])
    )

    return styled_table



##########################################################################################################


save = True



#################################################################################################################


# plot_aggregated_violin(aggregated_data, room_locations,save)
# plot_aggregated_boxplot(aggregated_data, room_locations,save,mae=not edc_loss_analysis)
# plot_aggregated_histogram(aggregated_data, room_locations,save)
# plot_histogram_true_values_for_room(aggregated_data_true,room, room_locations,save)  #

# plot_aggregated_scatter(aggregated_data, room_locations,save)


if not edc_loss_analysis:
    table_name = f'Stats_table_{"DT10" if DT_analysis else "T60"}'
else:
    table_name = f'Stats_table_EDC'

stats_table = generate_stats_table(aggregated_data, room_locations, rooms,mae = not edc_loss_analysis)

# # # # # Save the statistics table as a text file
stats_table.to_csv(table_name + '.txt', index=False, sep='\t')



# # # # Example usage
styled_table = style_stats_table(stats_table)
styled_table.to_excel(table_name +'.xlsx', index=False)

# plot_stats_table(stats_table)


# location_analysis(table_name +'.xlsx')



