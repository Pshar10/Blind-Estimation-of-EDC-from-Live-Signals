from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import numpy as np
import os
import warnings
import math
warnings.simplefilter(action='ignore', category=FutureWarning)

"""A script to analyse noise roburstness test plots"""

global edc_loss_analysis
global save_base_dir
global DT_analysis
log_dir = f'/home/prsh7458/work/R2D/test_runs_noise_test'

def extract_tb_data(logdir, tag):
    ea = event_accumulator.EventAccumulator(logdir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    if tag in ea.scalars.Keys():
        scalar_events = ea.scalars.Items(tag)
        values = [s.value for s in scalar_events]
        return values
    else:
        return []
   


position_analysis =False
edc_loss_analysis = False  ####### TRUE if we want to analyse EDC else T60/DT10 analysis
DT_analysis = False  ####### TRUE if we want to analyse DT10 else T60 analysis
Training_data = False ####### TRUE if we want to analyse Training data, only few plots work with this
save_bool=False #turn this to true in order to save the plots


base_logdir = f'/home/prsh7458/work/R2D/test_runs_noise_test'
SNR_levels = [10,20,30,40,50] 
rooms = ["HL00W", "HL01W", "HL02WL", "HL02WP", "HL03W", "HL04W", "HL05W", "HL06W", "HL08W"] 
room_locations = ["BC", "FC", "FR", "SiL", "SiR"]

##########################################################################################################

# room_name = "HL01W"
# Noise_source = "SiR"
# speech_source = "FR"


edc_loss_analysis = False

########################################################################################################


tag = 'Loss/total_loss_test_edc' if edc_loss_analysis else 'Loss/total_loss_test' 

all_SNR_data = []
for SNR_level in SNR_levels :
    all_room_data = []
    for room_idx, room in enumerate(rooms):
        combined_data=[]
        for loc_idx, location in enumerate(room_locations):
            # if location == Noise_source : continue # add for rever noise
            # log_dir_comb =  f'/home/prsh7458/work/R2D/test_runs_noise_test/R2DNet__S1_{room}_{location}_N1_{Noise_source}_SNR_{SNR_level}' #use for reverb noise
            log_dir_comb =  f'/home/prsh7458/work/R2D/test_runs_dry_noise_test/R2DNet__S1_{room}_{location}_SNR_{SNR_level}'
            data = extract_tb_data(log_dir_comb, tag) 
            combined_data.append(data)
        all_room_data.append(combined_data)
    all_SNR_data.append(all_room_data)


# Convert the nested list to a NumPy array
all_SNR_data_array = np.array(all_SNR_data, dtype=object)  # Use dtype=object for mixed-type data or complex structures
print("Shape of all_SNR_data using NumPy:", all_SNR_data_array.shape)


# source_room_locations = [location for location in room_locations if location != Noise_source]

# location_index = source_room_locations.index(speech_source)
# room_index = rooms.index(room_name)

# # Extract mean loss values for the specified room and location across all SNR levels
# mean_losses = []
# for SNR_data in all_SNR_data:
#     # SNR_data[room_index] gives us all location data for the chosen room at the current SNR
#     # We take the location_index, which corresponds to our chosen location (adjusted for skipping)
#     mean_loss = np.mean(SNR_data[room_index][location_index])
#     mean_losses.append(mean_loss)

# # Corresponding SNR levels (should not include skipped or adjusted indices)
# SNR_levels_adjusted = [10, 20, 30, 40, 50]  # Adjust this if your data structure has changed

# # Plotting the results
# plt.figure(figsize=(10, 5))
# plt.plot(SNR_levels_adjusted, mean_losses, marker='o', linestyle='-', color='b')
# plt.title(f'Mean Loss vs. SNR for Room "{rooms[room_index]}", Source_Location "{source_room_locations[location_index]}" , Noise_Location "{Noise_source}"', fontsize=21)
# plt.xlabel('SNR Level (dB)', fontsize=21)
# plt.ylabel('Mean Loss (dB)', fontsize=21)
# plt.ylim([0,10]) if edc_loss_analysis else plt.ylim([0,0.8])
# plt.grid(True)
# plt.xticks((SNR_levels_adjusted))
# plt.tick_params(axis='y', labelsize=19)  
# plt.tick_params(axis='x', labelsize=19) 
# plt.show()



# # Set the aesthetic style of the plots
# sns.set(style="whitegrid")

# plt.figure(figsize=(15, 8))
# for i, room_name in enumerate(rooms):
#     source_room_locations = [location for location in room_locations if location != Noise_source]
#     location_index = source_room_locations.index(speech_source) if speech_source in source_room_locations else -1

#     room_index = rooms.index(room_name)

#     # Extract mean loss values for the specified room and location across all SNR levels
#     mean_losses = []
#     for SNR_data in all_SNR_data:
#         mean_loss = np.mean(SNR_data[room_index][location_index])
#         mean_losses.append(mean_loss)

#     # Plot using Seaborn
#     sns.lineplot(x=SNR_levels, y=mean_losses, marker='o', label=f'{room_name}')

# plt.title('Mean Loss vs. SNR for Various Rooms', fontsize=21)
# plt.xlabel('SNR Level (dB)', fontsize=21)
# plt.ylabel('Mean Loss (dB)', fontsize=21)
# plt.grid(True)
# plt.xticks(SNR_levels)
# plt.ylim([0, 10] if 'edc_loss_analysis' in locals() and edc_loss_analysis else [0, 0.8])
# plt.legend(fontsize=14)
# plt.tick_params(axis='both', labelsize=19)
# plt.show()


data_list = []

# Iterate over SNR levels, rooms, and locations with safe checks
for i, SNR_level in enumerate(SNR_levels):
    for j, room in enumerate(rooms):
        for k, location in enumerate(room_locations):
            # if location == Noise_source or k >= len(all_SNR_data[i][j]):
                # continue

            # Safe access to the loss data
            room_data = all_SNR_data[i][j][k]
            for loss in room_data:
                loss =loss**0.5 if not edc_loss_analysis else loss
                data_list.append({'Room': room, 'SNR Level': SNR_level, 'Loss': loss, 'Location': location})
# Convert list to DataFrame
df = pd.DataFrame(data_list)




###########

plt.figure(figsize=(16, 9))
# sns.boxplot(x='Room', y='Loss', hue='SNR Level', data=df, palette="Set3")  # 'Set3' is a colorful qualitative palette
# sns.violinplot(x='Room', y='Loss', hue='SNR Level', data=df, palette="Set3")
sns.lineplot(x='Room', y='Loss', hue='SNR Level', data=df, palette="bright", marker='o')


titlestr = 'L1 EDC Loss' if edc_loss_analysis else r' $T_{0-15}$ Loss'
plt.title(f'{titlestr} Across different Rooms and SNR Levels', fontsize=21)
plt.xlabel('Room', fontsize=21)
plt.ylabel('Loss (dB)', fontsize=21) if edc_loss_analysis else plt.ylabel('Loss (secs)', fontsize=21) 
plt.legend(title='SNR Level (dB)', loc='upper right', fontsize=21, title_fontsize=21, bbox_to_anchor=(1, 1), ncol=5)

plt.grid(True)
plt.tick_params(axis='both', labelsize=21)
# plt.show()

# # Settings
# locations = df['Location'].unique()
# num_locations = len(locations)
# fig, axes = plt.subplots(num_locations, 1, figsize=(15, num_locations * 3), sharex=True, sharey=True)

# # Plot each location on its own subplot
# for i, location in enumerate(locations):
#     sns.lineplot(ax=axes[i], x='Room', y='Loss', hue='SNR Level', 
#                  data=df[df['Location'] == location], palette="bright", marker='o')
#     axes[i].set_title(f'Location: {location}', fontsize=21)
    
#     # Only show y-axis labels on the first plot
#     if i > 0:  
#         axes[i].set_ylabel('')
    
# # Remove x-ticks for all but the last plot
# for ax in axes[:-1]:
#     ax.tick_params(labelbottom=False)

# # Set common X and Y labels
# fig.text(0.5, 0.04, 'Room', ha='center', va='center', fontsize=21)
# fig.text(0.04, 0.5, 'Loss', va='center', rotation='vertical', fontsize=21)

# # Remove existing legends from subplots
# for ax in axes:
#     ax.get_legend().remove()

# # Draw the plot
# plt.tight_layout()

# # Add a single common legend outside the figure on the right
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, bbox_to_anchor=(1, 0.5), loc='center left', fontsize=19, title='SNR Level', title_fontsize=19)

# # Adjust subplot parameters to fit the figure size and legend
# plt.subplots_adjust(right=0.85)

# # Show the plot
# plt.show()


######################RMSE###############
df['Squared Loss'] = df['Loss'] ** 2

# Calculate mean squared error (MSE) for each combination of 'Room' and 'SNR Level'
grouped = df.groupby(['Room', 'SNR Level'])['Squared Loss'].mean().reset_index()

# Calculate the RMSE by applying the square root to the mean squared errors
grouped['RMSE'] = np.sqrt(grouped['Squared Loss'])

# Plotting the RMSE values
# plt.figure(figsize=(15, 8))
# for room in grouped['Room'].unique():
#     subset = grouped[grouped['Room'] == room]
#     plt.plot(subset['SNR Level'], subset['RMSE'], marker='o', linestyle='-', label=room)

# plt.title('RMSE vs. SNR Levels for Various Rooms', fontsize=21)
# plt.xlabel('SNR Level (dB)', fontsize=21)
# plt.ylabel('RMSE (dB)', fontsize=21)
# plt.legend(title='Room', fontsize=19)
# plt.grid(True)
# plt.xticks(grouped['SNR Level'].unique(), rotation=45)  # Ensuring SNR levels are clearly labeled
# plt.show()
plt.figure(figsize=(16, 9))
sns.lineplot(x='Room', y='RMSE', hue='SNR Level', data=grouped, palette="bright") 

# Plot configuration
titlestr = 'L1 EDC Loss' if edc_loss_analysis else r' $T_{0-15}$ Loss'
plt.title(f'{titlestr} Across different Rooms and SNR Levels', fontsize=21)
plt.xlabel('Room', fontsize=21)
plt.ylabel('RMSE (dB)' if edc_loss_analysis else 'RMSE (secs)', fontsize=21)
plt.legend(title='SNR Level (dB)', loc='upper right', fontsize=21, title_fontsize=21, bbox_to_anchor=(1, 1), ncol=5)
plt.grid(True)
plt.tick_params(axis='both', labelsize=21)
# plt.show()


def set_label_alignment(angle):
    if angle >= np.pi/2 and angle <= 3*np.pi/2:
        return 'right'
    else:
        return 'left'
    


pivot_df = grouped.pivot(index='Room', columns='SNR Level', values='RMSE')

# Prepare the angles for the radar chart
labels = pivot_df.columns
num_vars = len(labels)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Draw one line per room
for i, row in pivot_df.iterrows():
    data = row.tolist()
    data += data[:1]  # Complete the data loop
    ax.plot(angles, data, label=i, linewidth=2.5)  # Add each room's line
    ax.fill(angles, data, alpha=0.1)  # Fill area under the line

# Labels for each point
ax.set_xticks(angles[:-1])
ax.set_xticklabels([f'{int(l)} dB' for l in labels], fontsize=21)  # SNR levels as labels

ax.tick_params(pad=30)  # Increase padding distance as needed
start, stop, step = (0, 12, 2) if edc_loss_analysis else (0, 1.2, 0.2)
r_grids = np.arange(start=start, stop=stop+0, step=step)  # Include stop in range
r_labels = [str(round(grid, 2)) for grid in r_grids]

# Set radial grids without default labels
ax.set_rgrids(r_grids, labels=['']*len(r_grids), angle=angles[0], fontsize=21)  # Empty labels to use custom positioning

# Custom placement of grid labels
for r, label in zip(r_grids, r_labels):
    angle_rad = np.pi / 4  # Diagonal angle, adjust as needed for direction
    ax.text(angle_rad, r, label, horizontalalignment='center', verticalalignment='center', fontsize=21, alpha=0.7)


textstr = 'RMSE EDC Loss (dB)' if edc_loss_analysis else r'$T_{0-15}$ RMSE Loss (secs)'

titlestr = f'Radar Chart showing {textstr} for different SNR levels across different Rooms'
plt.title(f'{titlestr}', fontsize=21, position=(0.5, 1.1))

# Legend
legend = ax.legend(title='Room', loc='upper right', bbox_to_anchor=(1.6, 1), fontsize=21, title_fontsize='21')
legend.get_frame().set_facecolor('#FFFFFF')  # Light background for legend

# Grid and ticks
ax.yaxis.set_tick_params(labelsize=19)
plt.grid(True)

plt.show()

