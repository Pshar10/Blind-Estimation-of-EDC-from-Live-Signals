import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial


"""A script for plotting heatmaps from xlsx files"""




EDC_loss = False
DT_analysis = False
RMS_plot = True

col_selection  = "Root Mean Square" if RMS_plot else "Mean"

file_path = 'Stats_table_EDC.xlsx' if EDC_loss else 'Stats_table_DT10.xlsx' if DT_analysis else 'Stats_table_T60.xlsx'

df = pd.read_excel(file_path)

# Pivot the table to get mean values for each room in each location
pivot_df = df.pivot(index='Room', columns='Location', values=col_selection)

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
if not RMS_plot:
    fig.text(0.04, 0.5, 'Mean L1 Loss for different rooms', va='center', rotation='vertical', fontsize=15)
else:
    fig.text(0.04, 0.5, 'Root Mean Square Loss for different rooms', va='center', rotation='vertical', fontsize=15)

# Remove unused axes if necessary
if len(axs) > len(locations):
    fig.delaxes(axs[-1])

plt.show()


mean_values.set_index('Room', inplace=True)

# Calculating the mean of all locations' loss for each room
mean_values['Mean_Loss'] = mean_values.mean(axis=1)

plt.figure(figsize=(10, 6))

plt.plot(mean_values.index, mean_values['Mean_Loss'], '-o', color='skyblue', markersize=8, linewidth=2, label='Mean Loss')

# Adding text labels for each point to display the mean loss value
for idx, value in enumerate(mean_values['Mean_Loss']):
    plt.text(mean_values.index[idx], value, f'{value:.2f}', ha='center', va='bottom')

plt.xlabel('Room')
plt.ylabel('Mean Loss')
if not RMS_plot:
    plt.title('Mean of All Locations Loss for Each Room')
else:
    plt.title('Root Mean Square Loss of All Locations Loss for Each Room')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.legend()

plt.show()