import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""A script for plotting heatmaps from xlsx files"""
# Set the flag based on your condition
edc_analysis = True
RMS_plot = True

col_selection  = "Root Mean Square" if RMS_plot else "Mean"
# Choose the file based on the condition
file_to_load = 'Stats_table_EDC.xlsx' if edc_analysis else 'Stats_table_T60.xlsx'

# Read the Excel file
df = pd.read_excel(file_to_load)

# Pivot the DataFrame so that 'Room' is on the x-axis and 'Location' on the y-axis
# We assume 'Location' and 'Room' are columns in the DataFrame along with 'Mean'
df_pivot = df.pivot("Room", "Location", col_selection)

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Draw the heatmap
ax = sns.heatmap(df_pivot, annot=True, fmt=".2f", linewidths=.5, cmap='Blues', annot_kws={"size": 24}, vmin=0 if not edc_analysis else None, vmax=0.5 if not edc_analysis else None)


# Enhance the size for labels.
ax.tick_params(axis='x', labelsize=20, rotation=45) # For x-axis tick labels
ax.tick_params(axis='y', labelsize=20, rotation=0)  # For y-axis tick labels
ax.set_xlabel('Location', fontsize=24)
ax.set_ylabel('Room', fontsize=24)

if not RMS_plot:
    plt.title(r"Mean L1 EDC loss across all rooms and locations (dB)", fontsize=25) if edc_analysis else plt.title(r'MAE $T_{0-15}$ Loss for all Rooms and Locations (sec)', fontsize=25)
else:
    plt.title(r"Root Mean Square EDC loss across all rooms and locations (dB)", fontsize=25) if edc_analysis else plt.title(r'Root Mean Square $T_{0-15}$ Loss for all Rooms and Locations (sec)', fontsize=25)

# Set the colorbar labels font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=24)
plt.subplots_adjust(left=0.102, bottom=0.126, right=0.786, top=0.936, wspace=0.2, hspace=0.195)

# Show the plot
plt.show()