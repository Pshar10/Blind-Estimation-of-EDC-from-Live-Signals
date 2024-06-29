import plotly.express as px
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.patches as patches

""" A script to plot the Loss graphs for Each coordinate in individual room"""






# Function to draw a simple speaker icon
def draw_speaker_icon(ax, center, size=0.1):
    # Speaker body (square)
    speaker_body = patches.Rectangle((center[0] - size/2, center[1] - size/2), size, size, 
                                     fill=True, color='black', zorder=3)
    ax.add_patch(speaker_body)

    # Sound waves (triangles)
    for i in range(1, 4):
        wave = patches.RegularPolygon((center[0] + size/2, center[1]), 3, size/(2*i), 
                                      orientation=np.pi/2, fill=False, edgecolor='black', zorder=3)
        ax.add_patch(wave)


def extract_tb_data(logdir, position_tags, loss_tag):
    ea = event_accumulator.EventAccumulator(logdir,
                                            size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    # Extract position data
    positions = []
    for tag in position_tags:
        if tag in ea.scalars.Keys():
            scalar_events = ea.scalars.Items(tag)
            positions.append([s.value for s in scalar_events])

    # Extract loss data
    losses = []
    if loss_tag in ea.scalars.Keys():
        scalar_events = ea.scalars.Items(loss_tag)
        losses = [s.value for s in scalar_events]

    return positions, losses



base_logdir = '/home/prsh7458/work/R2D/test_runs_position'
# Rooms and locations
rooms = ["HL00W", "HL01W", "HL02WL", "HL02WP", "HL03W", "HL04W", "HL05W", "HL06W", "HL08W"] 
#rooms = ["HL00W"] 
room_locations = ["BC", "FC", "FR", "SiL", "SiR"]
#room_locations = ["BC"]
loss_tag = 'Loss/total_loss_test'
position_tags = ['Position/x_coordinate', 'Position/y_coordinate', 'Position/z_coordinate']

# Predefined source positions
source_positions = {
    "BC": [-1.7, -1.81, 1.73],
    "FC": [2.28, 0, 1.73],
    "FR": [3.4, -2.87, 1.73],
    "SiL": [1.31, 2.97, 1.73],
    "SiR": [0.02, -2.91, 1.73]
}


save_bool = False
save_base_dir = '/home/prsh7458/work/R2D/Loss_Graphs/loss_loc/grid_graphs'

# for room in rooms:
#     for loc in room_locations:
#         logdir = f'{base_logdir}/R2DNet_{room}_{loc}'
#         positions, losses = extract_tb_data(logdir, position_tags, loss_tag)

#         if positions and losses:
#             positions = np.array(positions).T

#             # Set the z coordinate of all positions to the z value of the source position
#             z_value = source_positions[loc][2]
#             positions[:, 2] = z_value  # This sets all z values to the source's z

#             # Create 3D plot
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')

#             # Plot listener positions with color based on loss
#             sc = ax.scatter(positions[:, 0], positions[:, 1], np.full_like(losses, z_value), c=losses, cmap='hot', vmin=1, vmax=25)

#             # Plot source position for this location with a slightly larger marker size
#             source_pos = source_positions[loc]
#             ax.scatter(source_pos[0], source_pos[1], source_pos[2], color='blue', s=100, depthshade=False, label=f'Source: {loc}')

#             # Adding color bar
#             plt.colorbar(sc, label='Loss')

#             # Labeling and title
#             ax.set_xlabel('X Coordinate')
#             ax.set_ylabel('Y Coordinate')
#             ax.set_zlabel('Z Coordinate')
#             plt.title(f'3D Grid for Room {room}, Location {loc}')
#             plt.legend()

#             # Show plot or save to file
#             if save_bool:
#                 save_path = os.path.join(save_base_dir, f'3DGrid_{room}_{loc}.png')
#                 plt.savefig(save_path)
#             else:
#                 plt.show()

import plotly.graph_objects as go

def plot_2d_loss_contour_plotly(grouped, room, loc):
    # Creating the contour plot
    fig = go.Figure(data=go.Contour(
        z=grouped['mean_loss'],
        x=grouped['x'],  # X coordinates
        y=grouped['y'],  # Y coordinates
        colorscale='Blues',
        contours=dict(
            coloring='heatmap',
            showlabels=True,
        )
    ))

    # Set the range of the plot to the range of the data points
    x_range = [grouped['x'].min(), grouped['x'].max()]
    y_range = [grouped['y'].min(), grouped['y'].max()]
    fig.update_layout(
        title=f'2D Loss Contour for Room {room}, Location {loc}',
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        xaxis_range=x_range,
        yaxis_range=y_range
    )

    # Display the plot
    fig.show()



global_min_loss = float('inf')
global_max_loss = float('-inf')

# First pass to determine global min and max loss values
for room in rooms:
    for loc in room_locations:
        logdir = f'{base_logdir}/R2DNet_{room}_{loc}'
        _, losses = extract_tb_data(logdir, position_tags, loss_tag)
        if losses:
            local_min = min(losses)
            local_max = max(losses)
            global_min_loss = min(global_min_loss, local_min)
            global_max_loss = max(global_max_loss, local_max)

for room in rooms:
    for loc in room_locations:
        logdir = f'{base_logdir}/R2DNet_{room}_{loc}'
        positions, losses = extract_tb_data(logdir, position_tags, loss_tag)

        if positions and losses:
            positions = np.array(positions).T
            df = pd.DataFrame(positions, columns=['x', 'y', 'z'])
            df['loss'] = losses
            grouped = df.groupby(['x', 'y']).agg(mean_loss=('loss', 'mean')).reset_index()
            fig, ax = plt.subplots()

            # Set the background color
            ax.set_facecolor('black')  # You can choose a different shade as needed

            # Increase marker size in scatter plot
            marker_size = 100  # Adjust this value as needed
            sc = ax.scatter(grouped['x'], grouped['y'], c=grouped['mean_loss'], s=marker_size,
                            cmap='Blues', vmin=global_min_loss, vmax=global_max_loss)
                            # cmap='Blues', vmin=0, vmax=2)

            source_pos = source_positions[loc]
            ax.scatter(source_pos[0], source_pos[1], color='red', marker='x', label="Source", s=marker_size)
            plt.colorbar(sc, label='Mean Loss')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            plt.title(f'2D Projection for Room {room}, Location {loc}')
            ax.legend(loc='lower center', bbox_to_anchor=(0, -0.15), ncol=1)
            save_path = os.path.join(save_base_dir, f'2D_Projection_{room}_{loc}.png')

            # plot_2d_loss_contour_plotly(grouped,room,loc)
            if save_bool:
                plt.savefig(save_path)
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)
