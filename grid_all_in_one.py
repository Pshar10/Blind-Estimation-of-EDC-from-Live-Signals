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
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

""" A script to plot the Loss graphs for Each coordinate in the room for all the rooms combined"""

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
# rooms = ["HL00W"] 
room_locations = ["BC", "FC", "FR", "SiL", "SiR"]
# room_locations = ["BC", "FC", "FR"]
edc_mae_loss = True

if not edc_mae_loss:
    loss_tag = 'Loss/total_loss_test'
    # loss_tag = 'Loss/total_loss_test_DT'
else:
    loss_tag = 'Loss/total_loss_test_edc'

position_tags = ['Position/x_coordinate', 'Position/y_coordinate', 'Position/z_coordinate']

# Predefined source positions
source_positions = {
    "BC": [-1.7, -1.81, 1.73],
    "FC": [2.28, 0, 1.73],
    "FR": [3.4, -2.87, 1.73],
    "SiL": [1.31, 2.97, 1.73],
    "SiR": [0.02, -2.91, 1.73]
}


save_base_dir = '/home/prsh7458/work/R2D/Loss_Graphs/loss_loc/grid_graphs'



# Create an empty DataFrame to hold all the data
all_data = pd.DataFrame()

# Loop to gather data from all rooms and locations
for room in rooms:
    for loc in room_locations:
        logdir = f'{base_logdir}/R2DNet_{room}_{loc}'
        positions, losses = extract_tb_data(logdir, position_tags, loss_tag)

        if positions and losses:
            positions = np.array(positions).T
            df = pd.DataFrame(positions, columns=['x', 'y', 'z'])
            exp = 0.5 # FOR MAE
            # df['loss'] = np.clip(np.array(losses)**exp,0,0.3) ####################################################correct the logic here as of now its mae #########
            if edc_mae_loss:
                df['loss'] = np.array(losses)
            else:
                df['loss'] = np.clip(np.array(losses)**exp,0,0.5)

            grouped = df.groupby(['x', 'y']).agg(mean_loss=('loss', 'mean')).reset_index()
            grouped['room'] = room  # Add a column for the room
            grouped['location'] = loc  # Add a column for the location

            all_data = pd.concat([all_data, grouped])

# Assuming 'all_data' is already created and contains 'room' and 'location' columns
if not all_data.empty:
    num_rooms = len(rooms)
    num_locations = len(room_locations)
    
    # Set a minimum subplot width and calculate total figure width
    min_subplot_width = 200  # Minimum width for each subplot
    fig_width = max(min_subplot_width * num_rooms, 550)  # Ensure the figure is at least 550px wide

    # Calculate figure height based on the number of locations
    subplot_height = 150  # Height for each subplot row
    fig_height = subplot_height * num_locations
    
    # Create a Faceted Grid using Plotly with dynamic adjustments
    fig = px.scatter(all_data, x='x', y='y', color='mean_loss',
                    facet_col='room', facet_row='location',
                    color_continuous_scale='Blues',  #virdis
                    title='2D Loss Projection Across Rooms and Locations',
                    height=fig_height, width=fig_width)

    # Update layout for better readability and adjust legend and grid
    fig.update_layout(
        xaxis_title='X Coordinate', 
        yaxis_title='Y Coordinate',
        title_font_size=28,  # Increase title font size
        plot_bgcolor='black',  # Set background color to black
        paper_bgcolor='black',  # Set paper background color to black
        font_color="white",  # Set font color to white for better visibility
        font_size=28,  # Increase general font size for the plot
        legend=dict(
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1.065,
            font=dict(  # Adjust font size specifically for legend
                size=28,  # Increase legend font size
                color="white"  # Ensure legend font color is white for visibility
            ),
            # Remove prefix from legend labels
            title=None,
            itemsizing='trace'  
        ),
        xaxis=dict(showgrid=False),  # Remove x-axis grid
        yaxis=dict(showgrid=False)  # Remove y-axis grid
    )

    # Adjust legend labels to remove the "room =" and "location =" prefixes
    for i in range(len(fig.layout.annotations)):
        text = fig.layout.annotations[i].text
        fig.layout.annotations[i].text = text.split('=')[-1].strip()  # Remove the prefix

    # Adjusting marker size and turning off the grid
    fig.update_traces(marker=dict(size=5))  # Increase marker size as needed
    fig.update_xaxes(showgrid=False, zeroline=False, title_font=dict(size=26))  # Increase x-axis title font size
    fig.update_yaxes(showgrid=False, zeroline=False, title_font=dict(size=26))  # Increase y-axis title font size

    # Add source position markers for each room-location combination
    for loc_idx, loc in enumerate(room_locations, 1):
        for room_idx, room in enumerate(rooms, 1):
            source_pos = source_positions[loc]
            fig.add_trace(go.Scatter(
                x=[source_pos[0]], 
                y=[source_pos[1]], 
                mode='markers',
                marker_symbol='x',
                marker_color='red', 
                marker_size=6.9, 
                name=f"Source",
                legendgroup=loc,
                showlegend=(room_idx == 1 and loc_idx == 1)
            ), row=(len(room_locations) - loc_idx + 1), col=room_idx)

    fig.show()
else:
    print("No data available for plotting.")



###################################################################################################################################################



save_bool = False




###################################################################################################################################################


if save_bool:
    # Save the figure as an HTML file
    file_name = '2D_T60_Loss_Projection.html'  # or '2D_Loss_Projection.png' for an image file

    # Full path for saving the figure
    save_path = os.path.join(save_base_dir, file_name)

    # Save the figure as an HTML file
    fig.write_html(save_path)
