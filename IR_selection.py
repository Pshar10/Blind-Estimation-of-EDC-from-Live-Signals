import os
import random
import shutil
from tqdm import tqdm


"""A script for ranfom selection of RIRs"""

# Directory paths
source_dir = "/home/prsh7458/Desktop/scratch4/RIR_dataset_room"
destination_dir = "/home/prsh7458/Desktop/scratch4/speech_data/Ilmenau_test_train_IR"

# Unique room locations
room_locations = ["BC", "FC", "FR", "SiL", "SiR"]

# Function to get 40 random files for each location from each folder
def get_random_files_for_each_location():
    # Check if destination directory exists, if not, create it
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Get all folders in the source directory and exclude the last one
    all_folders = sorted([f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))])
    folders_to_process = all_folders[:]  

    # Total number of operations for the progress bar
    total_operations = len(folders_to_process) * len(room_locations)

    # Initialize the progress bar
    with tqdm(total=total_operations, desc="Copying Files") as pbar:
        # Iterate through each folder in the source directory, excluding the last one
        for folder in folders_to_process:
            folder_path = os.path.join(source_dir, folder)

            # Check if it's a directory
            if os.path.isdir(folder_path):
                # Process for each location
                for location in room_locations:
                    # Get all files for this location in the folder
                    location_files = [f for f in os.listdir(folder_path) if f.endswith(".wav") and f.split('_')[2].split('.')[0] == location]

                    # Randomly select 40 files
                    if len(location_files) > 70:
                        selected_files = random.sample(location_files, 70)
                    else:
                        selected_files = location_files

                    # Copy selected files to the destination directory with new naming convention
                    for i, file in enumerate(selected_files, start=1):
                        new_name = f"{i}_{location}_{folder}.wav"
                        src_file_path = os.path.join(folder_path, file)
                        dest_file_path = os.path.join(destination_dir, new_name)
                        shutil.copy(src_file_path, dest_file_path)

                    # Update the progress bar
                    pbar.update(1)

# Execute the function
get_random_files_for_each_location()




# # Directory paths
# source_dir = "/home/prsh7458/work/RIR_dataset_room"
# destination_dir = "/home/prsh7458/work/speech_data/Ilmenau_test_IR"

# # Unique room locations
# room_locations = ["BC", "FC", "FR", "SiL", "SiR"]

# # Function to get 40 random files for each location from each folder
# def get_random_files_for_each_location():
#     # Check if destination directory exists, if not, create it
#     if not os.path.exists(destination_dir):
#         os.makedirs(destination_dir)

#     # Get all folders in the source directory
#     all_folders = sorted([f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))])
#     folders_to_process = [all_folders[-1]]  # Only process the last folder

#     # Total number of operations for the progress bar
#     total_operations = len(folders_to_process) * len(room_locations)

#     # Initialize the progress bar
#     with tqdm(total=total_operations, desc="Copying Files") as pbar:
#         # Iterate through the last folder in the source directory
#         for folder in folders_to_process:
#             folder_path = os.path.join(source_dir, folder)

#             # Check if it's a directory
#             if os.path.isdir(folder_path):
#                 # Process for each location
#                 for location in room_locations:
#                     # Get all files for this location in the folder
#                     location_files = [f for f in os.listdir(folder_path) if f.endswith(".wav") and f.split('_')[2].split('.')[0] == location]

#                     # Randomly select 40 files
#                     if len(location_files) > 40:
#                         selected_files = random.sample(location_files, 40)
#                     else:
#                         selected_files = location_files

#                     # Copy selected files to the destination directory with new naming convention
#                     for i, file in enumerate(selected_files, start=1):
#                         new_name = f"{i}_{location}_{folder}.wav"
#                         src_file_path = os.path.join(folder_path, file)
#                         dest_file_path = os.path.join(destination_dir, new_name)
#                         shutil.copy(src_file_path, dest_file_path)

#                     # Update the progress bar
#                     pbar.update(1)

# # Execute the function
# get_random_files_for_each_location()
