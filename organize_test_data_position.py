import os
import shutil



"""A script to organize folder structure"""
# Base directory where the original files are located
source_base_dir = "/home/prsh7458/Desktop/scratch4/RIR_dataset_room_with_location"
# Directory where the organized files will be copied
destination_base_dir = "/home/prsh7458/Desktop/scratch4/speech_data/loc/position"

# Function to create subdirectories and copy files
def organize_and_copy_files(source_base_directory, destination_base_directory):
    # Iterate over room folders in the base directory
    for room_name in os.listdir(source_base_directory):
        room_dir_path = os.path.join(source_base_directory, room_name)
        if os.path.isdir(room_dir_path):
            # Iterate over files in the room directory
            for file_name in os.listdir(room_dir_path):
                if file_name.endswith(".wav"):
                    # Extract location from the file name
                    parts = file_name.split('_')
                    if len(parts) >= 6:
                        location = parts[2]

                        # Paths for new subdirectories in the destination
                        destination_room_dir = os.path.join(destination_base_directory, room_name)
                        destination_location_dir = os.path.join(destination_room_dir, location)

                        # Create subdirectories if they don't exist
                        os.makedirs(destination_location_dir, exist_ok=True)

                        # Copy the file to the new directory
                        src_file_path = os.path.join(room_dir_path, file_name)
                        dest_file_path = os.path.join(destination_location_dir, file_name)
                        shutil.copy(src_file_path, dest_file_path)

# Call the function
organize_and_copy_files(source_base_dir, destination_base_dir)
