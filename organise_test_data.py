import os
import shutil


"""A script to organize folder structure """


# Base directory where the original files are located
source_dir = "/home/prsh7458/Desktop/scratch4/speech_data/Ilmenau_test_train_IR"
# Directory where the organized files will be copied
destination_base_dir = "/home/prsh7458/Desktop/scratch4/speech_data/loc"
data_dir = os.path.join(destination_base_dir, "data")
room_locations = ["BC", "FC", "FR", "SiL", "SiR"]

# Function to create subdirectories and copy files
def organize_and_copy_files(source_directory, destination_directory, locations):
    # Create the 'data' directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)

    # Iterate over files in the source directory
    for file_name in os.listdir(source_directory):
        if file_name.endswith(".wav"):
            # Extract room location and room name from the file name
            parts = file_name.split('_')
            if len(parts) >= 4:
                location = parts[1]
                room_name = parts[-1].split('.')[0]

                # Check if location is in the specified list
                if location in locations:
                    # Paths for new subdirectories
                    room_dir = os.path.join(destination_directory, room_name)
                    location_dir = os.path.join(room_dir, location)

                    # Create subdirectories if they don't exist
                    os.makedirs(location_dir, exist_ok=True)

                    # Copy the file to the new directory
                    src_file_path = os.path.join(source_directory, file_name)
                    dest_file_path = os.path.join(location_dir, file_name)
                    shutil.copy(src_file_path, dest_file_path)

# Call the function
organize_and_copy_files(source_dir, data_dir, room_locations)
