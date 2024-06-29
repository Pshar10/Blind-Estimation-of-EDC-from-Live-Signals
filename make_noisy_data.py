import os
import shutil
import random


"""A script to select data for noise test"""

source_base_dir = "/home/prsh7458/Desktop/scratch4/speech_data/loc/data"
destination_dir = "/home/prsh7458/Desktop/scratch4/noise_roburstness_test/room_data"

def select_random_files (source_base_dir,destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    for room_name in os.listdir(source_base_dir):
        for location in os.listdir(os.path.join(source_base_dir, room_name)):
            room_dir_path = os.path.join(source_base_dir, room_name,location)
            if not os.path.exists(os.path.join(destination_dir, room_name,location)):
                os.makedirs(os.path.join(destination_dir, room_name,location)) 
            for file_name in os.listdir(room_dir_path):
                files = [f for f in os.listdir(room_dir_path) if f.endswith(".wav")]
                selected_files = random.sample(files, 10) if len(files) > 10 else files
                for i, file in enumerate(selected_files, start=1):
                    src_file_path = os.path.join(source_base_dir,room_name, location,file)
                    new_name = f"{i}_{location}_{room_name}.wav"
                    dest_file_path = os.path.join(destination_dir,room_name,location,new_name)
                    shutil.copy(src_file_path, dest_file_path)


