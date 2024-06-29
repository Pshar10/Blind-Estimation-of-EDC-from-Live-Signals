import os
import re
import shutil
import random


"""A script for random speech files selection from random IDs"""
def list_random_files_for_id(id, directory, num_files=5):
    """
    List num_files random files for a given ID in the specified directory.
    """
    # Match files that start with the ID followed by a hyphen
    pattern = str(id) + '-'
    files = [f for f in os.listdir(directory) if f.startswith(pattern)]

    # Select num_files random files from this list
    random_files = random.sample(files, min(num_files, len(files)))

    return sorted(random_files)

def copy_and_rename_files(file_names, source_directory, target_directory):
    """
    Copy the specified files from the source directory to the target directory with renamed filenames.
    """
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    file_counter = 1  # Counter to keep track of file numbers
    for id, files in file_names.items():
        gender = 'male' if id in male_ids else 'female'  # Determine the gender based on ID
        for file in files:
            new_filename = f"{file_counter}_{id}_{gender}.flac"
            shutil.copy(os.path.join(source_directory, file), os.path.join(target_directory, new_filename))
            file_counter += 1



file_path = "/home/prsh7458/Desktop/scratch4/decaynet/DecayFitNet/data/Librispeech/LibriSpeech/SPEAKERS.TXT"
# Process the file to separate male and female IDs for the "train-clean-360" subset
male_ids = []
female_ids = []
with open(file_path, 'r') as file:
    for line in file:
        if line.startswith(';') or not line.strip():
            continue  # Skip comments and empty lines
        parts = line.split('|')
        if len(parts) == 5:  # Ensure correct line format
            reader_id, gender, subset, _, _ = parts
            if subset.strip() == 'train-clean-100':
                if gender.strip() == 'M':
                    male_ids.append(reader_id.strip())
                elif gender.strip() == 'F':
                    female_ids.append(reader_id.strip())

# Select 5 random male and 5 random female IDs

num_ppl = 3

random_male_ids = random.sample(male_ids, min(num_ppl, len(male_ids)))
random_female_ids = random.sample(female_ids, min(num_ppl, len(female_ids)))

print("Random Male IDs:", random_male_ids)
print("Random Female IDs:", random_female_ids)

file_path = "/home/prsh7458/Desktop/scratch4/decaynet/DecayFitNet/data/Librispeech/LibriSpeech/SPEAKERS.TXT"
source_directory = "/home/prsh7458/Desktop/scratch4/decaynet/DecayFitNet/data/Librispeech/LibriSpeech/all_files"
target_directory = "/home/prsh7458/Desktop/scratch4/speech_data/raw_test_speech"



file_names = {}
for id in random_male_ids:
    file_names[id] = list_random_files_for_id(id, source_directory)
for id in random_female_ids:
    file_names[id] = list_random_files_for_id(id, source_directory)

# Copy the files
copy_and_rename_files(file_names, source_directory, target_directory)

# Print confirmation
print("Files copied successfully.")






