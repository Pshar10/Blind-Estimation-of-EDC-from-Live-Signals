from moviepy.editor import ImageSequenceClip
import os
import random

"""A script to make movie from the saved plots"""

base_logdir = '/home/prsh7458/Desktop/scratch4/R2D/curves/curves_loc'
rooms = ["HL00W", "HL01W", "HL02WL", "HL02WP", "HL03W", "HL04W", "HL05W", "HL06W", "HL08W"]
room_locations = ["BC", "FC", "FR", "SiL", "SiR"]
duration_per_image = 0.25  # Duration to display each image
max_images_per_folder = 15  # Maximum number of images to include from one folder

all_image_files = []

# Collecting image files from all folders
for room in rooms:
    for location in room_locations:
        dir_path = os.path.join(base_logdir, f'room_{room}_loc_{location}')
        image_files = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path)) if f.endswith('.png')]
        
        # Optionally, randomly select a subset of images from each folder
        if len(image_files) > max_images_per_folder:
            image_files = random.sample(image_files, max_images_per_folder)

        all_image_files.extend(image_files)

# Shuffle the entire collection of images if desired
random.shuffle(all_image_files)

# Create and save the video if there are any images
if all_image_files:
    clip = ImageSequenceClip(all_image_files, durations=[duration_per_image] * len(all_image_files))
    output_path = os.path.join(base_logdir, 'combined_movie.mp4')
    clip.write_videofile(output_path, fps=1/duration_per_image)
