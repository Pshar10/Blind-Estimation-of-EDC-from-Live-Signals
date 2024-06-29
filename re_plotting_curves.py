import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
"""Plot saved figures in a grid """
def plot_random_images_from_directory(directory_path):
    """
    Plots 4 random images from the given directory in a 2x2 subplot grid.
    
    Parameters:
    - directory_path: str, path to the directory containing image files.
    """
    # Get all files in the directory
    all_files = os.listdir(directory_path)
    
    # Filter out files that are images (assuming PNG, JPG, JPEG extensions)
    image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Select 4 random images
    selected_images = random.sample(image_files, min(len(image_files), 4))
    
    # Set up the figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    for ax, image in zip(axs.ravel(), selected_images):
        img_path = os.path.join(directory_path, image)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')  # Hide the axis
        # ax.set_title(image)
    
    plt.tight_layout()
    plt.show()


rooms = ["HL00W", "HL01W", "HL02WL", "HL02WP", "HL03W", "HL04W", "HL05W", "HL06W", "HL08W"] 
room_locations = ["BC", "FC", "FR", "SiL", "SiR"]

for i,r in enumerate(rooms):
    for j,l in enumerate(room_locations):
        path = '/home/prsh7458/Desktop/scratch4/R2D/curves/curves_loc/room_' + r + '_loc_'+ l
        plot_random_images_from_directory(path)