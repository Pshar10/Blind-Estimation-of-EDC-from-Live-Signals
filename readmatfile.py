import h5py
import numpy as np
import random
from matplotlib import pyplot as plt

# Open the .mat file
with h5py.File('/home/prsh7458/work/R2D/edc-estimation/edc_data/edcs_processed.mat', 'r') as f_edcs:
    # Access the 'edcData' dataset
    edcData_refs = f_edcs['edcData']

    # Initialize an empty list to store the EDC data
    edcData_list = []

    # Loop over each reference in the columns
    for i in range(edcData_refs.shape[1]):  # Iterate over columns
        ref = edcData_refs[0, i]  # Accessing the reference
        edcData_list.append(np.array(f_edcs[ref]).flatten())

    # Check the number of EDCs retrieved
    print('Number of EDCs retrieved:', len(edcData_list))

# Choose a random EDC to plot
random_index = random.randint(0, len(edcData_list) - 1)
random_edc = edcData_list[random_index]

# Plotting the randomly selected EDC
plt.figure()
plt.plot(random_edc)
plt.title(f'Randomly Selected EDC (Index: {random_index})')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.show()
