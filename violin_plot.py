import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

"""A script plotting simple violin plot from the log files of tensorbeard"""

def find_recent_log_file(directory, file_prefix):
    # List all files in the directory that start with the given prefix
    files = glob.glob(os.path.join(directory, f"{file_prefix}*"))
    
    # Find the most recently modified file
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    
    return latest_file


def extract_total_loss(log_file_path, epoch, mode):
    total_loss_values = []

    with open(log_file_path, 'r') as file:
        for line in file:
            if f"{mode} Epoch: {epoch}" in line:
                parts = line.split('\t')
                for part in parts:
                    if 'Total Loss:' in part:
                        try:
                            total_loss = float(part.split(':')[1].split(',')[0].strip())
                            total_loss_values.append(total_loss)
                        except ValueError as e:
                            print(f"Error parsing line: {line}")
                            print(e)
                        break

    return total_loss_values

def plot_violin(total_loss_values, epoch, mode):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=total_loss_values)
    plt.title(f'Violin Plot of {mode} Total Loss Values for Epoch {epoch}')
    plt.xlabel('Total Loss')
    plt.show()

# Path to your log file
    
directory = '/home/prsh7458/work/R2D/edc-estimation/'
file_prefix = 'lsf'

# log_file_path = find_recent_log_file(directory, file_prefix)
# if log_file_path:
#     print(f"Most recent log file: {log_file_path}")
# else:
#     print("No log files found with the specified prefix.")


log_file_path = '/home/prsh7458/work/R2D/edc-estimation/lsf_1443451_gpu.log'
epoch = 100
# mode = 'Train'  # or 'Test'
mode = 'Test' 
# Extract total loss values and plot
total_loss_values = extract_total_loss(log_file_path, epoch, mode)
plot_violin(total_loss_values, epoch, mode)

