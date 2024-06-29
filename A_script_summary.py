import os

# Assuming you want to list .py files in the current working directory
directory_path = '.'  # Use '.' for current directory or specify the path

# List all files in the directory
files = os.listdir(directory_path)

# Filter out the .py files
py_files = [file for file in files if file.endswith('.py')]

# Write the names of the .py files to a text file
with open('py_files_list.txt', 'w') as f:
    for py_file in py_files:
        f.write(f"{py_file}\n")

print("List of .py files has been written to py_files_list.txt")
