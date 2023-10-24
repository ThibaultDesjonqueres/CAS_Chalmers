
import os
import shutil
import gzip

# Define the path of the folder containing the data folders
data_folder_path = r'C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052\DATA_Received'

# Create the directory to store the unzipped files if it does not exist
unzipped_folder = os.path.join(data_folder_path, 'unzipped')
if not os.path.exists(unzipped_folder):
    os.makedirs(unzipped_folder)

# Loop through all the data folders
for data_folder in os.listdir(data_folder_path):
    current_data_folder_path = os.path.join(data_folder_path, data_folder)
    if os.path.isdir(current_data_folder_path):
        # Loop through all the files in the data folder
        for filename in os.listdir(current_data_folder_path):
            if filename.endswith('.gz'):
                # Open the gzipped file and read the contents
                with gzip.open(os.path.join(current_data_folder_path, filename), 'rb') as f:
                    file_content = f.read()
                # Write the unzipped content to a file in the unzipped folder
                with open(os.path.join(unzipped_folder, filename[:-3]), 'wb') as f:
                    f.write(file_content)

# After processing all the data folders, delete all folders except for the "unzipped" folder
for folder in os.listdir(data_folder_path):
    folder_path = os.path.join(data_folder_path, folder)
    if os.path.isdir(folder_path) and folder != 'unzipped':
        shutil.rmtree(folder_path)

