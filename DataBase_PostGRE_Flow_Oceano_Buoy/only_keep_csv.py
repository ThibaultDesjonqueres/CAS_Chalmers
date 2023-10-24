import os

def keep_csv_files(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('science_log.csv')]
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file not in csv_files:
            os.remove(file_path)

# Replace 'folder_path' with the path to the folder you want to filter
# folder_path = r'C:\Users\thiba\OneDrive\Documents\Float\F10052\unzipped'
folder_path = r'C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052\DATA_Received\unzipped'
keep_csv_files(folder_path)
