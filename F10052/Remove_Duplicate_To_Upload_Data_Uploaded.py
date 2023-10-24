import os
import shutil

# Define the source and destination directories
data_uploaded_dir = r'C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052\Data_Uploaded'
to_upload_dir = r'C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052\To_Upload'

# Get a list of filenames in each directory
data_uploaded_files = os.listdir(data_uploaded_dir)
to_upload_files = os.listdir(to_upload_dir)

# Iterate through the files in the "To_Upload" directory
for file_to_upload in to_upload_files:
    # Check if the file exists in the "Data_Uploaded" directory
    if file_to_upload in data_uploaded_files:
        file_to_upload_path = os.path.join(to_upload_dir, file_to_upload)
        
        # Delete the file from the "To_Upload" directory
        os.remove(file_to_upload_path)
        print(f"Deleted '{file_to_upload}' from 'To_Upload'")

print("File checking and deletion completed.")
