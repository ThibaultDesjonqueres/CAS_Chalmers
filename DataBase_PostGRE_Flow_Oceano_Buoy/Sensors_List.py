import csv
import os

# Replace 'folder_path' with the actual path of the folder containing your CSV files
folder_path = r'C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052\DATA_Received\unzipped'


# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

for csv_file in csv_files:
    csv_file_path = os.path.join(folder_path, csv_file)
    unique_first_elements = set()  # Set to store unique first elements of each file

    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row:  # Check if the row is not empty
                first_element = row[0].split(',')[0]  # Extract the first element before the comma
                unique_first_elements.add(first_element)

    print(f"Sensors list for {csv_file}:")
    print(unique_first_elements)
    print()