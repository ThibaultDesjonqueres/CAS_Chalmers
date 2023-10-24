import os
import csv
from collections import defaultdict

# Specify the folder path containing your CSV files
folder_path = r'C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10051\DATA_Received\unzipped'


# Create a dictionary to store unique values and their counts for each CSV file
csv_data = defaultdict(lambda: defaultdict(int))

# Iterate through each CSV file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        csv_file_path = os.path.join(folder_path, filename)
        
        # Open the CSV file
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            
            # Iterate through each row in the CSV file
            for row in csv_reader:
                if row:  # Check if the row is not empty
                    first_variable = row[0].split(',')[0]  # Extract the first variable
                    csv_data[filename][first_variable] += 1  # Increment the count
            
            # Print the unique values and their counts for the current CSV file
            print(f"CSV File: {filename}")
            for value, count in csv_data[filename].items():
                print(f"{value}: {count}")
            print('-' * 40)
