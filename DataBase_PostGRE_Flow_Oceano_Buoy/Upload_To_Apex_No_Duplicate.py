# import psycopg2
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, types,MetaData, text, insert

def clean_column_names(column_name):
    cleaned_name = column_name.replace('#', '').lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    return cleaned_name.lower()


insertion_successful = False  # Initialize a flag to track successful insertions


# PostgreSQL connection parameters
db_user = 'postgres'
db_password = 'calamar'
db_host = 'localhost'
db_port = '5432'
db_name = 'test'

# Create a SQLAlchemy engine
conn_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
db = create_engine(conn_string)

# Establish a connection
conn = db.connect()
print("Opened db successfully")


table_name = 'float_f10052'

# Using SQLAlchemy's MetaData to check if the table exists
metadata = MetaData()
metadata.reflect(bind=db)
table_exists = table_name in metadata.tables
if table_exists :
    print("It exists")
if not table_exists:
    print("Table 'float_f10053' does not exist.")

folder_path = r"C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052\To_Upload"
csv_folder_path = os.path.join(folder_path, "csv_to_sql")

if not os.path.exists(csv_folder_path):
    print("csv_to_sql Created")
    os.makedirs(csv_folder_path)
else:
    print("csv_to_sql already exists")

# Fetch existing data from the database based on primary key or unique columns
table_name = 'float_f10052'
primary_key_columns = ['timestamp', 'filename']  # Modify this with your actual primary key or unique columns
existing_data_query = f"SELECT {', '.join(primary_key_columns)} FROM {table_name}"
existing_data = pd.read_sql(existing_data_query, conn)

# Initialize a list to store new rows for insertion
new_rows = []

# List all files in the folder
file_list = os.listdir(folder_path)

# Loop over the files and read each CSV into a DataFrame
for filename in file_list:
    if filename.endswith('.csv'):
        print(filename)
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        df.rename(columns=clean_column_names, inplace=True)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values(by='timestamp', inplace=True)
        replacements = {
            'object': types.VARCHAR,
            'float64': types.FLOAT, 
            'int64': types.INT, 
            'datetime64[ns]': types.TIMESTAMP, 
            'timedelta64[ns]': types.VARCHAR            
        }
        
        df['filename'] = filename
        # Define the SQLAlchemy data types for columns
        sqlalchemy_dtypes = {col: replacements.get(str(dtype), types.VARCHAR) for col, dtype in zip(df.columns, df.dtypes)}
        
        col_str = ", ".join(f"{n} {d}" for n, d in zip(df.columns, sqlalchemy_dtypes.values()))
        print("DataFrame shape for", filename, ":", df.shape)
        # print(df.head(2))
        
        # Filter out rows that already exist in the database
        unique_columns = primary_key_columns
        new_rows_df = df[~df[unique_columns].isin(existing_data.to_dict(orient='list')).all(axis=1)]
        
        unique_columns = primary_key_columns
        new_rows_df = df[~df[unique_columns].isin(existing_data.to_dict(orient='list')).all(axis=1)]
        
        # if not new_rows_df.empty:
        #     new_rows.extend(new_rows_df.to_dict(orient='records'))
        #     existing_data = pd.concat([existing_data, new_rows_df[unique_columns]], ignore_index=True)
        # else:
        #     print("All rows in", filename, "already exist in the database.")
        if not new_rows_df.empty:
            new_rows.extend(new_rows_df.to_dict(orient='records'))
            existing_data = pd.concat([existing_data, new_rows_df], ignore_index=True)
        else:
            print("All rows in", filename, "already exist in the database.")


processed_files = set()  # Initialize a set to keep track of processed filenames

for filename in file_list:
    if filename.endswith('.csv') and filename not in processed_files:
        processed_files.add(filename)  # Add the filename to the set
        
        print("Processing file:", filename)  # Print the CSV file being processed
        file_path = os.path.join(folder_path, filename)
        
        # Rest of your processing code

        # ... (rest of the code remains unchanged)
path_processed =  r'C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052'
processed_folder_path = os.path.join(path_processed, "Data_Uploaded")
        
if new_rows:
    table = metadata.tables[table_name]  # Get the table object from metadata
    ins = insert(table).values(new_rows)  # Create an insert statement
    print("Inserting rows from file:", filename) 
    try:
        conn.execute(ins)  # Execute the insert statement
        conn.commit()  # Commit the transaction
        print("Data inserted successfully.")
        insertion_successful = True  # Set the flag to True if insertion is successful
    except Exception as e:
        print("An error occurred during insertion:", str(e))

# ... (rest of the code remains unchanged)

if insertion_successful:
    # Move the processed files to the 'Data_Uploaded' folder
    processed_folder_path = os.path.join(path_processed, "Data_Uploaded")

    if not os.path.exists(processed_folder_path):
        os.makedirs(processed_folder_path)

    for filename in file_list:
        if filename.endswith('.csv'):
            source_path = os.path.join(folder_path, filename)
            destination_path = os.path.join(processed_folder_path, filename)
            os.rename(source_path, destination_path)

    print("Processed CSV files moved to 'Data_Uploaded' folder.")
else:
    print("No successful insertions were made. Files will not be moved to 'Data_Uploaded' folder. If the files had already been uploaded BEFORE it will also not make the files move")

# Close the connection
conn.close()



#%%   
# if new_rows:
#     table = metadata.tables[table_name]  # Get the table object from metadata
#     ins = insert(table).values(new_rows)  # Create an insert statement

#     # Print the name of the CSV file being inserted
#     print("Inserting rows from file:", filename)  # 'filename' refers to the last processed file

#     try:
#         conn.execute(ins)  # Execute the insert statement
#         conn.commit()  # Commit the transaction
#         print("Data inserted successfully.")
#     except Exception as e:
#         print("An error occurred during insertion:", str(e))
    
    
# conn.close()
# path_processed =  r'C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052'
# processed_folder_path = os.path.join(path_processed, "Data_Uploaded")

# if not os.path.exists(processed_folder_path):
#     os.makedirs(processed_folder_path)

# for filename in file_list:
#     if filename.endswith('.csv'):
#         source_path = os.path.join(folder_path, filename)
#         destination_path = os.path.join(processed_folder_path, filename)
#         os.rename(source_path, destination_path)

# print("Processed CSV files moved to 'Data_Uploaded' folder.")