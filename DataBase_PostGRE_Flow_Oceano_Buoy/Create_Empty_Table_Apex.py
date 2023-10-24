import psycopg2
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, types


###
# This code creates the empty table based on 1 csv file. 
# Then I would like to use To improve 2 to avoid duplicates 
# At this point I can 
#
#
###
def clean_column_names(column_name):
    cleaned_name = column_name.replace('#', '').lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    return cleaned_name.lower()

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

folder_path = r"C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052\To_Upload"
# csv_folder_path = os.path.join(folder_path, "csv_to_sql")

# if os.path.exists(csv_folder_path):
#     print("csv_to_sql already exists")
# else:
#     print("csv_to_sql Created")
#     os.makedirs(csv_folder_path)

# List all files in the folder
file_list = os.listdir(folder_path)


# Loop over the files and read each CSV into a DataFrame
for filename in file_list:
    if filename.endswith('.csv'):
        print(filename)
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        table_name = 'float_f10052'
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
        print(df.head(2))
        
        # Check for duplicates based on primary key or unique columns
        primary_key_columns = ['timestamp', "filename"]  # Modify this with your actual primary key or unique columns
    
        # df.to_sql(name=table_name, con=conn, if_exists='append', index=False, method='multi', chunksize=1000, \
        #           dtype=sqlalchemy_dtypes, index_label=primary_key_columns)
        
        df.iloc[:0].to_sql(name=table_name, con=conn, if_exists='append', index=False, dtype=sqlalchemy_dtypes, index_label=primary_key_columns)
        # df.to_sql(name=table_name, con=conn, if_exists='append', index=False, dtype=sqlalchemy_dtypes, index_label=primary_key_columns)
        break
        
conn.commit()
conn.close()
