"""
As it is, I save and upload rows which do not have the most info (such as saving GPS long loat many times, without TPS)
I could solve this by droppoing the NaNs, but then if only 1 sensor fails (say S), I also loose P and T. 
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
# Uncomment to make a map of the location
# import geopandas as gpd
# from shapely.geometry import Point
# import contextily as ctx

# Set the folder path
base_path = r"C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052"
ctd_plots_folder = os.path.join(base_path, 'CTD_Plots')
to_upload_folder = os.path.join(base_path, 'To_Upload')

# Create folders if they don't exist
if not os.path.exists(ctd_plots_folder):
    os.makedirs(ctd_plots_folder)

if not os.path.exists(to_upload_folder):
    os.makedirs(to_upload_folder)

cwd = os.getcwd()
dataFolder_bin_to_Csv = r'C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052\DATA_Received\unzipped'
folder_path = os.path.join(cwd, dataFolder_bin_to_Csv)


# Initialize an empty list to store the filtered dataframes
filtered_dfs = []
freq = '10S'
# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    try :
        # Check if the file ends with ".science_log.csv"
        if file_name.endswith(".science_log.csv"):
            print("file_name = ", file_name)
            # Read the CSV file into a pandas dataframe
            file_path = os.path.join(folder_path, file_name)
            
            df = pd.read_csv(file_path, delimiter='\t', header=None)
    
            merged_df = pd.DataFrame()

                
            try: 
                df_LGR_CP_PTSCI = df[df.apply(lambda row: row[0].split(",")[0] == "LGR_CP_PTSCI", axis=1)]
                df_LGR_CP_PTSCI = df_LGR_CP_PTSCI[0].str.split(',', expand=True)
                df_LGR_CP_PTSCI.columns = ['Sensor_Type_LGR_CP_PTSCI','Timestamp', 'Pressure_LGR_CP_PTSCI', 'Temperature_LGR_CP_PTSCI', 'Salinity_LGR_CP_PTSCI', 'Conductivity_LGR_CP_PTSCI', 'Internal_conductivity_temperature_LGR_CP_PTSCI', 'Samples_LGR_CP_PTSCI']
                df_LGR_CP_PTSCI["Timestamp"] = pd.to_datetime(df_LGR_CP_PTSCI["Timestamp"], format="%Y%m%dT%H%M%S")
                df_LGR_CP_PTSCI = df_LGR_CP_PTSCI.set_index('Timestamp')
                df_LGR_CP_PTSCI = df_LGR_CP_PTSCI.resample(freq).ffill()
                df_LGR_CP_PTSCI = df_LGR_CP_PTSCI.reset_index(drop=False)
                # Assuming your DataFrame is called df_LGR_CP_PTSCI
                columns_to_convert = ['Pressure_LGR_CP_PTSCI', 'Temperature_LGR_CP_PTSCI', 'Salinity_LGR_CP_PTSCI', 'Conductivity_LGR_CP_PTSCI', 'Internal_conductivity_temperature_LGR_CP_PTSCI', 'Samples_LGR_CP_PTSCI']
                
                for column in columns_to_convert:
                    df_LGR_CP_PTSCI[column] = pd.to_numeric(df_LGR_CP_PTSCI[column], errors='coerce')
                print("Yes LGR_CP_PTSCI")
            except Exception as e:
                print("NO LGR_CP_PTSCI - Error:", e)
                
                
            try : 
                df_GPS = df[df.apply(lambda row: row[0].split(",")[0] == "GPS", axis=1)]
                df_GPS = df_GPS[0].str.split(',', expand=True)
                df_GPS.columns = ['Sensor_Type_GPS', "Timestamp", 'Latitude', 'Longitude', 'Satellites']
                df_GPS["Timestamp"] = pd.to_datetime(df_GPS["Timestamp"], format="%Y%m%dT%H%M%S")
                df_GPS = df_GPS.set_index('Timestamp')
                df_GPS = df_GPS.resample(freq).ffill()
                df_GPS = df_GPS.reset_index(drop=False)
                
                # Set the data types for each column
                data_types = {
                    'Sensor_Type_GPS': str,
                    'Latitude': float,
                    'Longitude': float,
                    'Satellites': int
                }
                df_GPS['Satellites'] = df_GPS['Satellites'].fillna(0)
                df_GPS = df_GPS.astype(data_types)
                

                print("YES GPS")
            except Exception as e:
                print("NO GPS - Error:", e)  
                
            try : 
                df_ID = (df[df.apply(lambda row: row[0].split(",")[2].startswith("Float ID"), axis=1)])
                df_ID = df_ID[0].str.split(',', expand=True)
                df_ID.columns = ['Message', "Timestamp", 'ID']
                float_ID = df_ID['ID'].iloc[0]
                
                data_types_ID = {
                'Message': str,
                'ID': str  # Assuming 'ID' is a numeric value and should be converted to float
                }
                df_ID = df_ID.astype(data_types_ID)
                print("YES ID")
            except Exception as e:
                print("NO ID - Error:", e)
                # continue
            
            # try: 
            #     df_LGR_PTSCI = df[df.apply(lambda row: row[0].split(",")[0] == "LGR_PTSCI", axis=1)]
            #     df_LGR_PTSCI = df_LGR_PTSCI[0].str.split(',', expand=True)
            #     df_LGR_PTSCI.columns = ['Sensor_Type_LGR_PTSCI','Timestamp', 'Pressure_LGR_PTSCI', 'Temperature_LGR_PTSCI', 'Salinity_LGR_PTSCI', 'Conductivity_LGR_PTSCI', 'Internal_conductivity_temperature_LGR_PTSCI']
            #     df_LGR_PTSCI["Timestamp"] = pd.to_datetime(df_LGR_PTSCI["Timestamp"], format="%Y%m%dT%H%M%S")
            #     df_LGR_PTSCI = df_LGR_PTSCI.set_index('Timestamp')
            #     df_LGR_PTSCI = df_LGR_PTSCI.resample(freq).ffill()
            #     df_LGR_PTSCI = df_LGR_PTSCI.reset_index(drop=False)
            #     print("Yes LGR_PTSCI")
            # except Exception as e:
            #     print("NO LGR_PTSCI - Error:", e)
            
            # try: 
            #     df_LGR_P = df[df.apply(lambda row: row[0].split(",")[0] == "LGR_P", axis=1)]
            #     df_LGR_P = df_LGR_P[0].str.split(',', expand=True)
            #     df_LGR_P.columns = ['Sensor_Type_LGR_P','Timestamp', 'Pressure_LGR_P']
            #     df_LGR_P["Timestamp"] = pd.to_datetime(df_LGR_P["Timestamp"], format="%Y%m%dT%H%M%S")
            #     df_LGR_P = df_LGR_P.set_index('Timestamp')
            #     df_LGR_P = df_LGR_P.resample(freq).ffill()
            #     df_LGR_P = df_LGR_P.reset_index(drop=False)
            #     print("Yes LGR_P")
            # except Exception as e:
            #     print("NO LGR_P - Error:", e)
                


    
            # Define your DataFrames df_CP, df_O2, df_PTS, df_FLBB, df_GPS here
            try:
                # Merge df_CP and df_O2 on 'Timestamp' using 'outer' join
                df_temp = pd.merge(df_LGR_CP_PTSCI, df_GPS, on='Timestamp', how='outer')
                if not merged_df.empty:
                    merged_df = pd.merge(merged_df, df_temp, on='Timestamp', how='outer')
                    print("Merged CP_PTSCI, GPS, and existing merged data.")
                else:
                    merged_df = df_temp
                    print("Merged CP_PTSCI and GPS.")
            except Exception as e:
                print("Error while merging CP_PTSCI and GPS:", e)


       
            # try:
            #     # Merge merged_df and df_PTS on 'Timestamp' using 'outer' join
            #     df_temp = pd.merge(merged_df, df_LGR_PTSCI, on='Timestamp', how='outer')
            #     if not merged_df.empty:
            #         merged_df = df_temp
            #         print("Merged df_ID and existing merged data.")
            #     else:
            #         merged_df = df_temp
            #         print("Merged df_LGR_PTSCI.")
            # except Exception as e:
            #     print("Error while merging df_LGR_PTSCI:", e)
                

            # try:
            #     # Merge merged_df and df_FLBB on 'Timestamp' using 'outer' join
            #     df_temp = pd.merge(merged_df, df_LGR_P, on='Timestamp', how='outer')
            #     if not merged_df.empty:
            #         merged_df = df_temp
            #         print("Merged LGR_P and existing merged data.")
            #     else:
            #         merged_df = df_temp
            #         print("Merged LGR_P.")
            # except Exception as e:
            #     print("Error while merging LGR_P:", e)
                

            try : 
                first_non_nan_index_Longitude = merged_df['Longitude'].first_valid_index()
                longitude = merged_df.at[first_non_nan_index_Longitude, 'Longitude']
                first_non_nan_index_Latitude = merged_df['Latitude'].first_valid_index()
                latitude = merged_df.at[first_non_nan_index_Latitude, 'Latitude'] 
                
                # # Create new columns with the first non-zero values
                merged_df['Latitudes_fixed'] = latitude
                merged_df['Longituded_fixed'] = longitude
                # print(2)
                
            except :

                continue
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
            # Calculate temperature range while ignoring NaNs
            T = [np.nanmin(merged_df.Temperature_LGR_CP_PTSCI), np.nanmax(merged_df.Temperature_LGR_CP_PTSCI)]
            # Calculate salinity range while ignoring NaNs
            S = [np.nanmin(merged_df.Salinity_LGR_CP_PTSCI), np.nanmax(merged_df.Salinity_LGR_CP_PTSCI)]

            # T = [0,3]
            # S = [32,33]
            fontsize = 15
            try :
                # Plot 1: Temperature Vs depth
                ax1.plot(merged_df.Temperature_LGR_CP_PTSCI, merged_df.Pressure_LGR_CP_PTSCI)
                ax1.set_xlabel("Temperature (°C)", fontsize=fontsize)
                ax1.set_ylabel("Depth (m)", fontsize=fontsize)
                ax1.set_ylim([merged_df.Pressure_LGR_CP_PTSCI.min(), merged_df.Pressure_LGR_CP_PTSCI.max()]) # set y limit
                ax1.set_xlim([T[0],T[1]]) # set y limit
                ax1.invert_yaxis()  # Invert y-axis
                ax1.tick_params(axis='both', which='major', labelsize=fontsize)
            except :
                print("No Temperature Data")
                # continue
    
            try: 
                # Plot 2: Salinity_CP_PTSCI Vs Depth
                ax2.plot(merged_df.Salinity_LGR_CP_PTSCI, merged_df.Pressure_LGR_CP_PTSCI)
                ax2.set_xlabel("Salinity [PSU]", fontsize=fontsize)
                ax2.set_ylabel("Depth (m)", fontsize=fontsize)
                ax2.set_ylim([merged_df.Pressure_LGR_CP_PTSCI.min(), merged_df.Pressure_LGR_CP_PTSCI.max()]) # set y limit
                ax2.set_xlim([S[0],S[1]]) # set y limit
                ax2.invert_yaxis()  # Invert y-axis
                ax2.tick_params(axis='both', which='major', labelsize=fontsize)
            except :
                print("No Salinity_CP_PTSCI Data")
                # continue
    

            
            # Add main title
            begin_date = merged_df["Timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
            end_date = merged_df["Timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S") 
            title = f"CTD profile \n {begin_date} - {end_date} \n {float_ID} \n Coordinates : {latitude}\u00B0 {longitude}\u00B0"
            fig.suptitle(title, fontsize=15, fontweight='bold')
            
            float_ID = float_ID.replace(' ', '_')
            float_ID = float_ID.replace(':', '')
            folder_name = "CTD_Plots"
    
            # Define the full path to the folder
            base_path = r"C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test"
        
            # Combine the base path with the folder name
            full_folder_path = os.path.join(base_path, folder_name)
    
            if not os.path.exists(full_folder_path):
                print("slkslkslkslk")
                os.makedirs(full_folder_path)
            
            
            # Save the figure as a jpg file
            begin_date =  datetime.strptime(begin_date, '%Y-%m-%d %H:%M:%S')
            begin_date = begin_date.strftime('%Y-%m-%d')
            file_name = f"{float_ID}_{begin_date}.jpg"
            file_path = os.path.join(ctd_plots_folder, file_name)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            
            # Save the DataFrame as a CSV file
            file_name = f"{float_ID}_{begin_date}.csv"
            file_path = os.path.join(to_upload_folder, file_name)
            merged_df = merged_df.sort_values(by='Timestamp')
            # merged_df = merged_df.dropna()
            merged_df['float_id'] = float_ID
            # Convert the "float id" column to string data type
            merged_df['float_id'] = merged_df['float_id'].astype(str)
            merged_df.to_csv(file_path, index=False, sep=',')
            

    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        continue
        
        
        

