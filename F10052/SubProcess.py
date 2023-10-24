import subprocess

# Define the base path
base_path = r"C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\Python_Scripts\F10052"

def run_script_with_output(script_name):
    result = subprocess.run(["python", script_name], capture_output=True, text=True, cwd=base_path)
    return result.stdout, result.stderr

# Run advanced_sbd_data_file_processor.py
stdout_script1, stderr_script1 = run_script_with_output("advanced_sbd_data_file_processor.py")
print("Output of advanced_sbd_data_file_processor.py:")
print(stdout_script1)
#%%
# Run advanced_unzipper.py
stdout_script2, stderr_script2 = run_script_with_output("advanced_unzipper.py")
print("Output of advanced_unzipper.py:")
print(stdout_script2)
print("Error of advanced_unzipper.py:")
print(stderr_script2)
#%%
# Run apf11dec.py
stdout_script3, stderr_script3 = run_script_with_output("apf11dec.py")
print("Output of apf11dec.py:")
print(stdout_script3)
print("Error of apf11dec.py:")
print(stderr_script3)
#%%
# Run only_keep_csv.py
stdout_script4, stderr_script4 = run_script_with_output("only_keep_csv.py")
print("Output of only_keep_csv.py:")
print(stdout_script4)
print("Error of only_keep_csv.py:")
print(stderr_script4)

#%%
# Run only_keep_csv.py
stdout_script41, stderr_script41 = run_script_with_output("Sensors_list_and_Count.py")
print("Output of Sensors_list_and_Count.py:")
print(stdout_script41)
print("Error of Sensors_list_and_Count.py:")
print(stderr_script41)

#%%
# Run Advanced_CTD_Plotter.py
stdout_script, stderr_script = run_script_with_output("Advanced_CTD_Plotter.py")
print("Output of Advanced_CTD_Plotter.py:")
print(stdout_script)
print("Error of Advanced_CTD_Plotter.py:")
print(stderr_script)
#%%
# Run Create_Empty_Table_Apex.py
stdout_script5, stderr_script5 = run_script_with_output("Create_Empty_Table_Apex.py")
print("Output of Create_Empty_Table_F10051.py:")
print(stdout_script5)
print("Error of Create_Empty_Table_F10051.py:")
print(stderr_script5)

#%%
# Run Remove_Duplicates
stdout_script6, stderr_script6 = run_script_with_output("Remove_Duplicate_To_Upload_Data_Uploaded.py")
print("Output of Remove_Duplicate_To_Upload_Data_Uploaded.py:")
print(stdout_script6)
print("Error of Remove_Duplicate_To_Upload_Data_Uploaded.py:")
print(stderr_script6)

#%%
# Run Upload_To_Apex_No_Duplicate.py
stdout_script6, stderr_script6 = run_script_with_output("Upload_To_Apex_No_Duplicate.py")
print("Output of Upload_To_Apex_No_Duplicate.py:")
print(stdout_script6)
print("Error of Upload_To_Apex_No_Duplicate.py:")
print(stderr_script6)
