# import os

# # Get the path to the user's desktop folder
# desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# # Define the file name and message
# file_name = "task_output.txt"
# message = "Task ran well"

# # Create the full file path
# file_path = os.path.join(desktop_path, file_name)

# # Write the message to the file
# with open(file_path, "w") as file:
#     file.write(message)

# print(f"File '{file_name}' with message '{message}' created on the desktop.")

#%%


import subprocess

# Define the paths to your two scripts
F10051_path = r"C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\Python_Scripts\F10051_Regular_TSP\SubProcess.py"
F10052_path = r"C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\Python_Scripts\F10052\SubProcess.py"
F10053_path = r"C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\Python_Scripts\F10053_Biology_Float\SubProcess.py"
    
try:
    # Run script1 as a subprocess
    subprocess.run(['python', F10051_path], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running script1: {e}")
except Exception as e:
    print(f"An unexpected error occurred while running script1: {e}")

try:
    # Run script2 as a subprocess after script1 completes
    subprocess.run(['python', F10052_path], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running script2: {e}")
except Exception as e:
    print(f"An unexpected error occurred while running script2: {e}")

try:
    # Run script3 as a subprocess after script1 completes
    subprocess.run(['python', F10053_path], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running script3: {e}")
except Exception as e:
    print(f"An unexpected error occurred while running script3: {e}")


