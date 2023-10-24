import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Specify the directory to monitor
directory_to_watch = r"C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10051\DATA_Received"

# Specify the Python script to run when a new folder is created
script_to_run = r"C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\Python_Scripts\F10051_Regular_TSP\SubProcess.py"

class FolderCreationHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            print(f"New folder created: {event.src_path}")
            # Execute the specified script when a new folder is created
            subprocess.Popen(['python', script_to_run])

if __name__ == "__main__":
    # Create an observer to watch the specified directory
    observer = Observer()
    event_handler = FolderCreationHandler()
    observer.schedule(event_handler, path=directory_to_watch, recursive=False)

    print(f"Monitoring directory: {directory_to_watch}")

    # Start monitoring
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
