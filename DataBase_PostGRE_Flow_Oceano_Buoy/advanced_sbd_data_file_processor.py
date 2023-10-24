import os
import glob
import struct
import shutil
import io
import builtins
import sys

SBD_NEW_FILE_START_CHAR = b'\x01'.decode('utf-8')
SBD_CONT_FILE_START_CHAR = b'\x02'.decode('utf-8')

# base_folder = r'C:\Users\thiba\OneDrive\Documents\Float\F10052'
base_folder = r'C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052\DATA_Received'
# reception = r'C:\Users\thiba\OneDrive\Documents\Float'
reception = r'C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052'

def sbd2file(datapath,folder_name):
    """
    This function does all the work 
    try:
        pass
    except Exception:
        passhat was described in the file documentation comments.
    """
    # If no archive directory, make one
    arcdir = os.path.join(reception, 'F10052_Processed\\'+ str(folder_name))
    if not os.path.isdir(arcdir):
        os.makedirs(arcdir)
    num_msgs_to_process = 0
    output_filename_path = None
    msgfns = []
    prevmsg = ''
    output_file = None
    
    # Glob the files in the folder and sort the file list.  This ensures that files are 
    # processed in the order they were sent.
    sbd_files_to_process = sorted(glob.glob(os.path.join(datapath, '*.sbd')))
    print("# of SBD files to process is %d" % (len(sbd_files_to_process)))

    for sbd_filename_path in sbd_files_to_process:
        
        # Open the file for reading binary
        sbd_data_file_contents = io.open(sbd_filename_path, 'rb').read()

        first_char = sbd_data_file_contents[0]
        if isinstance(first_char, int):
            first_char = builtins.chr(first_char)
            

        # File name is <imei>_<momsn>.sbd, get the momsn from it.
        fnseq = sbd_filename_path[sbd_filename_path.rfind('_') + 1 : sbd_filename_path.rfind('.')]
        try:
            seq = int(fnseq)
        except ValueError:
            print('Invalid file name', sbd_filename_path)
            continue
        # First byte of message indicates start message or continuation
        if first_char == SBD_NEW_FILE_START_CHAR:
            print(output_filename_path, num_msgs_to_process, "Yoyo")

            # It's a start of file, check that existing file is finished
            if (output_filename_path != None) and (num_msgs_to_process != 0):  #Here i changed (output_filename_path != 0) into (output_filename_path != None) 
                print('file %s is incomplete' % output_filename_path)
            # File name is null terminated string after type and length
            nullpos = sbd_data_file_contents.find(b'\0', 3)
            # Data starts after the null
            output_filename_path = sbd_data_file_contents[3 : nullpos].decode('utf-8')
            # Number of messages is 2 byte binary after type
            num_msgs_to_process = struct.unpack('>H', sbd_data_file_contents[1:3])[0]
            if num_msgs_to_process <= 0:
                print("Error: # of messages reported in SBD new file header is %d" % (num_msgs_to_process))
                continue
            # Start writing data
            output_file = open(os.path.join(datapath, output_filename_path), 'wb')
            
            if output_file is not None:
                output_file.write(sbd_data_file_contents[nullpos + 1:])
                print('message %d starts file %s, %d messages' % (seq, output_filename_path, num_msgs_to_process))
                msgfns = [sbd_filename_path]
                print(num_msgs_to_process)
                num_msgs_to_process -= 1
            else:
                print("Could not create file %s" % (output_filename_path))
                sys.exit(2)
        else:
            
            # It's a continuation, check for possible problems
            if first_char != SBD_CONT_FILE_START_CHAR:
                # Wrong type
                print('Unknown message type %d in file %s' % (sbd_data_file_contents[0], sbd_filename_path))
                continue
            if sbd_data_file_contents == prevmsg:
                print('Got retry in file:', sbd_filename_path)
                continue
            elif num_msgs_to_process <= 0:
                # Extra junk at end
                print('extraneous message ignored:', sbd_filename_path)
            else:
                # Nope, all good
                prevmsg = sbd_data_file_contents
                
                # Let's make sure we have an output_file that opened as a result of finding an 
                # SBD new file message
                if output_file is not None:
                    output_file.write(sbd_data_file_contents[1:])
                    msgfns.append(sbd_filename_path)
                    num_msgs_to_process -= 1

        if num_msgs_to_process <= 0 and output_file is not None:
            print("Yoyo3")
            # A file is done, close it and report
            output_file.close()
            # Clear the output_file now that we finished creating the original file
            output_file = None
            print('saved file %s' % output_filename_path)
            # Archive the component message files
            for arcfn in msgfns:
                shutil.move(arcfn, arcdir)
            msgfns = []

def process_all_folders(base_folder):
    """
    This function processes all subfolders inside the base_folder.
    """
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            print(folder_name,folder_path)
            print(f"Processing data in folder: {folder_name}")
            sbd2file(folder_path, folder_name)

if __name__ == '__main__':
    process_all_folders(base_folder)
