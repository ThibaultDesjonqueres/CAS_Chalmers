# Copyright 2020 Teledyne Webb Research
#
# The information contained herein is the property of Teledyne Webb Research
# and is considered proprietary. This information may not be used for any
# other purpose, copied, transferred or disclosed to third parties, reverse
# engineered, modified or improved without written consent from Teledyne
# Webb Research.
#
'''
@organization: Teledyne Webb Resarch
@file: apf11dec.py

@note: Change Global when changing the version

@summary: convert an APF11 binary data file to .csv

Revision History:
@change: 12-Nov-2012 dpingal@teledyne.com Initial
@change: 16-Nov-2012 dpingal@teledyne.com Added 'z' format for null terminated strings
@change: 24-Jan-2013 dpingal@teledyne.com Catches exception, prints out info about bad input.  Added debug option.
@change: 25-Feb-2013 dpingal@teledyne.com Fixed timestamp date format
@change: 27-Dec-2014 lbovie@teledyne.com Added RBR log items PTSC & PTSC_BINDATA
@change: 13-Feb-2016 Brian.Leslie@teledyne.com Updated record ids
@change: 13-Feb-2016 Brian.leslie@teledyne.com Fixed MANTIS 3482
@change: 06-Sep-2018 Brian.Leslie@teledyne.com Removed Seascan support and added RBR LGR
@change: 11-Feb-2020 Brian.Leslie@teledyne.com refactored decoder without id to support EMA and IRAD logs

@todo: verify data decimal precision matches decoder output
'''

import os
import struct
import time

# Here, indicate the path with your .bin data, to convert it into .csv 
cwd = os.getcwd()
# dataFolder_bin_to_Csv =  r"C:\Users\thiba\OneDrive\Documents\Float\F10052\unzipped"
dataFolder_bin_to_Csv =  r"C:\Users\thiba\OneDrive - University of Gothenburg\FLOAT_Test\FLOATS_List\F10052\DATA_Received\unzipped"

folder_path = os.path.join(cwd, dataFolder_bin_to_Csv)

debug = False
__version__="2.16.1"

WITH_ID = True
WITHOUT_ID = False

def get_byte(byte):
    try:
        # python 2
        byte = ord(byte)
    except:
        # python 3
        byte = byte
    return byte

def dump(bin):
    result = ''
    for byte in bin:
        result += '%02x ' % get_byte(byte)
    return result[:-1]

def unpack_with_final_asciiz(fmt, dat):
    """
    Unpack binary data, handling a null-terminated string at the end
    (and only at the end) automatically.

    The first argument, fmt, is a struct.unpack() format string with the
    following modfications:
    If fmt's last character is 'z', the returned string will drop the NUL.
    """
    # Just pass on if no special behavior is required
    if fmt[-1] != 'z':
        return struct.unpack(fmt, dat)

    # Use format string to get size of contained string and rest of record
    str_len = len(dat) - struct.calcsize(fmt[:-1])
    new_fmt = '%s%ds' % (fmt[:-1], str_len)
    record = struct.unpack(new_fmt, dat)
    # decode message binary data
    new_record = (record[0], record[1].decode())
    return new_record

# Convert Unix timestamp to a human readable string
def ts_str(ts):
    return time.strftime('%Y%m%dT%H%M%S', time.gmtime(ts))

# Note that format specifier appears to round, not truncate
def f0_str(val):
    return '%0.0f' % val

def f1_str(val):
    return '%0.1f' % val

def f2_str(val):
     return '%0.2f' % val

def f3_str(val):
     return '%0.3f' % val

def f4_str(val):
     return '%0.4f' % val

def f5_str(val):
     return '%0.5f' % val
 
def f6_str(val):
     return '%0.6f' % val

# Instances of this class represent a single record type
class datarecord(object):
    # Dictionaries of custom formatters:
    # formats = what's the underlying type (must be valid for struct.unpack)
    # processors = functions to call after converting
    # e.g. timestamp is a long int, gets converted to string by ts_str
    # also, we can handle a trailing null-terminated string
    formats = { 'T': 'I', '0':'f', '1':'f', '2':'f', '3':'f', '4':'f', '5':'f', '6':'f'}
    posts = { 'T': ts_str, '0': f0_str, '1':f1_str, '2':f2_str, '3':f3_str, '4':f4_str, '5':f5_str, '6':f6_str, 'f':f5_str }

    # create a record
    def __init__(self, id, name, fmt, fields):
        self.id = id # id = binary record ID in file
        # fmt = a format string to convert type (see struct module doco)
        self.fmt = '<' + ''.join([ datarecord.formats.get(c) or c for c in fmt ])
        self.post = tuple([ datarecord.posts.get(c) for c in fmt ])
        # name = record name
        self.name = name
        # fields = names of output fields
        self.fields = fields

    # decode a binary record
    def decode(self, record):
        if self.id != None:
            id = struct.unpack('<BB', record[0:2])[0]
            if self.id != id:
                print('decode: expected ID %d, got %d' % (self.id, id))
                return None
            data = unpack_with_final_asciiz(self.fmt, record[1:])
        else:
             data = unpack_with_final_asciiz(self.fmt, record)
        result = []
        for n, value in enumerate(data):
            if self.post[n]:
                result.append(self.post[n](value))
            else:
                result.append(value)
        return result

# Instances of this class lookup the record type and deal with it appropriately
class decoder(object):
    # Empty to start
    def __init__(self, with_id):
        self.records = {}
        self.with_id = with_id

    def __getitem__(self, id):
        return self.records[id].name

    # Add type definition to our list
    def add_record_with_id(self, id, name, fmt, fields):
        self.records[id] = datarecord(id, name, fmt, fields)

    # Add type definition to our list
    def add_record_without_id(self, fmt, fields):
        self.records = datarecord(None, None, fmt, fields)
    
    # obtains decoder record fields
    def fields(self):
        return self.records.fields

    # Decode a data record
    def decode(self, data):
        if self.with_id:
            # get ID field from record and look it up in self.records
            record_index = get_byte(data[0])
            record = self.records[record_index]
            # use that decoder to interpret the record
            result = [record.name] + record.decode(data)
        else:
            # use that decoder to interpret the record
            result = self.records.decode(data)
        return result

'''
Vitals Log Decoder
'''
# Make the decoders
vitals = decoder(WITH_ID)

# LOG_VITALS_MESSAGE
vitals.add_record_with_id(0, 'Message', 'Tz', ('timestamp', 'message'))

# ----- Core Vital Logs -----

#   LOG_VITALS_CORE
vitals.add_record_with_id(1, 'VITALS_CORE', 'T3H3H333H33h', ('timestamp', 'air_bladder(dbar)', 'air_bladder(cnts)',
                                                             'battery_voltage(V)', 'battery_voltage(cnts)',
                                                             'humidity', 'leak_detect(V)',
                                                             'vacuum(dbar)', 'vacuum(cnts)',
                                                             'coulomb(AHrs)',
                                                             'battery_current(mA)', 'battery_current_raw'))

# LOG_VITALS_IRIDIUM_CSQ
vitals.add_record_with_id(2, 'RSSI', 'TB', ('timestamp', 'RSSI'))

#LOG_VITALS_WATCHDOG_CNT
vitals.add_record_with_id(3, 'WD_CNT', 'Ti',  ('Timestamp', 'Events(count)'))

# ----- Optional Vital Logs -----

# LOG_VITALS_AIR
vitals.add_record_with_id(25, 'AIR', 'TIIfHfffffffHHfHfHBfHffHHf', ('Timestamp', 'num_updates', 'total_seconds', 'last_battery_voltage_reading_in_volts',
                                                                    'last_battery_voltage_reading_in_counts', 'last_battery_current_reading_in_milliamps',
                                                                    'last_coulomb_reading_in_mA_hr', 'total_coulomb_usage_in_mA_hr', 'battery_current_avg_in_milliamps',
                                                                    'battery_current_max_in_milliamps', 'battery_voltage_avg_in_volts', 'battery_voltage_min_in_volts',
                                                                    'battery_voltage_avg_in_counts', 'battery_voltage_min_in_counts', 'internal_vacuum_pressure_in_dbar',
                                                                    'internal_vacuum_pressure_counts', 'air_bladder_pressure_in_dbar', 'air_bladder_pressure_counts',
                                                                    'num_pulses', 'last_air_pump_current_reading_in_milliamps', 'last_air_pump_current_reading_in_counts',
                                                                    'air_pump_current_avg_in_milliamps', 'air_pump_current_max_in_milliamps', 'air_pump_current_avg_in_counts',
                                                                    'air_pump_current_max_in_counts', 'air_pump_volt_secs'))

# LOG_VITALS_BUOYANCY
vitals.add_record_with_id(26, 'BUOYANCY', 'TIIfHfffffffHHHHfHffHHIIff', ('Timestamp', 'num_updates', 'total_seconds', 'last_battery_voltage_reading_in_volts',
                                                                         'last_battery_voltage_reading_in_counts', 'last_battery_current_reading_in_milliamps',
                                                                         'last_coulomb_reading_in_mA_hr', 'total_coulomb_usage_in_mA_hr', 'battery_current_avg_in_milliamps',
                                                                         'battery_current_max_in_milliamps', 'battery_voltage_avg_in_volts', 'battery_voltage_min_in_volts',
                                                                         'battery_voltage_avg_in_counts', 'battery_voltage_min_in_counts', 'start_position_count', 'stop_position_count',
                                                                         'last_buoy_pump_current_reading_in_milliamps', 'last_buoy_pump_current_reading_in_counts',
                                                                         'buoy_pump_current_avg_in_milliamps', 'buoy_pump_current_max_in_milliamps', 'buoy_pump_current_avg_in_counts',
                                                                         'buoy_pump_current_max_in_counts', 'last_pot_change', 'total_pot_change', 'start_coulomb_in_amphrs', 'end_coulomb_in_amphrs'))

# LOG_VITALS_PRESSURE_SENSOR
vitals.add_record_with_id(27, 'PRESSURE', 'TIIfHfffffffHHfHffH', ('Timestamp', 'num_updates', 'total_seconds', 'last_battery_voltage_reading_in_volts',
                                                                  'last_battery_voltage_reading_in_counts', 'last_battery_current_reading_in_milliamps',
                                                                  'last_coulomb_reading_in_mA_hr', 'total_coulomb_usage_in_mA_hr', 'battery_current_avg_in_milliamps',
                                                                  'battery_current_max_in_milliamps', 'battery_voltage_avg_in_volts', 'battery_voltage_min_in_volts',
                                                                  'battery_voltage_avg_in_counts', 'battery_voltage_min_in_counts', 'battery_voltage_in_volts',
                                                                  'battery_voltage_in_counts', 'battery_current_in_milliamps', 'current_in_milliamps', 'current_in_counts'))

# LOG_VITALS_TELECOMM
vitals.add_record_with_id(28, 'COMMS', 'TIIfHfffffffHHI', ('Timestamp', 'num_updates', 'total_seconds', 'last_battery_voltage_reading_in_volts', 
                                                           'last_battery_voltage_reading_in_counts', 'last_battery_current_reading_in_milliamps',
                                                           'last_coulomb_reading_in_mA_hr', 'total_coulomb_usage_in_mA_hr', 'battery_current_avg_in_milliamps',
                                                           'battery_current_max_in_milliamps', 'battery_voltage_avg_in_volts', 'battery_voltage_min_in_volts',
                                                           'battery_voltage_avg_in_counts', 'battery_voltage_min_in_counts', 'total_num_bytes'))

# ----- Feature Vital Logs -----

# LOG_VITALS_ICE_DETECT
vitals.add_record_with_id(50, 'ICE_DETECT', 'Ti34i',('timestamp', 'mission','medianP', 'medianT', 'samples'))

# LOG_VITALS_ICE_CAP
vitals.add_record_with_id(51, 'ICE_CAP', 'Ti34i', ('timestamp', 'mission','medianP', 'medianT', 'samples'))

# LOG_VITALS_ICE_BREAKUP
vitals.add_record_with_id(52, 'ICE_BREAKUP', 'Ti34i', ('timestamp', 'mission','medianP', 'medianT', 'samples'))


# ----- Experimental Vital Logs -----

# None

'''
Science Log Decoder
'''
science = decoder(WITH_ID)


# ----- Core Science Logs -----

# LOG_SCIENCE_MESSAGE
science.add_record_with_id(0, 'Message', 'Tz', ('timestamp', 'message'))

# LOG_SCIENCE_GPS
science.add_record_with_id(1, 'GPS', 'T66i', ('timestamp', 'latitude', 'longitude','nsat'))


# ----- SBE CTD Sensor Science Logs -----
 
# LOG_SCIENCE_CTD_CP_BINDATA
science.add_record_with_id(10, 'CTD_bins', 'TIH3', ('timestamp', 'samples', 'bins', 'maxpress'))

# LOG_SCIENCE_CTD_P
science.add_record_with_id(11, 'CTD_P', 'T2', ('timestamp', 'pressure'))

# LOG_SCIENCE_CTD_PT
science.add_record_with_id(12, 'CTD_PT', 'T24', ('timestamp', 'pressure', 'temperature'))

# LOG_SCIENCE_CTD_PTS
science.add_record_with_id(13, 'CTD_PTS', 'T244', ('timestamp', 'pressure', 'temperature', 'salinity'))

# LOG_SCIENCE_CTD_CP_PTS
science.add_record_with_id(14, 'CTD_CP', 'T244h', ('timestamp', 'pressure', 'temperature', 'salinity', 'samples'))

# LOG_SCIENCE_CTD_PTSH
science.add_record_with_id(15, 'CTD_PTSH', 'T2446', ('timestamp', 'pressure', 'temperature', 'salinity', 'ph'))

# LOG_SCIENCE_CTD_CP_PTSH
science.add_record_with_id(16, 'CTD_CP_H', 'T244h6h', ('timestamp', 'pressure', 'temperature', 'salinity', 'samples', 'ph', 'samples'))


# ----- RBR LGR Sensors Science Logs -----

# LOG_SCIENCE_RBR_LGR_PTSCI
science.add_record_with_id(20, 'LGR_PTSCI', 'Tfffff', ('timestamp','pressure','temperature', 'salinity', 'conductivity', 'internal_temperature'))

# LOG_SCIENCE_RBR_LGR_CP_PTSCI
science.add_record_with_id(21, 'LGR_CP_PTSCI', 'Tfffffh',  ('timestamp', 'pressure', 'temperature', 'salinity', 'conductivity', 'internal_temperature', 'samples'))

# LOG_SCIENCE_RBR_LGR_CP_PT
science.add_record_with_id(22, 'LGR_CP_PT', 'Tffh',  ('timestamp', 'pressure', 'temperature', 'samples'))

# LOG_SCIENCE_RBR_LGR_P
science.add_record_with_id(23, 'LGR_P', 'Tf', ('timestamp', 'pressure'))

# LOG_SCIENCE_RBR_LGR_PT
science.add_record_with_id(24, 'LGR_PT', 'Tff', ('timestamp', 'pressure', 'temperature'))

# LOG_SCIENCE_RBR_LGR_PTS
science.add_record_with_id(25, 'LGR_PTS', 'Tfff', ('timestamp', 'pressure', 'temperature', 'salinity'))

# LOG_SCIENCE_RBR_LGR_PTSC
science.add_record_with_id(26, 'LGR_PTSC', 'Tffff', ('timestamp', 'pressure', 'temperature', 'salinity', 'conductivity'))


# ----- Optode Sensor Science Logs -----

# LOG_SCIENCE_OPTODE
science.add_record_with_id(40, 'O2', 'Tffffffffff', ('timestamp', 'O2', 'AirSat', 'Temp', 'CalPhase', 'TCPhase', 'C1RPh', 'C2RPh', 'C1Amp', 'C2Amp', 'RawTemp'))


# ----- FLBB Sensor Science Logs -----

# LOG_SCIENCE_FLBB
science.add_record_with_id(50, 'FLBB', 'Thhh', ('timestamp', 'chl_sig', 'bsc_sig', 'therm_sig'))

# LOG_SCIENCE_FLBB_BB
science.add_record_with_id(51, 'FLBB_BB', 'Thhhh', ('timestamp', 'chl_sig', 'bsc_sig0', 'bsc_sig1','therm_sig'))

# LOG_SCIENCE_FLBB_CD
science.add_record_with_id(52, 'FLBB_CD', 'Thhhh', ('timestamp', 'chl_sig', 'bcs_sig', 'cd_sig', 'therm_sig'))

# LOG_SCIENCE_FLBB_CFG
science.add_record_with_id(53, 'FLBB_CFG', 'Thh', ('timestamp', 'chl_wave', 'bsc_wave'))

# LOG_SCIENCE_FLBB_BB_CFG
science.add_record_with_id(54, 'FLBB_BB_CFG', 'Thhh', ('timestamp', 'chl_wave', 'bsc_wave0', 'bsc_wave1'))

# LOG_SCIENCE_FLBB_CD_CFG
science.add_record_with_id(55, 'FLBB_CD_CFG', 'Thhh', ('timestamp', 'chl_wave', 'bsc_wave', 'cd_wave'))

# ----- Radiance Sensor Science Logs -----

 # LOG_SCIENCE_RAD
science.add_record_with_id(60, '504R', 'Tffff', ('timestamp', 'channel1',  'channel2',  'channel3',  'channel4'))

# LOG_SCIENCE_IRAD
science.add_record_with_id(61, '504I', 'Tffff', ('timestamp', 'channel1',  'channel2',  'channel3', 'channel4'))


# ----- Crover Sensor Science Logs -----

# LOG_SCIENCE_CROVER
science.add_record_with_id(70, 'CROVER', 'Thhhhf', ('timestamp', 'reference', 'raw_sig', 'corr_sig', 'therm', 'attenuation(m^-1)'))

# ----- Attitude/Compass Sensor Science Logs -----

# LOG_SCIENCE_COMPASS
science.add_record_with_id(80, 'Compass', 'Tffff', ('timestamp', 'heading', 'pitch', 'roll', 'dip'))


# ----- JFE Advantech's RINKO-FT sensor Science Logs -----

# LOG_SCIENCE_RINKO_FT
science.add_record_with_id(90, 'O2', 'THHHHHHI', ('timestamp', 'temperature', 'dissolved_oxygen', 'blue_phase', 'red_phase', 'blue_amplitude', 'red_amplitude', 'accumulated_led_time'))

# ----- Satlantic SUNA (Deep) Sensor Science Logs -----

# LOG_SCIENCE_SUNA

science.add_record_with_id(100, 'NO3', 'T2', ('timestamp', 'nitrate'))

# ----- Seascan RTC Science Logs -----

# LOG_SCIENCE_SEASCAN_RTC

science.add_record_with_id(110, 'SRTC', 'TI', ('timestamp', 'rtc_time'))

# ----- Seascan RAFOS Science Logs -----

# LOG_SCIENCE_SEASCAN_RAFOS

science.add_record_with_id(115, 'RAFOS', 'TBHBHBHBHBHBH', ('timestamp', 'correlation1', 'sample1', 'correlation2', 'sample2', 'correlation3', 'sample3', 'correlation4', 'sample4', 'correlation5', 'sample5', 'correlation6', 'sample6'))

# ----- APLUW EMA Sensor Science Logs -----

# LOG_SCIENCE_EMA

science.add_record_with_id(120, 'EMA', 'Tffffffffffffffffffffff', ('timestamp', 'rotp_hx', 'rotp_hy', 'e1_coef40', 'e1_coef41', 'e2_coef40', 'e2_coef41',
                                                        'e1_mean4', 'e2_mean4', 'e1_sdev4', 'e2_sdev4', 'buoy_pos_c0', 'hxdt_sdev', 'hydt_sdev',
                                                        'bt_mean', 'hz_mean', 'ax_mean', 'ax_sdev', 'ay_mean', 'ay_sdev', 'az_mean', 'hx_mean', 'hy_mean'))

# LOG_SCIENCE_EMA_CFG

science.add_record_with_id(121, 'EMA_CFG', 'TBBfB', ('timestamp', 'ReqSamples', 'SlideSamples', 'EMALogMaxPressure', 'EMALogMissionCycle'))

# ----- TriOS RAMSES Science Logs -----

# LOG_SCIENCE_TRIOS_RAMSES

science.add_record_with_id(125, 'IRAD', 'THffff', ('timestamp', 'integration_time', 'temperature', 'pressure', 'pre_inclination', 'post_inclination'))

'''
EMA Log Decoder
'''
ema = decoder(WITHOUT_ID)

ema.add_record_without_id('THBHHBBBBBBBBffffffffHHHHHHHHBdlBdlH', ('timestamp', 'buoy_pos', 'age', 'overflow', 'seqno',
                                                     'ZR_nsum', 'BT_nsum', 'Hz_nsum', 'Hy_nsum', 'Hx_nsum', 'Az_nsum', 'Ay_nsum', 'Ax_nsum', 
                                                     'ZR_mean', 'BT_mean', 'Hz_mean', 'Hy_mean', 'Hx_mean', 'Az_mean', 'Ay_mean', 'Ax_mean',
                                                     'ZR_pp', 'BT_pp', 'Hz_pp', 'Hy_pp', 'Hx_pp', 'Az_pp', 'Ay_pp', 'Ax_pp',
                                                     'E1_nsum', 'E1_mean', 'E1_pp',
                                                     'E2_nsum', 'E2_mean', 'E2_pp',
                                                     'CRC'))

'''
IRAD Log Decoder
'''
irad = decoder(WITHOUT_ID)
irad_rec_len = 514
irad.add_record_without_id('THHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH',
             ['timestamp']) # rest of data is 255 raw ordinate values and do not need a header

EXPERIMENTAL = range(225,255)
EXCLUDE_EXPERIMENTAL=False
# Convert a .bin file to .csv
def decode(fn):
    ifile = open(fn, 'rb')
    ofile = open(os.path.splitext(fn)[0] + '.csv', 'w')
    
    # print heading
    if fn.find("ema_log") >= 0:
        ofile.write(','.join([str(x) for x in ema.fields()]) + '\n')

    while True:
        if fn.find("irad_log") >= 0:
            rec_len = irad_rec_len
            rec = ifile.read(rec_len)
            if len(rec) == 0:
                # no more records to process
                break
        else:
            len_chr = ifile.read(1)
            if len(len_chr) < 1:
                # no more records to process
                break
    
            rec_len = ord(len_chr)
            rec = ifile.read(rec_len)
            record_id = get_byte(rec[0])
            if record_id in EXPERIMENTAL:
                if EXCLUDE_EXPERIMENTAL:
                    continue
        try:
            # handle decoding based on file name
            if fn.find("science_log") >= 0:
                ofile.write(','.join([str(x) for x in science.decode(rec)]) + '\n')
                if debug:
                    print('at %d len %d record_id %d %s' % (ifile.tell(), rec_len, record_id, science[record_id]))
            elif fn.find("vitals_log") >= 0:
                ofile.write(','.join([str(x) for x in vitals.decode(rec)]) + '\n')
                if debug:
                    print('at %d len %d record_id %d %s' % (ifile.tell(), rec_len, record_id, vitals[record_id]))
            elif fn.find("ema_log") >= 0:
                ofile.write(','.join([str(x) for x in ema.decode(rec)]) + '\n')
                if debug:
                    print('at %d len %d' % (ifile.tell(), rec_len))
            elif fn.find("irad_log") >= 0:
                ofile.write(','.join([str(x) for x in irad.decode(rec)]) + '\n')
                if debug:
                    print('at %d len %d' % (ifile.tell(), irad_rec_len))
            if debug:
                print(dump(rec[1:]))
        except:
            if fn.find("science_log") >= 0:
                print('ERROR at %d len %d record_id %d %s' % (ifile.tell(), rec_len, record_id, science[record_id]))
            elif fn.find("vitals_log") >= 0:
                print('ERROR at %d len %d record_id %d %s' % (ifile.tell(), rec_len, record_id, vitals[record_id]))
            elif fn.find("ema_log") >= 0 or fn.find("irad_log") >= 0:
                print('ERROR at %d len %d' % (ifile.tell(), rec_len))
            print('- can\'t interpret:', dump(rec[1:]))
        if debug:  
            print('---------------------------------------------------')
            


if __name__ == '__main__':
    import sys, getopt
    
    file_extension = ".bin"  # Change this to the file extension you want to filter on
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]

    if len(files) == 0:
        print(f"No files found in {folder_path} with extension {file_extension}")
        sys.exit(-1)

    for fn in files:
        decode(os.path.join(folder_path, fn))
        # os.chmod(folder_path, 0o777)
        
