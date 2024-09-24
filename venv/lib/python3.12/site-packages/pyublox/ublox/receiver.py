import os
import os.path
import shutil
import struct
import tempfile
import time
import threading
import sys
import glob

import serial

from roktools import logger, rinex
from roktools.time import weektow_to_datetime

from . import helpers
from . import config as ublox_config
from . import core
from . import SERIAL_PORT_STR, BAUD_RATE_STR, DEFAULT_BAUD_RATE


# ------------------------------------------------------------------------------

class Receiver(threading.Thread):

    def __init__(self, serial_port=None, baudrate=None, file_rotation=rinex.FilePeriod.DAILY,
                 output_dir=".", receiver_name="UBLX", detect_messages=False):
        threading.Thread.__init__(self)

        self.slice = file_rotation

        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.connection_config = self.detect_connection(serial_port, baudrate)
        self.serial_port = self.connection_config.get(SERIAL_PORT_STR, None)
        self.baudrate = self.connection_config.get(BAUD_RATE_STR, None)
        self.serial_stream_for_recording = None

        logger.info('Connection with a u-blox receiver detected at port [ {} ] and baudrate [ {} ]'.format(self.serial_port, self.baudrate))

        self.receiver_config = None
        with serial.Serial(self.serial_port, self.baudrate) as serial_stream:
            logger.info(f'Opened serial connection [ {self.serial_port} ] with the receiver')
            self.receiver_config = ublox_config.get(serial_stream)
            if detect_messages:
                self.receiver_config['messages'] = __detect_messages__(serial_stream, timeout=core.TIMEOUT_DEF)

         
        self.receiver_name = receiver_name
        self.current_epoch_suffix = None
        self.fout = None

        self.do_recording = threading.Event()

    # ---

    def detect_connection(self, serial_port=None, baudrate=None):
        self.connection_config = detect_connection(serial_port, baudrate)
        return self.connection_config

    def get_connection(self):
        return self.connection_config

    # ---    

    def submit_configuration(self, doc):
        return self.__submit_configuration__(doc=doc)

    def submit_configuration_from_ucenter_file(self, config_file):
        return self.__submit_configuration__(ucenter_file=config_file)


    def __submit_configuration__(self, doc=None, ucenter_file=None):
        """
        Submit a configuration (using the Receiver configuration standard in JSON
        format)
        """

        receiver_was_recording = self.is_recording_ongoing()
        if receiver_was_recording:
            self.stop() 
            logger.warning("Stop recording session in order to reset receiver")

        res = False

        try:
           
            with serial.Serial(self.serial_port, self.baudrate) as serial_stream:

                if doc:
                    ublox_config.set_from_dict(serial_stream, doc)
                elif ucenter_file:                        
                    with open(ucenter_file, 'r') as fh:
                        ublox_config.set_from_ucenter_file(serial_stream, fh)

                self.receiver_config = ublox_config.get(serial_stream)

            logger.warning("Configuration submitted")

            res = True
     
        except serial.SerialException as e:
            logger.debug('Serial Exception error in submit configuration {}'.format(e))

        if receiver_was_recording:
            self.resume()
            logger.warning("Resuming data recording")

        return res
        
    # ---

    def get_configuration(self):
        return self.receiver_config

    # ---

    def set_file_rotation(self, slice_period):
        self.slice = slice_period

    def reset(self):
    
        receiver_was_recording = self.is_recording_ongoing()
        if receiver_was_recording:
            self.stop() 
            logger.warning("Stop recording session in order to reset receiver")

        with serial.Serial(self.serial_port, self.baudrate) as serial_stream:
            layer = 3
            payload = struct.pack('<BBBBBBBBBBBBB', 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, layer)
            core.send(core.PacketType.CFG_CFG, payload, serial_stream)
    
            core.wait_for_ack_nak(core.PacketType.CFG_CFG, serial_stream, timeout=core.TIMEOUT_DEF)
    
            logger.info(f'Device reset')

        if receiver_was_recording:
            self.resume()
            logger.warning("Resuming data recording")

        return True

    # ---

    def run(self):

        self.do_recording.set()

        while True:

            self.do_recording.wait()

            logger.info("Writing ubx data from _serial_stream [ {} ] for receiver [ {} ] to files "
                        "in folder [ {} ], with [ {} ] periodicity".format(
                               self.serial_port, self.receiver_name, self.output_dir, self.slice.name))

            with serial.Serial(self.serial_port, self.baudrate) as serial_stream, tempfile.TemporaryDirectory(prefix="pyublox_") as tempdir:

                logger.debug('Partial files will be written in this temporary folder [ {} ]'.format(tempdir))

                incoming_epoch_suffix = None
                num_packets = 0

                while self.do_recording.is_set():

                    try:
                        packet = core.get_next_packet(serial_stream, core.TIMEOUT_DEF)
                        packet_type, parsed_packet = core.parse(packet)
                    except Exception:
                        continue
                    
                    if packet_type == core.PacketType.RXM_RAWX:
                        tow = parsed_packet.rcvTow
                        week = parsed_packet.week
                        epoch = weektow_to_datetime(tow, week)
                        num_packets += 1
                        incoming_epoch_suffix = self.slice.build_rinex3_epoch(epoch)
                        if num_packets % 1000 == 0:
                            logger.debug("U-blox RXM_RAWX packet found! Epoch suffix {} , num_packets = {}".format(incoming_epoch_suffix, num_packets))

                    if not incoming_epoch_suffix:
                        continue

                    if not self.current_epoch_suffix or self.current_epoch_suffix != incoming_epoch_suffix:

                        self.__close_and_save_partial_file__()

                        self.current_epoch_suffix = incoming_epoch_suffix

                        filename = os.path.join(tempdir,'{}_{}.ubx'.format(self.receiver_name, self.current_epoch_suffix))
                        self.fout = open(filename, 'wb')
                        logger.info("Created new data file [ {} ]".format(os.path.basename(filename)))

                    self.fout.write(core.PREAMBLE)
                    self.fout.write(packet)
                    self.fout.flush()

                self.__close_and_save_partial_file__()

    # ---

    def stop(self):
        self.do_recording.clear()
        time.sleep(5)

    def resume(self):
        self.do_recording.set()

    def is_recording_ongoing(self):
        return self.do_recording.is_set()

    # ---

    def __close_and_save_partial_file__(self):

        if self.fout:
            filename = self.fout.name

            self.fout.close()

            src = filename
            dst = os.path.join(self.output_dir, os.path.basename(filename))

            logger.debug('Moving [ {} ] -> [ {} ]'.format(src, dst))

            shutil.move(src, dst)

        self.current_epoch_suffix = None
        self.fout = None



def autodetect_connection(serial_port=None, baudrate=None):
    """
    Attempt to find a connection for the receiver

    :return: If found, a dictionary with the connection settings
    """

    logger.debug('Input arguments for autodetect connection serial port [ {} ], baudrate [ {} ]'.format(serial_port, baudrate))

    serial_ports = [serial_port] if serial_port else __scan_serial_ports__()

    logger.debug('Checking serial ports [ {} ]'.format(serial_ports))

    if baudrate is None or serial_port is None:
        for serial_port in serial_ports:
            logger.debug('Checking serial port [ {} ]'.format(serial_port))
            baudrate = autodetect_baud_rate(serial_port)
            if baudrate: break
                
    if serial_port and baudrate:
        logger.info('u-blox found at serial [ {} ] and baudrate [ {} ]'.format(serial_port, baudrate))
    else:
        logger.warning('u-blox could not be found')

    return {
        SERIAL_PORT_STR: serial_port,
        BAUD_RATE_STR: baudrate
    }


def __scan_serial_ports__():
    """
    Attempt to find a serial port in which the receiver is connected. 

    :return: If found, a string with the device specification. Otherwise, None
    """

    serial_ports = serial.tools.list_ports.comports()

    serial_port = None
    for port in serial_ports:
        logger.debug('Checking port [ {} ]'.format(port))
        if 'u-blox GNSS receiver' in port:
            logger.debug('Candidate port found [ {} ]'.format(port))
            serial_port = port.device
            break

    if serial_port:
        serial_ports = [serial_port]
    else:
        serial_ports = scan_serial_ports()

    return serial_ports

# ------------------------------------------------------------------------------

def autodetect_baud_rate(serial_port):
   
    BAUD_RATE_LIST = [DEFAULT_BAUD_RATE, 9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]
    PACKET_TYPE_MON_VER = core.PacketType.MON_VER

    baudrate = None
    
    for baud_rate_candidate in BAUD_RATE_LIST:
        with serial.Serial(serial_port, baud_rate_candidate, timeout=1) as serial_stream:

            logger.debug(f'Autobauding: testing baud rate [ {baud_rate_candidate} ] and [ {serial_port} ]')

            core.send(PACKET_TYPE_MON_VER, b'', serial_stream)
            try:
                core.wait_for_packet(PACKET_TYPE_MON_VER, serial_stream)
                baudrate = baud_rate_candidate
                logger.debug(f'Autobauding: baud rate detected [ {baudrate} ] and [ {serial_port} ]')
                break
            except (RuntimeError, TimeoutError):
                logger.debug(f'Autobauding: baud rate [ {baud_rate_candidate} ]  and [ {serial_port} ] not working')


    if baudrate is None:
        logger.warning(f'Autobauding: could not find a working baudrate')

    return baudrate


def scan_serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

# ------------------------------------------------------------------------------

def detect_connection(serial_port=None, baudrate=None):

    logger.debug('Detect connection')

    serial_connection = autodetect_connection(serial_port, baudrate)
    serial_port = serial_connection.get(SERIAL_PORT_STR, None)
    baudrate = serial_connection.get(BAUD_RATE_STR, None)

    config = {}

    with serial.Serial(serial_port, baudrate=baudrate, timeout=core.TIMEOUT_DEF) as stream:
    
        logger.debug(f'Opened connection [ {serial_port} ] --> [ {stream} ]')

        config[SERIAL_PORT_STR] = serial_port
        if baudrate:
            config[BAUD_RATE_STR] = baudrate

        KEY_IDS = [core.KeyId.CFG_UART1_BAUDRATE, core.KeyId.CFG_UART1_STOPBITS, 
                    core.KeyId.CFG_UART1_PARITY, core.KeyId.CFG_UART1_DATABITS]

        key_values = core.submit_valget_packet(KEY_IDS, stream)

        key_id = core.KeyId.CFG_UART1_BAUDRATE
        if key_id in key_values:
            value = key_values[key_id]
            config[BAUD_RATE_STR] = core.parse_key_id_value(key_id, value)

        key_id = core.KeyId.CFG_UART1_STOPBITS
        if key_id in key_values:
            value = key_values[key_id]
            stopbits = core.parse_key_id_value(key_id, value)
            if stopbits == 1:
                config['stopbits'] = serial.STOPBITS_ONE
            elif stopbits == 2:
                config['stopbits'] = serial.STOPBITS_ONE_POINT_FIVE
            elif stopbits == 3:
                config['stopbits'] = serial.STOPBITS_TWO
            else:
                raise ValueError('Half bit not supported by pyserial')

        key_id = core.KeyId.CFG_UART1_PARITY
        if key_id in key_values:
            value = key_values[key_id]
            parity = core.parse_key_id_value(key_id, value)
            if parity == 0:
                config['parity'] = serial.PARITY_NONE
            elif parity == 1:
                config['parity'] = serial.PARITY_ODD
            elif parity == 2:
                config['parity'] = serial.PARITY_EVEN

        key_id = core.KeyId.CFG_UART1_DATABITS
        if key_id in key_values:
            value = key_values[key_id]
            bytesize = core.parse_key_id_value(key_id, value)
            if bytesize == 0:
                config['bytesize'] = serial.EIGHTBITS
            elif bytesize == 1:
                config['bytesize'] = serial.SEVENBITS
    
        logger.debug(f'Serial connection configuration {config}')

    return config

# ------------------------------------------------------------------------------

def detect_config(serial_port=None, baudrate=None, messages=False):

    config = {}

    try:
        config = detect_connection(serial_port=serial_port, baudrate=baudrate)
        serial_port = config.pop(SERIAL_PORT_STR)
        baudrate = config.pop(BAUD_RATE_STR)
    except Exception as e:
        logger.critical('Could not detect connection: {}. Re-run command'.format(str(e)))
        return None

    logger.info('Connection with a u-blox receiver detected at port [ {} ] and baudrate [ {} ]'.format(serial_port, baudrate))

    with serial.Serial(serial_port, baudrate, **config) as serial_stream:

        try:
            config = ublox_config.get(serial_stream)
            config[SERIAL_PORT_STR] = serial_port
            config[BAUD_RATE_STR] = baudrate
        except Exception as e:
            logger.critical(f'Unable to get receiver parameters. Re-launch the command::{str(e)}')
            return None

        if messages:
            config['messages'] = __detect_messages__(serial_stream, timeout=core.TIMEOUT_DEF)

    return config


def __detect_messages__(serial_stream, timeout=core.TIMEOUT_DEF):

    logger.debug("Detecting messages generated by the receiver")

    packet_types = set()

    try:
        t_start = time.time()

        for packet in core.ublox_packets(serial_stream, timeout):

            elapsed_time = time.time() - t_start

            ptype = packet[0:2]
            if ptype not in packet_types:
                packet_types.add(ptype)

            if (elapsed_time > 10):
                break
    except TimeoutError:
        pass

    return [core.PacketType(m).name for m in packet_types]
