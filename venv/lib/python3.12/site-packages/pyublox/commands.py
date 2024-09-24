import datetime
import json
import os.path
import sys
import serial
import signal
import time
import queue

from roktools import logger, rinex

from .ublox import config as ublox_config
from .ublox import core as ublox_core
from .ublox import constants
from .ublox import helpers
from .ublox import receiver
from .ublox import SERIAL_PORT_STR, BAUD_RATE_STR
from .ublox.constants import FILE_ROTATION_STR
from . import OUTPUT_DIR_STR, NAME_STR, FILE_STR, JSON_STR, RATE_STR, MESSAGES_STR

# ------------------------------------------------------------------------------

def config(**kwargs):

    serial_port = kwargs.get(SERIAL_PORT_STR, None)
    baudrate = kwargs.get(BAUD_RATE_STR, None)
    config_file = kwargs.get(FILE_STR, None)
    json_file = kwargs.get(JSON_STR, None)

    ublox_receiver = receiver.Receiver(serial_port=serial_port, baudrate=baudrate)

    if config_file:
        logger.info(f'Configuring device from file [ {config_file} ]')
        ublox_receiver.submit_configuration_from_ucenter_file(config_file)

    elif json_file:
        logger.info(f'Configuring device from JSON file [ {json_file} ]')
        with open(json_file, 'r') as fh:
            ublox_receiver.submit_configuration(json.loads(fh.read()))

    else:
        config_dict = {}

        if RATE_STR in kwargs:
            config_dict[constants.RATE_STR] = {
                constants.MEASUREMENTS_STR : kwargs[RATE_STR],
                constants.SOLUTION_STR : kwargs[RATE_STR] * 5
            }

        logger.info(f'Configuring device from command line options: {config_dict}')
        ublox_receiver.submit_configuration(config_dict)
 
# ------------------------------------------------------------------------------

def detect(serial_port=None, baudrate=None, messages=False):

    ublox_receiver = receiver.Receiver(serial_port=serial_port, baudrate=baudrate, detect_messages=messages)

    config = ublox_receiver.get_configuration()
    if not config:
        return False

    json.dump(config, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write('\n')
    return True



# ------------------------------------------------------------------------------

def reset(**kwargs):

    serial_port = kwargs.get(SERIAL_PORT_STR, None)
    baudrate = kwargs.get(BAUD_RATE_STR, None)
    ublox_receiver = receiver.Receiver(serial_port=serial_port, baudrate=baudrate)
    ublox_receiver.reset()

    return True

# ------------------------------------------------------------------------------

_receiver = None

# ------------------------------------------------------------------------------

def record(**kwargs):

    logger.debug('Record arguments ' + str(kwargs))

    slice_period = kwargs.get('slice', rinex.FilePeriod.DAILY)
    file_rotation = rinex.FilePeriod.from_string(slice_period)
    recorder_config = {
        SERIAL_PORT_STR: kwargs.get(SERIAL_PORT_STR, None),
        BAUD_RATE_STR: kwargs.get(BAUD_RATE_STR, None),
        FILE_ROTATION_STR: file_rotation,
        OUTPUT_DIR_STR: kwargs.get(OUTPUT_DIR_STR, '.'),
        'receiver_name': kwargs.get(NAME_STR, 'UBLX')
    }

    global _receiver
    _receiver = receiver.Receiver(**recorder_config)

    _receiver.start()


# ------------------------------------------------------------------------------

def interruption_handler(sig, frame):
    logger.info('You pressed Ctrl+C, gracefully closing files and serial streams')

    global _receiver
    _receiver.stop()

    sys.exit(0)

signal.signal(signal.SIGINT, interruption_handler)
