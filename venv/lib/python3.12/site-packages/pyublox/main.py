#!/usr/bin/env python
"""
Python client to interface with Ublox receivers

Usage:
    pyublox -h | --help
    pyublox --version
    pyublox detect [ -p <serial-port>] [-b <baudrate>] [-m] [-d <level>]
    pyublox config [ -p <serial-port>] [-b <baudrate>] [-f <file>] [-j <file>]
                   [ -r <rate> ] [ --obs-data ] [ --nav-data ] [ --llh-data ] [-d <level>]
    pyublox record [ -p <serial-port>] [-b <baudrate>] [ -n <name> ] [-o <dir>] 
                   [-s <periodicity>] [-d <level>]
    pyublox reset  [-d <level>] [-b <baudrate>] <serial-port>

Options:
    -h --help         Show this help
    -v --version      Show the version
    -d --debug (DEBUG | INFO | WARNING | CRITICAL)
                      Issue debug information (verbosity controlled by the given debug level) [default: CRITICAL]
    -p --serial-port <serial-port>
                      Serial port to work with (e.g. /dev/ttyACM0)
    -b --baudrate <baudrate> Baud rate of the port (e.g. 115200)
    -f --file <file>  Configure the receiver using a configuration file obtained 
                      with ublox u-center tool.
    -j --json <file>  Configure the receiver using a JSON file with the configuration
                      parameters. A sample of the format for the JSON file can 
                      be obtained with the `detect` command
    -m --messages     When detecting the connection, take a sample and check 
                      which messages are reported
    -r --rate <rate>  Data rate at which the observable will be recorded
    --obs-data        Switch to turn on recording of observables (GNSS measurements)
    --nav-data        Switch to turn on recording of navigation data
    --llh-data        Switch to turn on recording of position estimates
    -o --output-dir <dir>
                      Record the ublox stream to a file or to stdout. If a file 
                      is specified, the periodicity  [default: .]
    -n --name <name>  Specify the receiver name, that will be used to name the 
                      receiver [default: UBLX]
    -s --slice (daily | hourly | quarterly)
                      Slice the data in chunks of given periodicity [default: daily]

Commands:
    detect  Detect ublox device attached to serial port and output basic configuration
    config  Submit configuration options to ublox
    record  Record valid binary UBX packets received from the given port to a file
    reset   Reset the receiver (use when the receiver seems to be hanged or not responsive)
"""
import docopt
import pkg_resources
import sys

from . import commands

from .ublox import SERIAL_PORT_STR, BAUD_RATE_STR

SERIAL_PORT_OPTION_STR = 'serial-port'
BAUD_RATE_OPTION_STR = 'baudrate'
OUTPUT_DIR_OPTION_STR = 'output-dir'
RATE_OPTION_STR = 'rate'
FILE_OPTION_STR = 'file'
JSON_OPTION_STR = 'json'
MESSAGES_OPTION_STR = 'messages'
SLICE_OPTION_STR = 'slice'
NAME_OPTION_STR = 'name'

from roktools import logger

def main():
    """
    """

    version = pkg_resources.require("pyublox")[0].version

    args = docopt.docopt(__doc__, version=version, options_first=False)

    logger.set_level(args['--debug'])
    logger.debug(f"Start main, parsed arg\n {args}")

    command, command_args = __get_command__(args)
    logger.debug(f'Command arguments\n {command_args}')

    try:
        command(**command_args)
    except Exception as e:
        sys.stderr.write("FATAL: " + str(e))

    return 0

def __get_command__(args):

    command = None
    command_args = {}

    if args['config']:
        command = commands.config
        options = [SERIAL_PORT_OPTION_STR, BAUD_RATE_OPTION_STR, RATE_OPTION_STR, FILE_OPTION_STR, JSON_OPTION_STR]
        command_args = __build_command_args__(options, args)

    elif args['detect']:
        command = commands.detect
        options = [SERIAL_PORT_OPTION_STR, BAUD_RATE_OPTION_STR, MESSAGES_OPTION_STR]
        command_args = __build_command_args__(options, args)

    elif args['reset']:
        command = commands.reset
        command_args = {
            SERIAL_PORT_STR : args.get(f'<{SERIAL_PORT_OPTION_STR}>', None),
            BAUD_RATE_STR : args[f'<{BAUD_RATE_OPTION_STR}>']
        }

    elif args['record']:
        command = commands.record
        options = [SERIAL_PORT_OPTION_STR, OUTPUT_DIR_OPTION_STR, SLICE_OPTION_STR, NAME_OPTION_STR]
        command_args = __build_command_args__(options, args)

    return command, command_args


def __build_command_args__(parameters, arguments):

    command_args = {}
    for parameter in parameters:
        key = f'--{parameter}'
        if arguments[key]:
            parameter = parameter.replace('-','_')
            command_args[parameter] = arguments[key]

    return command_args


if __name__ == "__main__":

    return_code = main()
    sys.exit(return_code)
