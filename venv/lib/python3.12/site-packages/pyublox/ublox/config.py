import collections
import json
import os.path
import struct
import serial
import time
import threading
import tempfile
import shutil
import queue

from roktools import logger

from . import core
from . import constants
from . import helpers

# ------------------------------------------------------------------------------

def get(serial_stream):
    """
    Build an output dictionary of configuration from the settings defined in the
    package
    """

    config = {}

    key_ids = list(KEY_IDS[constants.LOGGING_STR].values()) + \
                __get_key_ids_from_constellations_key_ids_dict__()

    try:
        settings = core.submit_valget_packet(key_ids, serial_stream)
        import copy
        config = copy.deepcopy(KEY_IDS)

        config_logging = config[constants.LOGGING_STR]
        for config_item in config_logging:
            key_id = config_logging[config_item]
            value = core.parse_key_id_value(key_id, settings[key_id])
            if key_id == core.KeyId.CFG_RATE_MEAS:
                    value = value / 1000
            config_logging[config_item] = value

        config_constellations = config[constants.CONSTELLATIONS_STR]
        for config_constellation in config_constellations.values():
            key_id = config_constellation[constants.ENABLED_STR]
            value = core.parse_key_id_value(key_id, settings[key_id])
            config_constellation[constants.ENABLED_STR] = value


            config_signals = config_constellation[constants.SIGNALS_STR]
            for signal_str in config_signals:
                key_id = config_signals[signal_str]
                value = core.parse_key_id_value(key_id, settings[key_id])
                config_signals[signal_str] = value

    except:
        pass

    return config

# ------------------------------------------------------------------------------

def set_from_ucenter_file(serial_stream, fh):

    res = False

    if fh.readable():
        for line in fh:
            submit_res = core.submit_valset_from_valget_line(line, serial_stream)
            if submit_res:
                logger.info('Applied config line [ {} ]'.format(line))
            else:
                logger.warning('Could not apply config line [ {} ]'.format(line))

            res = res or submit_res
            time.sleep(0.2)

    return res

# ------------------------------------------------------------------------------

def set_from_json_file(serial_stream, fh):

    res = False

    if fh.readable():
        res = set_from_dict(serial_stream, json.loads(fh.read()))
        time.sleep(0.2)

    return res

# ------------------------------------------------------------------------------

def set_from_dict(serial_stream, config_dict):

    logger.debug(f'Incoming configuration parameters {config_dict}')
    config = {}

    logging_dict = config_dict.get(constants.LOGGING_STR, {})
    config.update(__logging_dict_to_keyids__(logging_dict))

    constellation_dict = config_dict.get(constants.CONSTELLATIONS_STR, {})
    config.update(__constellations_dict_to_keyids__(constellation_dict))

    if config:
        logger.debug(f'Configuration to apply {config}')
        core.submit_valset_packet(config, serial_stream)
        return True
    
    else:
        logger.warning('No configuration has been applied')
        return False

# ------------------------------------------------------------------------------

def __get_key_ids_from_constellations_key_ids_dict__():

    out = []

    for d in KEY_IDS[constants.CONSTELLATIONS_STR].values():
        key_id = d[constants.ENABLED_STR]
        out.append(key_id)

        signal_dict = d[constants.SIGNALS_STR]
        for key_id in signal_dict.values():
            out.append(key_id)

    return out

# ------------------------------------------------------------------------------

def __logging_dict_to_keyids__(config_dict):

    smoothing = config_dict.get(constants.SMOOTHING_STR, False)
    rate = config_dict.get(constants.RATE_STR, 1)
    measurements = config_dict.get(constants.MEASUREMENTS_STR, True)
    ephemeris = config_dict.get(constants.EPHEMERIS_STR, True)
    solutions = config_dict.get(constants.SOLUTION_STR, True)

    out = {
        core.KeyId.CFG_NAVSPG_USE_PPP : smoothing,
        core.KeyId.CFG_RATE_MEAS : int(rate * 1000), # s -> ms
        core.KeyId.CFG_MSGOUT_UBX_RXM_RAWX_USB : measurements,
        core.KeyId.CFG_MSGOUT_UBX_RXM_RAWX_UART1 : measurements,
        core.KeyId.CFG_MSGOUT_UBX_RXM_RAWX_UART2 : measurements,
        #core.KeyId.CFG_RATE_NAV : rate,
        core.KeyId.CFG_MSGOUT_UBX_NAV_POSECEF_USB : solutions, 
        core.KeyId.CFG_MSGOUT_UBX_NAV_POSECEF_UART1 : solutions, 
        core.KeyId.CFG_MSGOUT_UBX_NAV_POSECEF_UART2 : solutions, 
        core.KeyId.CFG_MSGOUT_UBX_RXM_SFRBX_USB : ephemeris,
        core.KeyId.CFG_MSGOUT_UBX_RXM_SFRBX_UART1 : ephemeris,
        core.KeyId.CFG_MSGOUT_UBX_RXM_SFRBX_UART2 : ephemeris
    }

    return out

__logging_dict_to_keyids__.__doc__ = f""" 
    {{
        {constants.SMOOTHING_STR}: false,
        {constants.RATE_STR}: 30,
        {constants.MEASUREMENTS_STR}: true,
        {constants.EPHEMERIS_STR}: true,
        {constants.SOLUTION_STR}: true,
        {constants.FILE_ROTATION_STR}: 86400
    }}
"""

# ------------------------------------------------------------------------------

def __constellations_dict_to_keyids__(constellations_config):
    """
    Method to convert from JSON configuration dictionary to a flat dictionary
    of receiver key_ids and values
    
    :params config_dict: is the full configuration dictioary that includes,
    in particular, the "constellations" key.
    {
            "BDS": {
                "enabled": true,
                "signals": {"2I":true}
            },
            "GAL": {
                "enabled": true,
                "signals": {"7B":true,"1C":true}
            },
            "GLO": {
                "enabled": true,
                "signals": {"2C":true,"1C":true}
            },
            "GPS": {
                "enabled": true,
                "signals": {"2L":true,"1C":true}
            },
            "QZSS": {
                "enabled": true,
                "signals": {"2W":true,"1C":true}
            }
    }
    """

    out = {}

    constellations_keyids = KEY_IDS[constants.CONSTELLATIONS_STR]

    for constellation_str in constellations_keyids:

        constellation_keyids = constellations_keyids[constellation_str]
        constellation_config = constellations_config.get(constellation_str, {})

        key_id = constellation_keyids[constants.ENABLED_STR]
        enabled = constellation_config.get(constants.ENABLED_STR, False)
        
        out[key_id] = enabled

        signals_keyids = constellation_keyids[constants.SIGNALS_STR]
        signals_config = constellation_config.get(constants.SIGNALS_STR, {})

        for signal_str in signals_keyids:
            key_id = signals_keyids[signal_str]
            enabled = signals_config.get(signal_str, False)
            out[key_id] = enabled

    return out

# ------------------------------------------------------------------------------

KEY_IDS = {
    constants.LOGGING_STR: {
        constants.SMOOTHING_STR: core.KeyId.CFG_NAVSPG_USE_PPP,
        constants.RATE_STR: core.KeyId.CFG_RATE_MEAS,
        constants.MEASUREMENTS_STR: core.KeyId.CFG_MSGOUT_UBX_RXM_RAWX_USB,
        constants.EPHEMERIS_STR: core.KeyId.CFG_MSGOUT_UBX_RXM_SFRBX_USB,
        constants.SOLUTION_STR: core.KeyId.CFG_MSGOUT_UBX_NAV_POSECEF_USB
    },
    constants.CONSTELLATIONS_STR:
    {
        constants.BDS_STR: {
            constants.ENABLED_STR: core.KeyId.CFG_SIGNAL_BDS_ENA,
            constants.SIGNALS_STR: {
                "2I": core.KeyId.CFG_SIGNAL_BDS_B1_ENA, 
                "7I": core.KeyId.CFG_SIGNAL_BDS_B2_ENA
            }   
        },
        constants.GAL_STR: {
            constants.ENABLED_STR: core.KeyId.CFG_SIGNAL_GAL_ENA,
            constants.SIGNALS_STR: {
                "1C": core.KeyId.CFG_SIGNAL_GAL_E1_ENA, 
                "7B": core.KeyId.CFG_SIGNAL_GAL_E5B_ENA
            }
        },
        constants.GLO_STR: {
            constants.ENABLED_STR: core.KeyId.CFG_SIGNAL_GLO_ENA,
            constants.SIGNALS_STR: {
                "1C": core.KeyId.CFG_SIGNAL_GLO_L1_ENA, 
                "2C": core.KeyId.CFG_SIGNAL_GLO_L2_ENA
            }
        },
        constants.GPS_STR: {
            constants.ENABLED_STR: core.KeyId.CFG_SIGNAL_GPS_ENA,
            constants.SIGNALS_STR: {
                "1C": core.KeyId.CFG_SIGNAL_GPS_L1CA_ENA, 
                "2L": core.KeyId.CFG_SIGNAL_GPS_L2C_ENA
            }
        },
        constants.QZSS_STR: {
            constants.ENABLED_STR: core.KeyId.CFG_SIGNAL_QZSS_ENA,
            constants.SIGNALS_STR: {
                "1C": core.KeyId.CFG_SIGNAL_QZSS_L1CA_ENA, 
                "2L": core.KeyId.CFG_SIGNAL_QZSS_L2C_ENA
            }
        }
    }
}

# ------------------------------------------------------------------------------
