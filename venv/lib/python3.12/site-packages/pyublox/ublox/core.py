import collections
import enum
import io
import sys
import serial
import serial.tools.list_ports
import time
import struct

from roktools import logger

from . import helpers

TIMEOUT_DEF = 5

def compute_checksum(message_without_synch_nor_checksum):
    """
    Compute the checksum from a message (without synch nor checksum)

    >>> MESSAGE = PREAMBLE + PacketType.RXM_RAWX.value + bytes.fromhex('10 00 00 00 00 00 00 c8 ba 40 00 00 12 00 00 01 5d a1 fa bd')
    >>> checksum = compute_checksum(MESSAGE[2:-2])
    >>> bytes(checksum).hex()
    'fabd'
    """

    CK_A = 0
    CK_B = 0

    for b in message_without_synch_nor_checksum:
        CK_A = (CK_A + b) & 0xff
        CK_B = CK_B + CK_A & 0xff
    
    return CK_A, CK_B

def validate_checksum(packet_without_sync):
    """

    >>> MESSAGE = PREAMBLE + PacketType.RXM_RAWX.value + bytes.fromhex('10 00 00 00 00 00 00 c8 ba 40 00 00 12 00 00 01 5d a1 fa bd')
    >>> validate_checksum(MESSAGE[2:])
    True
    """

    ck_a, ck_b = compute_checksum(packet_without_sync[:-2])
    ck_a_ref = packet_without_sync[-2]
    ck_b_ref = packet_without_sync[-1]

    is_valid = (ck_a == ck_a_ref and ck_b == ck_b_ref)
    if not is_valid:
        msg_str = helpers.print_bytes(packet_without_sync)
        logger.debug('Invalid checksum for packet [ {} ] (nbytes {}), expected [ {} ], got [ {} ]'.format(msg_str, len(packet_without_sync), bytes([ck_a_ref, ck_b_ref]), bytes([ck_a, ck_b])))

    return is_valid


def ublox_packets(stream, timeout):

    while(stream.is_open):
        packet = get_next_packet(stream, timeout)
        if packet:
            yield packet

def get_next_packet(stream, timeout):

    next_packet = None

    if stream.is_open:
        __go_to_next_packet__(stream, timeout)
        next_packet = get_next_package(stream)

    return next_packet
    

def get_next_package(stream):

    packet_type = stream.read(2)
    length = stream.read(2)

    payload = stream.read(int.from_bytes(length, "little"))

    ck_a = stream.read(1)
    ck_b = stream.read(1)

    packet = packet_type + length + payload + ck_a + ck_b

    valid_checksum = validate_checksum(packet)

    return packet if valid_checksum else None


def parse(packet_without_sync):

    packet_type = PacketType(packet_without_sync[0:2])

    parser_name = 'parse_' + packet_type.name.lower()
    parser = globals()[parser_name]

    fields = parser(packet_without_sync)
    
    return packet_type, fields


def extract_type_bytes(packet_without_sync):
    return packet_without_sync[0:2].hex()



def get_n_bytes_of_config_value(key_id):
    """
    Get the storage size of the configuration value from its KeyID, as per 
    description of section 7.2 of ublox interface description

    >>> get_n_bytes_of_config_value(KeyId.CFG_MSGOUT_UBX_RXM_RAWX_USB)
    1
    >>> get_n_bytes_of_config_value(KeyId.CFG_UART2_BAUDRATE)
    4
    >>> get_n_bytes_of_config_value(KeyId.CFG_RATE_MEAS)
    2
    """

    size = int((key_id.value >> 28) & 0x07)

    if size <= 2: return 1
    if size == 3: return 2
    if size == 4: return 4
    if size == 5: return 8


# -----

def send(packet_type, payload, serial_stream):
    """
    Send a payload of a given type to the ublox receiver
    """

    def build_packet(packet_type, payload):
    
        packet = struct.pack('<BBBBH', *PREAMBLE, *(packet_type.value), len(payload))
        packet += payload
        (ck_a, ck_b) = compute_checksum(packet[2:])
        packet += struct.pack('<BB', ck_a, ck_b)
     
        return packet

    packet = build_packet(packet_type, payload)

    bytes_str = helpers.print_bytes(packet)
    n_bytes_written = serial_stream.write(packet)
    logger.debug(f'Packet sent    [ {bytes_str} ], [ {n_bytes_written} ] bytes written')
    return n_bytes_written

# -----

def send_hex_string(string, serial_stream):
    """
    Send a serialized (stringified) hex string that contains both the packet
    type and payload. The method builds the rest of the packet and sends it to
    the receiver. The string expects the packet type, length and payload.

    An example to send CFG-MON-VER (poll request): "0A 04 00 00"
    """

    msg = bytes.fromhex(string)

    (ck_a, ck_b) = compute_checksum(msg)
    packet = struct.pack('<BB', *PREAMBLE) + msg + struct.pack('<BB', ck_a, ck_b)
    n_bytes_written = serial_stream.write(packet)

    logger.debug(f'Packet sent    [ {helpers.print_bytes(packet)} ], [ {n_bytes_written} ] bytes written')
    return n_bytes_written

# -----

def get_packet_length(packet_without_sync):
    return struct.unpack_from('<H', packet_without_sync, 2)[0]

class PacketTransaction:

    def __init__(self):
        self.packet_received = False
        self.ack_received = False

    def completed(self):
        return self.packet_received and self.ack_received


def submit_valget_packet(key_ids, serial_stream, timeout=TIMEOUT_DEF):

    PACKET_TYPE = PacketType.CFG_VALGET

    payload = struct.pack('<BBBB', 0, 0, 0, 0)

    for key_id in key_ids:
        payload += struct.pack('L', key_id.value)

    send(PACKET_TYPE, payload, serial_stream)

    packet_transaction = PacketTransaction()

    key_values = {}
    t_start = time.time()
    for packet in ublox_packets(serial_stream, timeout):

        elapsed_time = time.time() - t_start

        try:
            incoming_packet_type, parsed_packet = parse(packet)
        except KeyError:
            byte_str = helpers.print_bytes(packet)
            logger.debug('unsupported packet [ {} ... ]'.format(byte_str))
            continue

        logger.debug('packet detected    [ {} ]'.format(incoming_packet_type))

        if incoming_packet_type == PACKET_TYPE:
            cfg_data = parsed_packet.cfgData
            key_values = __extract_cfg_data__(cfg_data)
            logger.debug('detected [ {} ] configuration items'.format(len(key_values)))
            packet_transaction.packet_received = True
            if packet_transaction.completed():
                break

        elif incoming_packet_type == PacketType.ACK_ACK:
            if __ack_nck_corresponds_to_packet__(PACKET_TYPE, parsed_packet):
                logger.debug(f'ACK for [ {PACKET_TYPE} ] received!')
                packet_transaction.ack_received = True
                if packet_transaction.completed():
                    break

        elif incoming_packet_type == PacketType.ACK_NAK:
            if __ack_nck_corresponds_to_packet__(PACKET_TYPE, parsed_packet):
                logger.critical(f'NAK for [ {PACKET_TYPE} ] received!')
                break

        elif elapsed_time >= timeout:
            logger.critical('Time out reached without receiving the polled VALGET message')
            break

    return key_values

# ------------------------------------------------------------------------------

def submit_valset_packet(key_value_dict, serial_stream, timeout=TIMEOUT_DEF):
    """
    Set configuration items based on an input dictionary of key_ids and its
    corresponding value
    """

    PACKET_TYPE = PacketType.CFG_VALSET

    sorted_dict = dict(sorted(key_value_dict.items(), key=lambda x : x[0].value))

    for key_id,value in sorted_dict.items():
        
        # Transaction-less submission of packets
        payload = struct.pack('<BBBB', 0, 7, 0, 0)

        logger.debug('submit_valset_packet::key_id [ {} ] --> [ {} ]'.format(key_id, value))
        try:

            payload += struct.pack('<L', key_id.value)
            fmt = KEY_ID_VALUE_FORMATS.get(key_id, 'B')

            payload += struct.pack(fmt, value)
        except Exception as e:
            logger.warning(f'Error processing [ {key_id} ]: {str(e)}')
            continue

        send(PACKET_TYPE, payload, serial_stream)
        wait_for_ack_nak(PACKET_TYPE, serial_stream, timeout=TIMEOUT_DEF)

    logger.info('Ending transaction for VALSET!')



def wait_for_ack_nak(packet_type, serial_stream, timeout=TIMEOUT_DEF):

    t_start = time.time()
    for packet in ublox_packets(serial_stream, timeout):

        try:
            incoming_packet_type, parsed_packet = parse(packet)
        except KeyError:
            logger.debug('packet [ {} ] unsupported'.format(PacketType(packet[0:2])))
            continue

        logger.debug(f'Received packet {incoming_packet_type}')

        if incoming_packet_type == PacketType.ACK_ACK:
            if __ack_nck_corresponds_to_packet__(packet_type, parsed_packet):
                return True

        elif incoming_packet_type == PacketType.ACK_NAK:
            if __ack_nck_corresponds_to_packet__(packet_type, parsed_packet):
                raise RuntimeError(f'NAK for [ {packet_type} ] received!')

        elif time.time() - t_start >= timeout:
            raise TimeoutError(f'Time out reached without receiving a confirmation for {packet_type}')

    raise RuntimeError('End of input stream and no ACK/NCK received')

def wait_for_packet(packet_type, serial_stream, timeout=TIMEOUT_DEF):

    t_start = time.time()
    for packet in ublox_packets(serial_stream, timeout):

        try:
            incoming_packet_type, parsed_packet = parse(packet)
        except KeyError:
            logger.debug('packet [ {} ] unsupported'.format(PacketType(packet[0:2])))
            continue

        if incoming_packet_type == packet_type:
            return packet

        elif incoming_packet_type == PacketType.ACK_ACK:
            logger.info(f'ACK  for [ {packet_type} ] received!')

        elif incoming_packet_type == PacketType.ACK_NAK:
            if __ack_nck_corresponds_to_packet__(packet_type, parsed_packet):
                raise RuntimeError(f'NAK for [ {packet_type} ] received!')

        elif time.time() - t_start >= timeout:
            raise TimeoutError(f'Time out reached without receiving a confirmation for {packet_type}')

    raise RuntimeError(f'End of input stream and no packet {packet_type} received')




def get_string_from_bytearray(b):
    """
    >>> b =  bytes.fromhex("4d 4f 44 3d 5a 45 44 2d 46 39 50 00 00")
    >>> get_string_from_bytearray(b)
    'MOD=ZED-F9P'
    """
    s = str(b.decode('utf-8'))
    s = s[0:s.find('\x00')]
    return s

def get_receiver_info(serial_stream, timeout=5):

    send(PacketType.MON_VER, struct.pack('<'), serial_stream)

    for packet in ublox_packets(serial_stream, timeout):

        try:
            packet_type, parsed_packet = parse(packet)

            if packet_type == PacketType.MON_VER:

                res = {
                    'sw_version' : get_string_from_bytearray(parsed_packet.swVersion),
                    'hw_version' : get_string_from_bytearray(parsed_packet.hwVersion),
                }

                extension_bytearray = parsed_packet.extension
                for l in range(int(len(extension_bytearray) / 30)):
    
                    fields = get_string_from_bytearray(extension_bytearray[l*30:]).split('=')
                    if len(fields) == 2:
                        res[fields[0]] = fields[1]
                        
                return res

        except KeyError:
            pass


def submit_valset_from_valget_line(line, serial_stream):
    """
    Copy a CFG-VALGET line contents and use them to submit a CFG-VALSET 
    
    >>> line = 'MON-VER - 0A 04 DC 00 45 ...'
    >>> submit_valset_from_valget_line(line, None)
    False
    """
    SKIP_NBYTES = 8
    
    if not line.startswith('CFG-VALGET'):
        return False
    
    logger.debug(f'Sending contents of line [ {line} ]')
    packet = line.split(' - ')[1]
    payload_str = packet[SKIP_NBYTES*3:]
    
    payload = struct.pack('<BBBB', 0, 7, 0, 0)
    payload += bytes.fromhex(payload_str)
    
    n_bytes_sent = send(PacketType.CFG_VALSET, payload, serial_stream)
    wait_for_ack_nak(PacketType.CFG_VALSET, serial_stream)
    
    return n_bytes_sent
    


PAYLOAD_OFFSET=4

# Packet classes and parsers
Ack = collections.namedtuple('Ack', ['clsID', 'msgID'])
def parse_ack_ack(packet_without_sync):

    HEADER_FMT = '<BB'

    fields = struct.unpack_from(HEADER_FMT, packet_without_sync, PAYLOAD_OFFSET)

    return Ack(*fields)

def parse_ack_nak(packet_without_sync):
    return parse_ack_ack(packet_without_sync)

MonVer = collections.namedtuple('MonVer', ['swVersion', 'hwVersion', 'extension'])
def parse_mon_ver(packet_without_sync):

    length = get_packet_length(packet_without_sync)
    N = int((length - 40) / 30)
    HEADER_FMT = '<30s10s{}s'.format(N * 30)

    fields = struct.unpack_from(HEADER_FMT, packet_without_sync, PAYLOAD_OFFSET)

    return MonVer(*fields)

MonHw = collections.namedtuple('MonHw', ['pinSel', 'pinBank', 'pinDir', 'pinVal', 
                                         'noisePerMS', 'agcCnt', 'aStatus', 'aPower', 
                                         'flags', 'reserved1', 'usedMask', 'VP', 
                                         'jamInd', 'reserved2', 'pinIrq', 'pullH', 'pullL'])
def parse_mon_hw(packet_without_sync):

    HEADER_FMT = '<4p4p4p4pHHBBpB4p17pB2p4p4p4p'

    fields = struct.unpack_from(HEADER_FMT, packet_without_sync, PAYLOAD_OFFSET)

    return MonHw(*fields)

RxmRawx = collections.namedtuple('RxmRawx', ['rcvTow', 'week', 'leapS', 'numMeas', 'recStat', 'version', 'reserved1'])
def parse_rxm_rawx(packet_without_sync):

    HEADER_FMT = '<dHbBBB2s'

    fields = struct.unpack_from(HEADER_FMT, packet_without_sync, PAYLOAD_OFFSET)

    return RxmRawx(*fields)


RxmSfrbx = collections.namedtuple('RxmSfrbx', ['gnssId', 'svId', 'reserved1', 'freqId', 'numWords', 'chn', 'version', 'reserved2'])
def parse_rxm_sfrbx(packet_without_sync):

    HEADER_FMT = '<BBBBBBBB'

    fields = struct.unpack_from(HEADER_FMT, packet_without_sync, PAYLOAD_OFFSET)

    return RxmSfrbx(*fields)


NavStatus = collections.namedtuple('NavStatus', ['iTOW', 'gpsFix', 'flags', 'fixStat', 'flags2', 'ttff', 'msss'])
def parse_nav_status(packet_without_sync):

    HEADER_FMT = '<LBpppLL'

    fields = struct.unpack_from(HEADER_FMT, packet_without_sync, PAYLOAD_OFFSET)

    return NavStatus(*fields)


NavPosecef = collections.namedtuple('NavPosecef', ['iTOW', 'ecefX', 'ecefY', 'ecefZ', 'pAcc'])
def parse_nav_posecef(packet_without_sync):

    HEADER_FMT = '<LlllL'

    fields = struct.unpack_from(HEADER_FMT, packet_without_sync, PAYLOAD_OFFSET)

    return NavPosecef(*fields)


NavPosllh = collections.namedtuple('NavPosllh', ['iTOW', 'lon', 'lat', 'height', 'hMSL', 'hAcc', 'vAcc'])
def parse_nav_posllh(packet_without_sync):

    HEADER_FMT = '<LllllLL'

    fields = list(fields = struct.unpack_from(HEADER_FMT, packet_without_sync, PAYLOAD_OFFSET))

    # Scale parameters
    fields[1] = fields[1] * 1.0e-7
    fields[2] = fields[2] * 1.0e-7

    return NavPosllh(*fields)

CfgValget = collections.namedtuple('CfgValget', ['version', 'layer', 'reserved1', 'cfgData'])
def parse_cfg_valget(packet_without_sync):

    N = get_packet_length(packet_without_sync) - 4
    HEADER_FMT = '<BB2s{}s'.format(N)

    fields = struct.unpack_from(HEADER_FMT, packet_without_sync, PAYLOAD_OFFSET)

    return CfgValget(*fields)


PREAMBLE = b'\xb5\x62'

CLASS_ID_NAV = b'\x01'
CLASS_ID_RXM = b'\x02'
CLASS_ID_ACK = b'\x05'
CLASS_ID_CFG = b'\x06'
CLASS_ID_MON = b'\x0A'

def make_packet_type(class_id, msg_id):
    return PacketType(bytes([class_id, msg_id]))


class PacketType(enum.Enum):

    NAV_POSECEF = CLASS_ID_NAV + b'\x01'
    NAV_POSLLH = CLASS_ID_NAV + b'\x02'
    NAV_STATUS = CLASS_ID_NAV + b'\x03'
    NAV_SVIN = CLASS_ID_NAV + b'\x3b'

    RXM_SFRBX = CLASS_ID_RXM + b'\x13'
    RXM_RAWX = CLASS_ID_RXM + b'\x15'
    
    ACK_ACK = CLASS_ID_ACK + b'\x01'
    ACK_NAK = CLASS_ID_ACK + b'\x00'
    
    CFG_CFG = CLASS_ID_CFG + b'\x09'
    CFG_VALSET = CLASS_ID_CFG + b'\x8a'
    CFG_VALGET = CLASS_ID_CFG + b'\x8b'

    MON_VER = CLASS_ID_MON + b'\x04'
    MON_HW = CLASS_ID_MON + b'\x09'

class KeyId(enum.Enum):

    CFG_CLOCK_OSC_FREQ = 0x40a4000d
    CFG_I2COUTPROT_UBX = 0x10720001

    CFG_INFMSG_UBX_USB = 0x20920004

    CFG_MSGOUT_UBX_RXM_RAWX_I2C = 0x209102a4
    CFG_MSGOUT_UBX_RXM_RAWX_UART1 = 0x209102a5
    CFG_MSGOUT_UBX_RXM_RAWX_UART2 = 0x209102a6
    CFG_MSGOUT_UBX_RXM_RAWX_USB = 0x209102a7
    CFG_MSGOUT_UBX_RXM_RAWX_SPI = 0x209102a8

    CFG_MSGOUT_UBX_NAV_POSECEF_UART1 = 0x20910025
    CFG_MSGOUT_UBX_NAV_POSECEF_UART2 = 0x20910026
    CFG_MSGOUT_UBX_NAV_POSECEF_USB = 0x20910027
    CFG_MSGOUT_UBX_RXM_SFRBX_UART1 = 0x20910232
    CFG_MSGOUT_UBX_RXM_SFRBX_UART2 = 0x20910233
    CFG_MSGOUT_UBX_RXM_SFRBX_USB = 0x20910234

    CFG_MSGOUT_NMEA_ID_GSA_I2C = 0x209100bf
    CFG_MSGOUT_NMEA_ID_GSA_UART1 = 0x209100c0
    CFG_MSGOUT_NMEA_ID_GSA_UART2 = 0x209100c1
    CFG_MSGOUT_NMEA_ID_GSA_USB = 0x209100c2
    CFG_MSGOUT_NMEA_ID_GSA_SPI = 0x209100c3

    CFG_RATE_MEAS = 0x30210001
    CFG_RATE_NAV = 0x30210002
    CFG_RATE_TIMEREF = 0x20210003

    CFG_UART1_BAUDRATE = 0x40520001
    CFG_UART1_STOPBITS = 0x20520002
    CFG_UART1_DATABITS = 0x20520003
    CFG_UART1_PARITY = 0x20520004

    CFG_UART2_BAUDRATE = 0x40530001
    CFG_UART2_STOPBITS = 0x20530002
    CFG_UART2_DATABITS = 0x20530003
    CFG_UART2_PARITY   = 0x20530004

    CFG_SIGNAL_GPS_L1CA_ENA  = 0x10310001
    CFG_SIGNAL_GPS_L2C_ENA   = 0x10310003
    CFG_SIGNAL_GAL_E1_ENA    = 0x10310007
    CFG_SIGNAL_GAL_E5B_ENA   = 0x1031000a
    CFG_SIGNAL_BDS_B1_ENA    = 0x1031000d
    CFG_SIGNAL_BDS_B2_ENA    = 0x1031000e
    CFG_SIGNAL_QZSS_L1CA_ENA = 0x10310012
    CFG_SIGNAL_QZSS_L2C_ENA  = 0x10310015
    CFG_SIGNAL_GLO_L1_ENA    = 0x10310018
    CFG_SIGNAL_GLO_L2_ENA    = 0x1031001a
    CFG_SIGNAL_GPS_ENA       = 0x1031001f
    CFG_SIGNAL_GAL_ENA       = 0x10310021
    CFG_SIGNAL_BDS_ENA       = 0x10310022
    CFG_SIGNAL_QZSS_ENA      = 0x10310024
    CFG_SIGNAL_GLO_ENA       = 0x10310025

    CFG_NAVSPG_USE_PPP = 0x10110019

KEY_ID_VALUE_FORMATS = {
    KeyId.CFG_RATE_MEAS : 'H',
    KeyId.CFG_RATE_NAV: 'H',
    KeyId.CFG_CLOCK_OSC_FREQ : 'L',
    KeyId.CFG_UART1_BAUDRATE : 'L',
    KeyId.CFG_UART2_BAUDRATE : 'L',
    KeyId.CFG_NAVSPG_USE_PPP : '?',
    KeyId.CFG_MSGOUT_UBX_RXM_RAWX_USB : '?',
    KeyId.CFG_MSGOUT_UBX_NAV_POSECEF_USB : '?',
    KeyId.CFG_MSGOUT_UBX_RXM_SFRBX_USB : '?',
    KeyId.CFG_SIGNAL_GPS_L1CA_ENA : '?',
    KeyId.CFG_SIGNAL_GPS_L2C_ENA : '?',
    KeyId.CFG_SIGNAL_GAL_E1_ENA : '?',
    KeyId.CFG_SIGNAL_GAL_E5B_ENA : '?',
    KeyId.CFG_SIGNAL_BDS_B1_ENA : '?',
    KeyId.CFG_SIGNAL_BDS_B2_ENA : '?',
    KeyId.CFG_SIGNAL_QZSS_L1CA_ENA : '?',
    KeyId.CFG_SIGNAL_QZSS_L2C_ENA : '?',
    KeyId.CFG_SIGNAL_GLO_L1_ENA : '?',
    KeyId.CFG_SIGNAL_GLO_L2_ENA : '?',
    KeyId.CFG_SIGNAL_GPS_ENA : '?',
    KeyId.CFG_SIGNAL_GAL_ENA : '?',
    KeyId.CFG_SIGNAL_BDS_ENA : '?',
    KeyId.CFG_SIGNAL_QZSS_ENA : '?',
    KeyId.CFG_SIGNAL_GLO_ENA : '?'
}

# ------------------------------------------------------------------------------

def parse_key_id_value(key_id, value):
    """
    Get the value from packed format
    
    >>> parse_key_id_value(KeyId.CFG_RATE_MEAS, bytes([0xe8, 0x03]))
    1000
    >>> parse_key_id_value(KeyId.CFG_SIGNAL_GPS_ENA, bytes([0x00]))
    0
    >>> parse_key_id_value(KeyId.CFG_SIGNAL_GPS_ENA, bytes([0x01]))
    1
    """

    # Check if value is iterable
    try:
        iter(value)
    except TypeError:
        value = [value] # need to convert to iterable

    fmt = KEY_ID_VALUE_FORMATS.get(key_id, 'B')

    bytes_value = bytes(value)
    parsed_value = struct.unpack_from('<{}'.format(fmt), bytes_value)[0]

    return parsed_value

# ------------------------------------------------------------------------------

def build_key_id_value(key_id, value):
    """
    Pack the value according to the key_id

    >>> build_key_id_value(KeyId.CFG_RATE_MEAS, 1000).hex()
    'e803'
    """
   
    fmt = '<{}'.format(KEY_ID_VALUE_FORMATS.get(key_id, 'B'))
    return struct.pack(fmt, value)

# ------------------------------------------------------------------------------

def __go_to_next_packet__(stream, timeout):
    """
    Skip bytes in the stream until a new preamble is found
    """

    byte_1 = None
    byte_2 = None

    t_start = time.time()

    while(True):

        elapsed_time = time.time() - t_start
        
        if byte_1 != PREAMBLE[0]:
            byte_1 = int.from_bytes(stream.read(1), "big")
            
        else:
            byte_2 = int.from_bytes(stream.read(1), "big")

            if byte_2 == PREAMBLE[1]:
                break
            if byte_2 == PREAMBLE[0]:
                byte_1 = byte_2

        if elapsed_time > timeout:
            raise TimeoutError('Unable to find a packet in the given time frame')

    return True

# ------------------------------------------------------------------------------

def __ack_nck_corresponds_to_packet__(expected_type, ack_parsed_packet):

    class_id = ack_parsed_packet.clsID
    msg_id = ack_parsed_packet.msgID

    incoming_type = make_packet_type(class_id, msg_id)

    return expected_type == incoming_type

# ------------------------------------------------------------------------------

def __extract_cfg_data__(cfg_data):

    byte_stream = io.BytesIO(cfg_data)

    key_values = {}

    while True:
        raw = byte_stream.read(4)
        if len(raw) == 0:
            break

        key_id = KeyId(struct.unpack_from('<L', raw)[0])
        N = get_n_bytes_of_config_value(key_id)

        raw = byte_stream.read(N)
        if len(raw) == 0:
            break

        fmt = '<{}s'.format(N)
        value = struct.unpack_from(fmt, raw)[0]
        key_values[key_id] = value

    byte_stream.close()

    return key_values
