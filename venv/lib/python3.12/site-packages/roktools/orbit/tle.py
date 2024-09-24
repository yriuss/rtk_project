import datetime
import math
from typing import List, Union, IO

from ..gnss.types import ConstellationId, Satellite

from .kepler import Kepler, compute_semi_major_axis

REV_PER_DAY_TO_RAD_PER_S = math.tau / 86400.0


def parse_decimal_point(number: str) -> float:
    """
    Parse float numers with decimal point assumed and exponent

    >>> parse_decimal_point("0006703")
    0.0006703
    >>> f'{parse_decimal_point("-11606-4"):.4e}'
    '-1.1606e-05'
    >>> f'{parse_decimal_point("-11606+4"):.4e}'
    '-1.1606E+03'
    >>> f'{parse_decimal_point(" 11606-4"):.4e}'
    '1.1606e-05'
    """

    has_exponent = number[-2] == '-' or number[-2] == '+'

    if has_exponent:
        power = math.pow(10, int(number[-2:]))
        n = len(number[1:-2])
        return float(number[0:-2]) * math.pow(10, -n) * power
    else:
        power = math.pow(10, -len(number))
        return float(number) * power


def calculate_checksum(line: str) -> int:
    checksum = 0

    for character in line:
        if character.isdigit():
            checksum += int(character)
        elif character == '-':
            checksum += 1

    return checksum % 10


class TLE(object):

    def __init__(self, line1: str, line2: str, label=None) -> 'TLE':

        # Check integrity of TLE input
        id_1 = line1[2:7]
        id_2 = line2[2:7]
        if id_1 != id_2:
            raise RuntimeError(f'TLE lines correspond to different satellite catalog numbers {id_1} == {id_2}')

        chk = calculate_checksum(line1[:-1])
        chk_expected = int(line1[-1])
        if chk != chk_expected:
            raise RuntimeError(f'Invalid checksum for TLE line 1, got {chk}, expected {chk_expected}')
        chk = calculate_checksum(line2[:-1])
        chk_expected = int(line2[-1])
        if chk != chk_expected:
            raise RuntimeError(f'Invalid checksum for TLE line 2, got {chk}, expected {chk_expected}')

        self.label = label
        self.id = int(id_1)

        line = line1

        # Epoch
        year = int(line[18:20]) + 2000
        day_of_year = float(line[20:32])
        doy = int(day_of_year)
        fraction_of_day = day_of_year - doy
        f_hour = 24 * fraction_of_day
        hour = int(f_hour)
        f_min = 60 * (f_hour - hour)
        min = int(f_min)
        f_sec = 60 * (f_min - min)
        sec = int(f_sec)
        fraction_of_second = f_sec - sec
        datetime_str = f'{year} {doy} {hour} {min} {sec}'
        epoch = datetime.datetime.strptime(datetime_str, '%Y %j %H %M %S')
        offset = datetime.timedelta(seconds=fraction_of_second)
        self.toe = epoch + offset

        # Mean motion
        self.n_dot_rad_per_s = float(line[33:43]) * REV_PER_DAY_TO_RAD_PER_S
        self.n_dot_dot_rad_per_s2 = parse_decimal_point(line[44:52]) * REV_PER_DAY_TO_RAD_PER_S / 86400.0

        # Atmospheric drag coefficient
        self.bstar = parse_decimal_point(line[53:61])

        line = line2

        self.inclination_rad = math.radians(float(line[8:16]))
        self.RAAN_rad = math.radians(float(line[17:25]))
        self.eccentricity = parse_decimal_point(line[26:33])
        self.arg_perigee_rad = math.radians(float(line[34:42]))
        self.mean_anomaly_rad = math.radians(float(line[43:51]))
        self.mean_motion_rad_per_s = float(line[52:63]) * REV_PER_DAY_TO_RAD_PER_S

    def __repr__(self) -> str:
        out = f"""
        label: {self.label}
        id: {self.id}
        toe: {self.toe}
        n_dot[rad/s]: {self.n_dot_rad_per_s}
        n_dot_dot[rad/s^2]: {self.n_dot_dot_rad_per_s2}
        b*: {self.bstar}
        inclination[deg]: {math.degrees(self.inclination_rad)}
        RAAN[deg]: {math.degrees(self.RAAN_rad)}
        eccentricity: {self.eccentricity}
        arg_perigee[deg]: {math.degrees(self.arg_perigee_rad)}
        mean anomaly[deg]: {math.degrees(self.mean_anomaly_rad)}
        mean motion[rad/s]: {self.mean_motion_rad_per_s}
        """
        return out

    def get_satellite(self) -> Satellite:

        constellation = get_constellation_from_label(self.label)
        prn = self.id

        return Satellite(constellation, prn)

    def to_kepler(self) -> Kepler:
        """
        Rough mapping of the Two Line Element set into Keplerian parameters

        Use this method at your own risk, TLE elements cannot be considered
        classical orbital elements

        Based on https://blog.hardinglabs.com/tle-to-kep.html
        """

        a_m = compute_semi_major_axis(self.mean_motion_rad_per_s)
        return Kepler(self.toe,
                      a_m,  self.eccentricity, self.inclination_rad, self.RAAN_rad,
                      self.arg_perigee_rad, self.mean_anomaly_rad,
                      delta_n_dot_rad_per_s=self.n_dot_rad_per_s)


def read_celestrak(tle_source: Union[str, IO]) -> List[TLE]:
    """
    Read a NORAD General Perturbations (GP) file in TLE format, that can be found
    at https://celestrak.org/NORAD/elements/
    """
    tles = []

    # Check if tle_source is a string (filename) or a file handler
    if isinstance(tle_source, str):
        with open(tle_source, "r") as fh:
            _read_celestrak_from_stream(fh, tles)
    else:
        _read_celestrak_from_stream(tle_source, tles)

    return tles


def _read_celestrak_from_stream(file_handle: IO[str], tles: List[TLE]) -> None:

    while True:
        lines = [next(file_handle, None) for _ in range(3)]
        if any(line is None for line in lines):
            break

        lines = [line.strip() for line in lines]
        tles.append(TLE(lines[1], lines[2], label=lines[0]))


def get_constellation_from_label(label: str) -> ConstellationId:

    out = ConstellationId.UNKNOWN

    if 'ONEWEB' in label:
        out = ConstellationId.ONEWEB
    elif 'LEMUR' in label:
        out = ConstellationId.SPIRE
    elif 'STARLINK' in label:
        out = ConstellationId.STARLINK

    return out
