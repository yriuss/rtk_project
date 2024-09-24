import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, asdict
import datetime
import enum
import math
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Union, IO
import sys

from . import logger
from . import time
from .constants import SPEED_OF_LIGHT
from .file import process_filename_or_file_handler, skip_lines
from .gnss.types import ConstellationId, Band, TrackingChannel, Satellite

from .orbit.tle import TLE, read_celestrak
from .orbit.kepler import Kepler

RINEX_LINE_SYS_OBS_TYPES = "SYS / # / OBS TYPES"

RANDOM_STR = 'random'
ZERO_STR = 'zero'

SAT_STR = 'sat'
EPOCH_STR = 'epoch'
A_M_STR = 'a_m'
ECCENTRICITY_STR = 'eccentricity'
INCLINATION_DEG_STR = 'inclination_deg'
RIGHT_ASCENSION_DEG_STR = 'raan_deg'
ARG_PERIGEE_DEG_STR = 'arg_perigee_deg'
TRUE_ANOMALY_DEG_STR = 'true_anomaly_deg'

RINEX_BAND_MAP = {
    ConstellationId.GPS: {
        '1': Band.L1,
        '2': Band.L2,
        '5': Band.L5
    },
    ConstellationId.GLONASS: {
        '1': Band.G1,
        '4': Band.G1a,
        '2': Band.G2,
        '6': Band.G2a,
        '3': Band.G3
    },
    ConstellationId.GALILEO: {
        '1': Band.E1,
        '5': Band.E5a,
        '7': Band.E5b,
        '8': Band.E5,
        '6': Band.E6
    },
    ConstellationId.BEIDOU: {
        '2': Band.B1_2,
        '1': Band.B1,
        '5': Band.B2a,
        '7': Band.B2b,
        '8': Band.B2,
        '6': Band.B3
    },
    ConstellationId.QZSS: {
        '1': Band.L1,
        '2': Band.L2,
        '5': Band.L5,
        '6': Band.L6
    },
    ConstellationId.IRNSS: {
        '5': Band.L5,
        '9': Band.S
    },
    ConstellationId.SBAS: {
        '1': Band.L1,
        '5': Band.L5
    }
}


class EphType(enum.Enum):

    GPS_LNAV = f'{ConstellationId.GPS.value}_LNAV'
    GPS_CNAV = f'{ConstellationId.GPS.value}_CNAV'
    GPS_CNV2 = f'{ConstellationId.GPS.value}_CNV2'
    GAL_INAV = f'{ConstellationId.GALILEO.value}_INAV'
    GAL_FNAV = f'{ConstellationId.GALILEO.value}_FNAV'
    GLO_FDMA = f'{ConstellationId.GLONASS.value}_FDMA'
    QZS_LNAV = f'{ConstellationId.QZSS.value}_LNAV'
    QZS_CNAV = f'{ConstellationId.QZSS.value}_CNAV'
    QZS_CNV2 = f'{ConstellationId.QZSS.value}_CNV2'
    BDS_D1 = f'{ConstellationId.BEIDOU.value}_D1'
    BDS_D2 = f'{ConstellationId.BEIDOU.value}_D2'
    BDS_CNV1 = f'{ConstellationId.BEIDOU.value}_CNV1'
    BDS_CNV2 = f'{ConstellationId.BEIDOU.value}_CNV2'
    BDS_CNV3 = f'{ConstellationId.BEIDOU.value}_CNV3'
    SBS = f'{ConstellationId.SBAS.value}_SBAS'
    IRN_LNAV = f'{ConstellationId.IRNSS.value}_LNAV'
    LEO = f'{ConstellationId.LEO.value}'
    SPIRE = f'{ConstellationId.SPIRE.value}'
    STARLINK = f'{ConstellationId.STARLINK.value}'
    ONEWEB = f'{ConstellationId.ONEWEB.value}'

    @staticmethod
    def from_string(value: str) -> 'EphType':
        """
        Get the Ephemerides type from an input string (or raise an exception if not found)

        >>> EphType.from_string('G_LNAV')
        <EphType.GPS_LNAV: 'G_LNAV'>
        """

        for member in EphType:
            if member.value == value:
                return member

        raise ValueError(f"Value [ {value} ] could not be mapped into a Rinex 4 ephemeris type")


class RinexSatIdProvider(ABC):
    """
    This abstract class provides an interface to provide the RINEX Satellite ID and constellation
    """
    @abstractmethod
    def get_constellation_letter(self) -> str:
        pass

    def get_sat_number(self, norad_id) -> int:
        return norad_id


class RinexSatIdFactory:
    @staticmethod
    def create(constellation: ConstellationId) -> RinexSatIdProvider:
        if constellation == ConstellationId.STARLINK:
            return StarlinkRNXId()
        elif constellation == ConstellationId.SPIRE:
            return SpireRNXId()
        elif constellation == ConstellationId.ONEWEB:
            return OneWebRNXId()
        else:
            raise ValueError("Invalid constellation")


class StarlinkRNXId(RinexSatIdProvider):
    def get_constellation_letter(self) -> str:
        return ConstellationId.STARLINK.value


class SpireRNXId(RinexSatIdProvider):
    def get_constellation_letter(self) -> str:
        return ConstellationId.SPIRE.value


class OneWebRNXId(RinexSatIdProvider):
    def get_constellation_letter(self) -> str:
        return ConstellationId.ONEWEB.value


@dataclass
class ObservableType(object):
    type: str
    channel: TrackingChannel

    @staticmethod
    def from_string(observable_type: str) -> 'ObservableType':
        return ObservableType(observable_type[0], TrackingChannel.from_rinex3_code(observable_type))

    def __repr__(self):
        return f'{self.type:1s}{self.channel}'


@dataclass
class ObservableValue(object):
    value: float
    lli: int
    snr: int


@dataclass
class Clock(object):
    bias_s: float
    drift_s_per_s: float
    drift_rate_s_per_s2: float

    ref_epoch: datetime = None


class ClockModel(ABC):
    """
    This abstract class provides with an interface to provide with a Clock model
    """

    @abstractmethod
    def get_clock(self, satellite: Satellite, epoch: datetime.datetime) -> Clock:
        """
        Get the clock for a satellite and an epoch
        """
        pass


class ZeroClockModel(ClockModel):
    """
    Clock model with 0 bias and drift
    """

    def get_clock(self, satellite: Satellite, epoch: datetime.datetime) -> Clock:
        return Clock(0, 0, 0)


class GnssRandomClock(ClockModel):
    """
    Generate a clock model with random realizations for the clock bias and
    drift.
    """

    def __init__(self, bias_max_s: float = 0.0005, drift_max_s_per_s: float = 1.0e-11):
        """
        Initialize the object with the threshold values for the bias and drift.
        In certain works, the typical values of bias and drift for GPS, Galileo
        and Beidou are usually contained between -/+0.5ms and-/+1e-11 respectively
        """
        self.bias_max_s = bias_max_s
        self.drift_max_s_per_s = drift_max_s_per_s

        # Internal memory to store the state of the previous clock
        self.clocks = {}

    def get_clock(self, satellite: Satellite, epoch: datetime.datetime) -> Clock:

        clock = self.clocks.get(satellite, None)

        if clock is None:
            bias_s = np.random.uniform(low=-self.bias_max_s, high=self.bias_max_s)
            drift_s_per_s = np.random.uniform(low=-self.drift_max_s_per_s, high=self.drift_max_s_per_s)

            clock = Clock(bias_s, drift_s_per_s, 0.0, ref_epoch=epoch)

            self.clocks[satellite] = clock

        # Compute the bias for the updated clock
        dt = (epoch - clock.ref_epoch).total_seconds() if clock.ref_epoch else 0.0
        clock = Clock(clock.bias_s + clock.drift_s_per_s * dt, clock.drift_s_per_s, 0.0)

        return clock


@dataclass
class CodeBiases(ABC):
    """
    This abstract class provides with an interface to provide with the code biases
    """

    @abstractmethod
    def get_base_tgd(self) -> float:
        """
        Get the base group delay
        """
        return 0.0

    @abstractmethod
    def get_code_bias(self, channel: TrackingChannel) -> float:
        """
        Get the code bias for the given channel
        """
        return 0.0


class ZeroCodeBiases(CodeBiases):

    def get_base_tgd(self) -> float:
        return super().get_base_tgd()

    def get_code_bias(self, channel: TrackingChannel) -> float:
        return super().get_code_bias(channel)

@dataclass
class LEOCodeBiases(CodeBiases):
    """
    Class to handle the code biases for the generic LEO constellation
    """
    tgd_s: float
    isc_s9c_s: float

    def get_base_tgd(self) -> float:
        return self.tgd_s

    @abstractmethod
    def get_code_bias(self, channel: TrackingChannel) -> float:

        if channel == TrackingChannel.from_string('9C'):
            return self.isc_s9c_s

        raise ValueError(f'Tracking channel [ {channel} ] not supported by LEO constellations')


@dataclass
class Record(object):
    epoch: datetime.datetime
    sat: Satellite
    channel: TrackingChannel
    range: float
    phase: float
    doppler: float
    snr: float
    flags: str
    slip: int

    def set_value(self, observable_type: ObservableType, observable_value: ObservableValue) -> None:

        if observable_type.type == 'C':
            self.range = observable_value.value
        elif observable_type.type == 'L':
            self.phase = observable_value.value
        elif observable_type.type == 'D':
            self.doppler = observable_value.value
        elif observable_type.type == 'S':
            self.snr = observable_value.value
        else:
            raise TypeError(f'Unrecognise observable type {observable_type.type}')

        if observable_value.lli != 0:
            self.slip = 1
            self.flags = '00000100'

    def aslist(self) -> list:

        return [self.epoch, self.sat.constellation.value, str(self.sat), str(self.channel),
                f'{self.sat}{self.channel}', self.range, self.phase, self.doppler, self.snr, self.flags, self.slip]

    @staticmethod
    def get_list_fieldnames() -> List[str]:
        return ["epoch", "constellation", "sat", "channel", "signal", "range", "phase", "doppler", "snr", "flag", "slip"]


class EpochFlag(enum.Enum):

    OK = 0
    POWER_FAILURE = 1
    MOVING_ANTENNA = 2
    NEW_SITE = 3
    HEADER_INFO = 4
    EXTERNAL_EVENT = 5

    @staticmethod
    def get_from_line(line: str) -> 'EpochFlag':
        """
        Extract the epoch flag from the incoming line

        >>> EpochFlag.get_from_line(">                              4 95")
        <EpochFlag.HEADER_INFO: 4>
        >>> EpochFlag.get_from_line("> 2023 08 03 12 00  8.0000000  0 38")
        <EpochFlag.OK: 0>
        >>> EpochFlag.get_from_line(None)
        Traceback (most recent call last):
        ...
        ValueError: The line [ None ] does not have an Epoch Flag
        >>> EpochFlag.get_from_line("")
        Traceback (most recent call last):
        ...
        ValueError: The line [  ] does not have an Epoch Flag
        >>> EpochFlag.get_from_line("> 2023 08 03 1")
        Traceback (most recent call last):
        ...
        ValueError: The line [ > 2023 08 03 1 ] does not have an Epoch Flag
        """

        INDEX = 31

        if line and len(line) >= INDEX + 1 and line[0] == '>':
            epoch_flag = int(line[INDEX])
            return EpochFlag(epoch_flag)

        raise ValueError(f'The line [ {line} ] does not have an Epoch Flag')


class FilePeriod(enum.Enum):

    DAILY = 86400
    HOURLY = 3600
    QUARTERLY = 900
    UNDEFINED = 0

    @staticmethod
    def from_string(string):
        """
        Get the FilePeriod from a string

        >>> FilePeriod.from_string('daily')
        <FilePeriod.DAILY: 86400>

        >>> FilePeriod.from_string('DAILY')
        <FilePeriod.DAILY: 86400>
        """

        if (string.lower() == 'daily'):
            return FilePeriod.DAILY
        elif (string.lower() == 'quarterly'):
            return FilePeriod.QUARTERLY
        elif (string.lower() == 'hourly'):
            return FilePeriod.HOURLY
        else:
            return FilePeriod.UNDEFINED

    @staticmethod
    def list():
        """ Return a list of the available valid periodicities """
        return list([v.name for v in FilePeriod if v.value > 0])

    def build_rinex3_epoch(self, epoch):
        """
        Construct a Rinex-3-like epoch string

        >>> epoch = datetime.datetime(2020, 5, 8, 9, 29, 20)
        >>> FilePeriod.QUARTERLY.build_rinex3_epoch(epoch)
        '20201290915_15M'

        >>> FilePeriod.HOURLY.build_rinex3_epoch(epoch)
        '20201290900_01H'

        >>> FilePeriod.DAILY.build_rinex3_epoch(epoch)
        '20201290000_01D'
        """

        hour = epoch.hour if self != FilePeriod.DAILY else 0

        day_seconds = (epoch - epoch.combine(epoch, datetime.time())).total_seconds()

        minute = get_quarter_str(day_seconds) if self == FilePeriod.QUARTERLY else 0

        date_str = epoch.strftime('%Y%j')

        return '{}{:02d}{:02d}_{}'.format(date_str, hour, minute, self)

    def __str__(self):

        if self.value == FilePeriod.DAILY.value:
            return '01D'
        elif self.value == FilePeriod.QUARTERLY.value:
            return '15M'
        elif self.value == FilePeriod.HOURLY.value:
            return '01H'
        else:
            raise ValueError('Undefined FilePeriod value')


# ------------------------------------------------------------------------------

def strftime(epoch, fmt):
    """

    >>> epoch = datetime.datetime(2019, 8, 3, 10, 10, 10)
    >>> strftime(epoch, "ebre215${rinexhour}${rinexquarter}.19o")
    'ebre215k00.19o'

    >>> epoch = datetime.datetime(2019, 8, 3, 10, 50, 10)
    >>> strftime(epoch, "ebre215${RINEXHOUR}${rinexQUARTER}.19o")
    'ebre215k45.19o'

    >>> epoch = datetime.datetime(2019, 8, 3, 0, 0, 0)
    >>> strftime(epoch, "ebre215${rinexhour}${rinexquarter}.19o")
    'ebre215a00.19o'

    >>> epoch = datetime.datetime(2019, 8, 3, 23, 50, 10)
    >>> strftime(epoch, "ebre215${rinexhour}${rinexquarter}.19o")
    'ebre215x45.19o'
    """

    RINEX_HOUR = "abcdefghijklmnopqrstuvwxyz"

    PATTERN_HOUR = re.compile(r"\$\{rinexhour\}", re.IGNORECASE)
    PATTERN_QUARTER = re.compile(r"\$\{rinexquarter\}", re.IGNORECASE)

    hour = RINEX_HOUR[epoch.hour]
    quarter = get_quarter_str(epoch.minute * 60 + epoch.second)

    fmt = PATTERN_HOUR.sub(f"{hour}", fmt)
    fmt = PATTERN_QUARTER.sub(f"{quarter:02d}", fmt)

    return fmt


def get_quarter_str(seconds):
    """
    Get the Rinex quarter string ("00", "15", "30", "45") for a given number of seconds

    >>> get_quarter_str(100)
    0
    >>> get_quarter_str(920)
    15
    >>> get_quarter_str(1800)
    30
    >>> get_quarter_str(2900)
    45
    >>> get_quarter_str(3600 + 900)
    15
    """

    mod_seconds = seconds % 3600

    if mod_seconds < 900:
        return 0
    elif mod_seconds < 1800:
        return 15
    elif mod_seconds < 2700:
        return 30
    else:
        return 45


def get_channels(observables: List[ObservableType]) -> List[TrackingChannel]:
    """
    Get the channel list from a list of observables

    >>> C1C = ObservableType.from_string("C1C")
    >>> L1C = ObservableType.from_string("L1C")
    >>> C1W = ObservableType.from_string("C1W")
    >>> C2W = ObservableType.from_string("C2W")
    >>> L2W = ObservableType.from_string("L2W")
    >>> C2L = ObservableType.from_string("C2L")
    >>> L2L = ObservableType.from_string("L2L")
    >>> C5Q = ObservableType.from_string("C5Q")
    >>> L5Q = ObservableType.from_string("L5Q")
    >>> get_channels([C1C, L1C, C1W, C2W, L2W, C2L, L2L, C5Q, L5Q])
    [1C, 1W, 2W, 2L, 5Q]
    """

    res = []
    for observable in observables:
        channel = observable.channel

        if channel not in res:
            res.append(channel)

    return res


def get_obs_mapping(lines: List[str]) -> dict:
    """
    Get the observable mappings for a constellation

    >>> line = "G    9 C1C L1C C1W C2W L2W C2L L2L C5Q L5Q                  SYS / # / OBS TYPES"
    >>> get_obs_mapping([line])
    {'G': [C1C, L1C, C1W, C2W, L2W, C2L, L2L, C5Q, L5Q]}

    >>> lines = ["G   20 C1C L1C D1C S1C C1W L1W D1W S1W C2W L2W D2W S2W C2L  SYS / # / OBS TYPES", \
                 "       L2L D2L S2L C5Q L5Q D5Q S5Q                          SYS / # / OBS TYPES"]
    >>> get_obs_mapping(lines)
    {'G': [C1C, L1C, D1C, S1C, C1W, L1W, D1W, S1W, C2W, L2W, D2W, S2W, C2L, L2L, D2L, S2L, C5Q, L5Q, D5Q, S5Q]}
    """

    constellation = None
    values = None

    for line in lines:

        constellation_letter = line[0] if line[0] != ' ' else None
        values_partial = [ObservableType.from_string(s) for s in line[6:60].split()]

        if constellation_letter:
            constellation = constellation_letter
            values = values_partial

        else:
            values.extend(values_partial)

    return {constellation: values}


def to_dataframe(rinex_file: str) -> pd.DataFrame:

    with open(rinex_file, 'r') as file:

        constellation_observables = {}
        records = []

        # Header parsing
        for line in file:

            if "END OF HEADER" in line:
                break

            if RINEX_LINE_SYS_OBS_TYPES in line:
                lines = [line]

                n_observables = int(line[1:6])
                n_extra_lines = (n_observables - 1) // 13 if n_observables > 13 else 0
                for _ in range(n_extra_lines):
                    lines.append(next(file))

                constellation_observables.update(get_obs_mapping(lines))

        # Body parsing
        for line in file:

            epoch_flag = EpochFlag.get_from_line(line)

            if epoch_flag == EpochFlag.OK:

                epoch, _, n_lines = _parse_rnx3_epoch(line)

                for _ in range(n_lines):

                    line = next(file)

                    constellation = line[0]

                    observable_types = constellation_observables.get(constellation, None)
                    if observable_types is None:
                        continue

                    n_obs = len(observable_types)
                    channels = get_channels(observable_types)
                    satellite, observable_values = _parse_obs_line(line, n_obs)

                    epoch_records = [
                        Record(epoch, satellite, channel, math.nan, math.nan, math.nan, math.nan, '00000000', 0)
                        for channel in channels]

                    # Create a dictionary with channels as keys and records as values
                    epoch_records_dict = {record.channel: record for record in epoch_records}

                    for observable_type, observable_value in zip(observable_types, observable_values):

                        record = epoch_records_dict.get(observable_type.channel, None)
                        if record is not None:
                            record.set_value(observable_type, observable_value)

                    records.extend(epoch_records)

            else:
                n_lines_to_skip = int(line[33:])
                for _ in range(n_lines_to_skip):
                    line = next(file)

        dataframe = pd.DataFrame([record.aslist() for record in records], columns=Record.get_list_fieldnames())

        return dataframe


class Obs:

    def __init__(self, filename: str):
        self.filename = filename
        self.data = to_dataframe(self.filename)
        timetags = sorted(set([ts.to_pydatetime() for ts in self.data['epoch']]))
        if len(timetags) > 1:
            self.interval = time.get_interval(timetags)

    def compute_detrended_code_minus_carrier(self) -> pd.DataFrame:
        self.count_cycle_slips()

        grouped_data = self.data.groupby(['channel', 'sat', 'slipc'], group_keys=False)
        self.data = grouped_data.apply(lambda df: self.__compute_grouped_detrended_cmc(df))

    def count_cycle_slips(self):
        grouped_data = self.data.groupby(['channel', 'sat'])
        self.data['slipc'] = grouped_data['slip'].transform(lambda slip: slip.cumsum())

    def __compute_grouped_detrended_cmc(self, grouped_df):
        # grouped_df is rinex_obs.data grouped by 'channel', 'sat' and 'slipc'
        const = grouped_df['constellation'].iloc[0]
        constellation_id = ConstellationId.from_string(const)

        chan = grouped_df['channel'].iloc[0]
        band_frequency = RINEX_BAND_MAP[constellation_id][chan[0]]
        wavelength = SPEED_OF_LIGHT / band_frequency

        cmc = grouped_df['range'] - grouped_df['phase'] * wavelength

        CMC_ROLL_WINDOW_SAMPLES = 600 / self.interval
        if len(cmc) >= CMC_ROLL_WINDOW_SAMPLES:
            trend = cmc.rolling(20).median()
        else:
            trend = cmc.mean()

        grouped_df['cmc_detrended'] = cmc - trend

        return grouped_df


@dataclass
class NavBlock(ABC):
    """
    Abstract class for RINEX Navigation block
    """
    satellite: Satellite
    epoch: datetime

    @staticmethod
    def csv_fields() -> str:
        return None

    def to_csv(self) -> str:
        # Get all field names from the __dataclass_fields__ attribute
        field_names = [field.name for field in fields(self)]

        # Get the value of each field from the instance dictionary
        field_values = [getattr(self, field_name) for field_name in field_names]

        # Convert field values to strings and join them with commas
        body = ",".join(map(str, field_values))

        return f'{self.get_type().value},{body}'

    # Custom function to convert each NavBlock to a dict with string satellite
    def to_dict(self):
        block_dict = asdict(self)
        block_dict['satellite'] = str(self.satellite)  # Convert satellite to string
        return block_dict

    @abstractmethod
    def get_type(self) -> EphType:
        return None


@dataclass
class GpsLnavNavBlock(NavBlock):
    clock_bias_s: float
    clock_drift_sps: float
    clock_drift_rate_sps2: float

    iode: int
    crs: float
    deltan: float
    M0: float

    cuc: float
    e: float
    cus: float
    sqrtA: float

    toe_sow: float
    cic: float
    OMEGA0: float
    cis: float

    i0: float
    crc: float
    omega: float
    OMEGA_DOT: float

    idot: float
    codesL2: int
    toe_week: int
    l2p_flag: int

    accuracy: float
    health: int
    tgd: float
    iodc: int

    tx_time_tow: float
    fit_interval: int

    def csv_fields(self) -> str:
        return f"{self.get_type().value},{_SAT_STR},{_EPOCH_STR},\
{_CLK_BIAS_STR},{_CLK_DRIFT_STR},{_CLK_DRIFT_RATE_STR},\
{_IODE_STR},{_CRS_STR},{_DELTAN_STR},{_M0_STR},\
{_CUC_STR},{_E_STR},{_CUS_STR},{_SQRTA_STR},\
{_TOE_SOW_STR},{_CIC_STR},{_OMEGA0_STR},{_CIS_STR},\
{_I0_STR},{_CRC_STR},{_OMEGA_STR},{_OMEGA_DOT_STR},\
{_IDOT_STR},{_CODESL2_STR},{_TOE_WEEK_STR},{_L2PFLAG_STR},\
{_ACCURACY_STR},{_HEALTH_STR},{_TGD_STR},{_IODC_STR},\
{_TX_TIME_TOW_STR},{_FIT_INTERVAL_STR}"

    def get_type(self) -> EphType:
        return EphType.GPS_LNAV


@dataclass
class GpsCnavNavBlock(NavBlock):
    clock_bias_s: float
    clock_drift_sps: float
    clock_drift_rate_sps2: float

    adot: float
    crs: float
    deltan: float
    M0: float

    cuc: float
    e: float
    cus: float
    sqrtA: float

    top: float
    cic: float
    OMEGA0: float
    cis: float

    i0: float
    crc: float
    omega: float
    OMEGA_DOT: float

    idot: float
    deltan_dot: float
    urai_ned0: float
    urai_ned1: float

    urai_ed: float
    health: int
    tgd: float
    urai_ned2: float

    isc_l1ca: float
    isc_l2c: float
    isc_l5i5: float
    isc_l5q5: float

    tx_time_tow: float
    wn_op: int

    def csv_fields(self) -> str:
        return f"{self.get_type().value},{_SAT_STR},{_EPOCH_STR},\
{_CLK_BIAS_STR},{_CLK_DRIFT_STR},{_CLK_DRIFT_RATE_STR},\
{_ADOT_STR},{_CRS_STR},{_DELTAN_STR},{_M0_STR},\
{_CUC_STR},{_E_STR},{_CUS_STR},{_SQRTA_STR},\
{_T_OP_STR},{_CIC_STR},{_OMEGA0_STR},{_CIS_STR},\
{_I0_STR},{_CRC_STR},{_OMEGA_STR},{_OMEGA_DOT_STR},\
{_IDOT_STR},{_DELTAN_DOT_STR},{_URAI_NED0_STR},{_URAI_NED1_STR},\
{_URAI_ED_STR},{_HEALTH_STR},{_TGD_STR},{_URAI_NED2_STR},\
{_ISC_L1CA_STR},{_ISC_L2C_STR},{_ISC_L5I5_STR},{_ISC_L5Q5_STR},\
{_TX_TIME_TOW_STR},{_WN_OP_STR}"

    def get_type(self) -> EphType:
        return EphType.GPS_CNAV


@dataclass
class GpsCnv2NavBlock(NavBlock):
    clock_bias_s: float
    clock_drift_sps: float
    clock_drift_rate_sps2: float

    adot: float
    crs: float
    deltan: float
    M0: float

    cuc: float
    e: float
    cus: float
    sqrtA: float

    top: float
    cic: float
    OMEGA0: float
    cis: float

    i0: float
    crc: float
    omega: float
    OMEGA_DOT: float

    idot: float
    deltan_dot: float
    urai_ned0: float
    urai_ned1: float

    urai_ed: float
    health: int
    tgd: float
    urai_ned2: float

    isc_l1ca: float
    isc_l2c: float
    isc_l5i5: float
    isc_l5q5: float

    isc_l1cd: float
    isc_l1cp: float

    tx_time_tow: float
    wn_op: int

    def csv_fields(self) -> str:
        return f"{self.get_type().value},{_SAT_STR},{_EPOCH_STR},\
{_CLK_BIAS_STR},{_CLK_DRIFT_STR},{_CLK_DRIFT_RATE_STR},\
{_ADOT_STR},{_CRS_STR},{_DELTAN_STR},{_M0_STR},\
{_CUC_STR},{_E_STR},{_CUS_STR},{_SQRTA_STR},\
{_T_OP_STR},{_CIC_STR},{_OMEGA0_STR},{_CIS_STR},\
{_I0_STR},{_CRC_STR},{_OMEGA_STR},{_OMEGA_DOT_STR},\
{_IDOT_STR},{_DELTAN_DOT_STR},{_URAI_NED0_STR},{_URAI_NED1_STR},\
{_URAI_ED_STR},{_HEALTH_STR},{_TGD_STR},{_URAI_NED2_STR},\
{_ISC_L1CA_STR},{_ISC_L2C_STR},{_ISC_L5I5_STR},{_ISC_L5Q5_STR},\
{_ISC_L1CD_STR},{_ISC_L1CP_STR},\
{_TX_TIME_TOW_STR},{_WN_OP_STR}"

    def get_type(self) -> EphType:
        return EphType.GPS_CNV2


@dataclass
class GalNavBlock(NavBlock):
    clock_bias_s: float
    clock_drift_sps: float
    clock_drift_rate_sps2: float

    iodnav: float
    crs: float
    deltan: float
    M0: float

    cuc: float
    e: float
    cus: float
    sqrtA: float

    toe_gal_tow: int
    cic: float
    OMEGA0: float
    cis: float

    i0: float
    crc: float
    omega: float
    OMEGA_DOT: float

    idot: float
    data_source: int
    toe_gal_week: int

    sisa: float
    health: int
    bgd_e5a_e1: float
    bgd_e5b_e1: float

    tx_time_tow: float

    def csv_fields(self) -> str:
        return f"{self.get_type().value},{_SAT_STR},{_EPOCH_STR},\
{_CLK_BIAS_STR},{_CLK_DRIFT_STR},{_CLK_DRIFT_RATE_STR},\
{_IODNAV_STR},{_CRS_STR},{_DELTAN_STR},{_M0_STR},\
{_CUC_STR},{_E_STR},{_CUS_STR},{_SQRTA_STR},\
{_TOE_GAL_TOW_STR},{_CIC_STR},{_OMEGA0_STR},{_CIS_STR},\
{_I0_STR},{_CRC_STR},{_OMEGA_STR},{_OMEGA_DOT_STR},\
{_IDOT_STR},{_DATA_SOURCES_STR},{_TOE_GAL_WEEK_STR},,\
{_SISA_STR},{_HEALTH_STR},{_BGD_E5A_STR},{_BGD_E5B_STR},\
{_TX_TIME_TOW_STR}"

    def get_type(self) -> EphType:
        INAV_MASK = 0b101
        FNAV_MASK = 0b10

        if self.data_source & INAV_MASK:
            return EphType.GAL_INAV
        elif self.data_source & FNAV_MASK:
            return EphType.GAL_FNAV

        raise ValueError(f'Cannot extract the Galileo Eph type from the data sources [ {self.data_source} ]')


@dataclass
class BdsDNavBlock(NavBlock):
    clock_bias_s: float
    clock_drift_sps: float
    clock_drift_rate_sps2: float

    aode: int
    crs: float
    deltan: float
    M0: float

    cuc: float
    e: float
    cus: float
    sqrtA: float

    toe_bdt_tow: int
    cic: float
    OMEGA0: float
    cis: float

    i0: float
    crc: float
    omega: float
    OMEGA_DOT: float

    idot: float
    toe_bdt_week: int

    accuracy: float
    satH1: int
    tgd1: float
    tgd2: float

    tx_time_tow: float
    aodc: int

    eph_type: str

    def csv_fields(self) -> str:
        return f"{self.get_type().value},{_SAT_STR},{_EPOCH_STR},\
{_CLK_BIAS_STR},{_CLK_DRIFT_STR},{_CLK_DRIFT_RATE_STR},\
{_AODE_STR},{_CRS_STR},{_DELTAN_STR},{_M0_STR},\
{_CUC_STR},{_E_STR},{_CUS_STR},{_SQRTA_STR},\
{_TOE_BDT_TOW_STR},{_CIC_STR},{_OMEGA0_STR},{_CIS_STR},\
{_I0_STR},{_CRC_STR},{_OMEGA_STR},{_OMEGA_DOT_STR},\
{_IDOT_STR},{_TOE_BDT_WEEK_STR},\
{_ACCURACY_STR},{_SATH1_STR},{_TGD1_STR},{_TGD2_STR},\
{_TX_TIME_TOW_STR},{_AODC_STR}"

    def get_type(self) -> EphType:
        return EphType.from_string(self.eph_type)


@dataclass
class BdsCnv1NavBlock(NavBlock):
    """
    BEIDOU Navigation message on Beidou-3 B1C signal
    """
    clock_bias_s: float
    clock_drift_sps: float
    clock_drift_rate_sps2: float

    adot: float
    crs: float
    deltan: float
    M0: float

    cuc: float
    e: float
    cus: float
    sqrtA: float

    toe_bdt_tow: int
    cic: float
    OMEGA0: float
    cis: float

    i0: float
    crc: float
    omega: float
    OMEGA_DOT: float

    idot: float
    deltan_dot: float
    sattype: int
    t_op: float

    sisai_oe: float
    sisai_ocb: float
    sisai_oc1: float
    sisai_oc2: float

    isc_b1cd: float
    tgd_b1cp: float
    tgd_b2ap: float

    sismai: float
    health: int
    b1c_integrity_flags: int
    iodc: int

    tx_time_tow: float
    iode: int

    def csv_fields(self) -> str:
        return f"{self.get_type().value},{_SAT_STR},{_EPOCH_STR},\
{_CLK_BIAS_STR},{_CLK_DRIFT_STR},{_CLK_DRIFT_RATE_STR},\
{_ADOT_STR},{_CRS_STR},{_DELTAN_STR},{_M0_STR},\
{_CUC_STR},{_E_STR},{_CUS_STR},{_SQRTA_STR},\
{_TOE_BDT_TOW_STR},{_CIC_STR},{_OMEGA0_STR},{_CIS_STR},\
{_I0_STR},{_CRC_STR},{_OMEGA_STR},{_OMEGA_DOT_STR},\
{_IDOT_STR},{_DELTAN_DOT_STR},{_SAT_TYPE_STR},{_T_OP_STR},\
{_SISAI_OE_STR},{_SISAI_OCB_STR},{_SISAI_OC1_STR},{_SISAI_OC2_STR},\
{_ISC_B1CD_STR},{_TGD_B1CP},{_TGD_B2AP},\
{_SISMAI_STR},{_HEALTH_STR},{_B1C_INTEGRITY_FLAGS_STR},{_IODC_STR},\
{_TX_TIME_TOW_STR},{_IODE_STR}"

    def get_type(self) -> EphType:
        return EphType.BDS_CNV1


@dataclass
class BdsCnv2NavBlock(NavBlock):
    """
    BEIDOU Navigation message on Beidou-3 B2a signal
    """
    clock_bias_s: float
    clock_drift_sps: float
    clock_drift_rate_sps2: float

    adot: float
    crs: float
    deltan: float
    M0: float

    cuc: float
    e: float
    cus: float
    sqrtA: float

    toe_bdt_tow: int
    cic: float
    OMEGA0: float
    cis: float

    i0: float
    crc: float
    omega: float
    OMEGA_DOT: float

    idot: float
    deltan_dot: float
    sattype: int
    t_op: float

    sisai_oe: float
    sisai_ocb: float
    sisai_oc1: float
    sisai_oc2: float

    isc_b2ad: float
    tgd_b1cp: float
    tgd_b2ap: float

    sismai: float
    health: int
    b2a_integrity_flags: int
    iodc: int

    tx_time_tow: float
    iode: int

    def csv_fields(self) -> str:
        return f"{self.get_type().value},{_SAT_STR},{_EPOCH_STR},\
{_CLK_BIAS_STR},{_CLK_DRIFT_STR},{_CLK_DRIFT_RATE_STR},\
{_ADOT_STR},{_CRS_STR},{_DELTAN_STR},{_M0_STR},\
{_CUC_STR},{_E_STR},{_CUS_STR},{_SQRTA_STR},\
{_TOE_BDT_TOW_STR},{_CIC_STR},{_OMEGA0_STR},{_CIS_STR},\
{_I0_STR},{_CRC_STR},{_OMEGA_STR},{_OMEGA_DOT_STR},\
{_IDOT_STR},{_DELTAN_DOT_STR},{_SAT_TYPE_STR},{_T_OP_STR},\
{_SISAI_OE_STR},{_SISAI_OCB_STR},{_SISAI_OC1_STR},{_SISAI_OC2_STR},\
{_ISC_B2AD_STR},{_TGD_B1CP},{_TGD_B2AP},\
{_SISMAI_STR},{_HEALTH_STR},{_B2A_INTEGRITY_FLAGS_STR},{_IODC_STR},\
{_TX_TIME_TOW_STR},{_IODE_STR}"

    def get_type(self) -> EphType:
        return EphType.BDS_CNV2


@dataclass
class BdsCnv3NavBlock(NavBlock):
    """
    BEIDOU Navigation message on Beidou-3 B2b signal
    """
    clock_bias_s: float
    clock_drift_sps: float
    clock_drift_rate_sps2: float

    adot: float
    crs: float
    deltan: float
    M0: float

    cuc: float
    e: float
    cus: float
    sqrtA: float

    toe_bdt_tow: int
    cic: float
    OMEGA0: float
    cis: float

    i0: float
    crc: float
    omega: float
    OMEGA_DOT: float

    idot: float
    deltan_dot: float
    sattype: int
    t_op: float

    sisai_oe: float
    sisai_ocb: float
    sisai_oc1: float
    sisai_oc2: float

    sismai: float
    health: int
    b2b_integrity_flags: int
    tgd_b2bi: int

    tx_time_tow: float

    def csv_fields(self) -> str:
        return f"{self.get_type().value},{_SAT_STR},{_EPOCH_STR},\
{_CLK_BIAS_STR},{_CLK_DRIFT_STR},{_CLK_DRIFT_RATE_STR},\
{_ADOT_STR},{_CRS_STR},{_DELTAN_STR},{_M0_STR},\
{_CUC_STR},{_E_STR},{_CUS_STR},{_SQRTA_STR},\
{_TOE_BDT_TOW_STR},{_CIC_STR},{_OMEGA0_STR},{_CIS_STR},\
{_I0_STR},{_CRC_STR},{_OMEGA_STR},{_OMEGA_DOT_STR},\
{_IDOT_STR},{_DELTAN_DOT_STR},{_SAT_TYPE_STR},{_T_OP_STR},\
{_SISAI_OE_STR},{_SISAI_OCB_STR},{_SISAI_OC1_STR},{_SISAI_OC2_STR},\
{_SISMAI_STR},{_HEALTH_STR},{_B2B_INTEGRITY_FLAGS_STR},{_TGD_B2BI},\
{_TX_TIME_TOW_STR},{_IODE_STR}"

    def get_type(self) -> EphType:
        return EphType.BDS_CNV2


@dataclass
class RawNavBlock(object):
    satellite: Satellite
    epoch: datetime.datetime
    lines: List[str]

    def __repr__(self):
        return '\n'.join(self.lines)

    def __lt__(self, other):

        if self.epoch != other.epoch:
            return self.epoch < other.epoch

        elif self.satellite.constellation != other.satellite.constellation:
            return self.satellite.constellation < other.satellite.constellation

        return self.satellite.prn < other.satellite.prn

    def to_rinex2(self) -> str:

        out = ""

        id = self.satellite.prn
        if id > 99:
            raise ValueError('Cannot write a Rinex 2 nav block for satellite with id > 99')

        epoch_line = f'{id:2d}'
        clock_str = self.lines[1][23:]
        out = out + f'{epoch_line} {self.epoch.strftime("%y %m %d %H %M %S.0")}{clock_str}\n'
        for brdc_line in self.lines[2:]:
            out = out + f'{brdc_line[1:]}\n'
        # print extra lines
        out = out + '    0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00\n'
        out = out + f'{self.lines[4][1:23]} 0.000000000000e+00'

        return out

    def to_csv(self) -> str:
        pass

    def to_nav_block(self) -> NavBlock:
        """
        Convert to Navigation Block
        """

        # Get the type of EPH block
        eph_type = _get_eph_type_from_rinex4_nav_block_header(self.lines[0])
        if eph_type == EphType.GPS_LNAV:
            return self._to_GPS_LNAV()
        elif eph_type == EphType.GPS_CNAV:
            return self._to_GPS_CNAV()
        elif eph_type == EphType.GPS_CNV2:
            return self._to_GPS_CNV2()
        elif eph_type == EphType.GAL_FNAV or eph_type == EphType.GAL_INAV:
            return self._to_GAL()
        elif eph_type == EphType.BDS_D1 or eph_type == EphType.BDS_D2:
            return self._to_BDS_D(eph_type)
        elif eph_type == EphType.BDS_CNV1:
            return self._to_BDS_CNV1()
        elif eph_type == EphType.BDS_CNV2:
            return self._to_BDS_CNV2()
        elif eph_type == EphType.BDS_CNV3:
            return self._to_BDS_CNV3()

        raise ValueError(f'There is no NavBlock generator for [ {eph_type} ]')

    def _to_GPS_LNAV(self) -> GpsLnavNavBlock:

        sat_str, year, month, day, hour,  min, sec, clk_bias, clk_drift, clk_drift_rate = _parse_nav_epoch_line(self.lines[1])
        iode, crs, deltan, M0 = _parse_nav_orb_line(self.lines[2])
        cuc, e, cus, sqrtA = _parse_nav_orb_line(self.lines[3])
        toe_sow, cic, OMEGA0, cis = _parse_nav_orb_line(self.lines[4])
        i0, crc, omega, OMEGA_DOT = _parse_nav_orb_line(self.lines[5])
        idot, codesL2, toe_week, l2p_flag = _parse_nav_orb_line(self.lines[6])
        accuracy, health, tgd, iodc = _parse_nav_orb_line(self.lines[7])
        tx_time_tow, fit_interval, _, _ = _parse_nav_orb_line(self.lines[8])

        if fit_interval is None:
            fit_interval = 0

        return GpsLnavNavBlock(Satellite.from_string(sat_str),
                               datetime.datetime(year, month, day, hour, min, sec),
                               clk_bias, clk_drift, clk_drift_rate,
                               iode, crs, deltan, M0,
                               cuc, e, cus, sqrtA,
                               toe_sow, cic, OMEGA0, cis,
                               i0, crc, omega, OMEGA_DOT,
                               idot, int(codesL2), int(toe_week), int(l2p_flag),
                               accuracy, int(health), tgd, int(iodc),
                               tx_time_tow, int(fit_interval))

    def _to_GPS_CNAV(self) -> GpsCnavNavBlock:

        sat_str, year, month, day, hour,  min, sec, clk_bias, clk_drift, clk_drift_rate = _parse_nav_epoch_line(self.lines[1])
        adot, crs, deltan, M0 = _parse_nav_orb_line(self.lines[2])
        cuc, e, cus, sqrtA = _parse_nav_orb_line(self.lines[3])
        top, cic, OMEGA0, cis = _parse_nav_orb_line(self.lines[4])
        i0, crc, omega, OMEGA_DOT = _parse_nav_orb_line(self.lines[5])
        idot, deltan_dot, urai_ned0, urai_ned1 = _parse_nav_orb_line(self.lines[6])
        urai_ed, health, tgd, urai_ned = _parse_nav_orb_line(self.lines[7])
        isc_l1ca, isc_l2c, isc_l5i5, isc_l5q5 = _parse_nav_orb_line(self.lines[8])
        tx_time_tow, wn_op, _, _ = _parse_nav_orb_line(self.lines[9])

        return GpsCnavNavBlock(Satellite.from_string(sat_str),
                               datetime.datetime(year, month, day, hour, min, sec),
                               clk_bias, clk_drift, clk_drift_rate,
                               adot, crs, deltan, M0,
                               cuc, e, cus, sqrtA,
                               top, cic, OMEGA0, cis,
                               i0, crc, omega, OMEGA_DOT,
                               idot, deltan_dot, urai_ned0, urai_ned1,
                               urai_ed, health, tgd, urai_ned,
                               isc_l1ca, isc_l2c, isc_l5i5, isc_l5q5,
                               tx_time_tow, int(wn_op))

    def _to_GPS_CNV2(self) -> GpsCnv2NavBlock:

        sat_str, year, month, day, hour,  min, sec, clk_bias, clk_drift, clk_drift_rate = _parse_nav_epoch_line(self.lines[1])
        adot, crs, deltan, M0 = _parse_nav_orb_line(self.lines[2])
        cuc, e, cus, sqrtA = _parse_nav_orb_line(self.lines[3])
        top, cic, OMEGA0, cis = _parse_nav_orb_line(self.lines[4])
        i0, crc, omega, OMEGA_DOT = _parse_nav_orb_line(self.lines[5])
        idot, deltan_dot, urai_ned0, urai_ned1 = _parse_nav_orb_line(self.lines[6])
        urai_ed, health, tgd, urai_ned = _parse_nav_orb_line(self.lines[7])
        isc_l1ca, isc_l2c, isc_l5i5, isc_l5q5 = _parse_nav_orb_line(self.lines[8])
        isc_l1cd, isc_l1cp, _, _ = _parse_nav_orb_line(self.lines[9])
        tx_time_tow, wn_op, _, _ = _parse_nav_orb_line(self.lines[10])

        return GpsCnv2NavBlock(Satellite.from_string(sat_str),
                               datetime.datetime(year, month, day, hour, min, sec),
                               clk_bias, clk_drift, clk_drift_rate,
                               adot, crs, deltan, M0,
                               cuc, e, cus, sqrtA,
                               top, cic, OMEGA0, cis,
                               i0, crc, omega, OMEGA_DOT,
                               idot, deltan_dot, urai_ned0, urai_ned1,
                               urai_ed, health, tgd, urai_ned,
                               isc_l1ca, isc_l2c, isc_l5i5, isc_l5q5,
                               isc_l1cd, isc_l1cp,
                               tx_time_tow, int(wn_op))

    def _to_GAL(self) -> GalNavBlock:

        sat_str, year, month, day, hour,  min, sec, clk_bias, clk_drift, clk_drift_rate = _parse_nav_epoch_line(self.lines[1])
        iodnav, crs, deltan, M0 = _parse_nav_orb_line(self.lines[2])
        cuc, e, cus, sqrtA = _parse_nav_orb_line(self.lines[3])
        toe_tow, cic, OMEGA0, cis = _parse_nav_orb_line(self.lines[4])
        i0, crc, omega, OMEGA_DOT = _parse_nav_orb_line(self.lines[5])
        idot, datasources, toe_week, _ = _parse_nav_orb_line(self.lines[6])
        sisa, health, bgd_e5a, bgd_e5b = _parse_nav_orb_line(self.lines[7])
        tx_tm, _, _, _ = _parse_nav_orb_line(self.lines[8])

        return GalNavBlock(Satellite.from_string(sat_str),
                           datetime.datetime(year, month, day, hour, min, sec),
                           clk_bias, clk_drift, clk_drift_rate,
                           int(iodnav), crs, deltan, M0,
                           cuc, e, cus, sqrtA,
                           toe_tow, cic, OMEGA0, cis,
                           i0, crc, omega, OMEGA_DOT,
                           idot, int(datasources), int(toe_week),
                           sisa, health, bgd_e5a, bgd_e5b,
                           tx_tm)

    def _to_BDS_D(self, eph_type: EphType) -> BdsDNavBlock:

        sat_str, year, month, day, hour,  min, sec, clk_bias, clk_drift, clk_drift_rate = _parse_nav_epoch_line(self.lines[1])
        aode, crs, deltan, M0 = _parse_nav_orb_line(self.lines[2])
        cuc, e, cus, sqrtA = _parse_nav_orb_line(self.lines[3])
        toe_tow, cic, OMEGA0, cis = _parse_nav_orb_line(self.lines[4])
        i0, crc, omega, OMEGA_DOT = _parse_nav_orb_line(self.lines[5])
        idot, _, toe_week, _ = _parse_nav_orb_line(self.lines[6])
        accuracy, satH1, tgd1, tgd2 = _parse_nav_orb_line(self.lines[7])
        tx_tm, aodc, _, _ = _parse_nav_orb_line(self.lines[8])

        return BdsDNavBlock(Satellite.from_string(sat_str),
                            datetime.datetime(year, month, day, hour, min, sec),
                            clk_bias, clk_drift, clk_drift_rate,
                            int(aode), crs, deltan, M0,
                            cuc, e, cus, sqrtA,
                            toe_tow, cic, OMEGA0, cis,
                            i0, crc, omega, OMEGA_DOT,
                            idot, int(toe_week),
                            accuracy, int(satH1), tgd1, tgd2,
                            tx_tm, aodc, eph_type.value)

    def _to_BDS_CNV1(self) -> BdsCnv1NavBlock:

        sat_str, year, month, day, hour,  min, sec, clk_bias, clk_drift, clk_drift_rate = _parse_nav_epoch_line(self.lines[1])
        adot, crs, deltan, M0 = _parse_nav_orb_line(self.lines[2])
        cuc, e, cus, sqrtA = _parse_nav_orb_line(self.lines[3])
        toe_tow, cic, OMEGA0, cis = _parse_nav_orb_line(self.lines[4])
        i0, crc, omega, OMEGA_DOT = _parse_nav_orb_line(self.lines[5])
        idot, deltan_dot, sattype, t_op = _parse_nav_orb_line(self.lines[6])
        sisai_oe, sisai_ocb, sisai_oc1, sisai_oc2 = _parse_nav_orb_line(self.lines[7])
        isc_b1cd, _, tgd_b1cp, tgd_b2ap = _parse_nav_orb_line(self.lines[8])
        sismai, health, b1c_integrity_flags, iodc = _parse_nav_orb_line(self.lines[9])
        tx_time_tow, _, _, iode = _parse_nav_orb_line(self.lines[10])

        return BdsCnv1NavBlock(Satellite.from_string(sat_str),
                               datetime.datetime(year, month, day, hour, min, sec),
                               clk_bias, clk_drift, clk_drift_rate,
                               adot, crs, deltan, M0,
                               cuc, e, cus, sqrtA,
                               toe_tow, cic, OMEGA0, cis,
                               i0, crc, omega, OMEGA_DOT,
                               idot, deltan_dot, int(sattype), t_op,
                               sisai_oe, sisai_ocb, sisai_oc1, sisai_oc2,
                               isc_b1cd, tgd_b1cp, tgd_b2ap,
                               sismai, int(health), int(b1c_integrity_flags), int(iodc),
                               tx_time_tow, int(iode))

    def _to_BDS_CNV2(self) -> BdsCnv2NavBlock:

        sat_str, year, month, day, hour,  min, sec, clk_bias, clk_drift, clk_drift_rate = _parse_nav_epoch_line(self.lines[1])
        adot, crs, deltan, M0 = _parse_nav_orb_line(self.lines[2])
        cuc, e, cus, sqrtA = _parse_nav_orb_line(self.lines[3])
        toe_tow, cic, OMEGA0, cis = _parse_nav_orb_line(self.lines[4])
        i0, crc, omega, OMEGA_DOT = _parse_nav_orb_line(self.lines[5])
        idot, deltan_dot, sattype, t_op = _parse_nav_orb_line(self.lines[6])
        sisai_oe, sisai_ocb, sisai_oc1, sisai_oc2 = _parse_nav_orb_line(self.lines[7])
        isc_b2ad, _, tgd_b1cp, tgd_b2ap = _parse_nav_orb_line(self.lines[8])
        sismai, health, b2a_integrity_flags, iodc = _parse_nav_orb_line(self.lines[9])
        tx_time_tow, _, _, iode = _parse_nav_orb_line(self.lines[10])

        return BdsCnv2NavBlock(Satellite.from_string(sat_str),
                               datetime.datetime(year, month, day, hour, min, sec),
                               clk_bias, clk_drift, clk_drift_rate,
                               adot, crs, deltan, M0,
                               cuc, e, cus, sqrtA,
                               toe_tow, cic, OMEGA0, cis,
                               i0, crc, omega, OMEGA_DOT,
                               idot, deltan_dot, int(sattype), t_op,
                               sisai_oe, sisai_ocb, sisai_oc1, sisai_oc2,
                               isc_b2ad, tgd_b1cp, tgd_b2ap,
                               sismai, int(health), int(b2a_integrity_flags), int(iodc),
                               tx_time_tow, int(iode))

    def _to_BDS_CNV3(self) -> BdsCnv3NavBlock:

        sat_str, year, month, day, hour,  min, sec, clk_bias, clk_drift, clk_drift_rate = _parse_nav_epoch_line(self.lines[1])
        adot, crs, deltan, M0 = _parse_nav_orb_line(self.lines[2])
        cuc, e, cus, sqrtA = _parse_nav_orb_line(self.lines[3])
        toe_tow, cic, OMEGA0, cis = _parse_nav_orb_line(self.lines[4])
        i0, crc, omega, OMEGA_DOT = _parse_nav_orb_line(self.lines[5])
        idot, deltan_dot, sattype, t_op = _parse_nav_orb_line(self.lines[6])
        sisai_oe, sisai_ocb, sisai_oc1, sisai_oc2 = _parse_nav_orb_line(self.lines[7])
        sismai, health, b2b_integrity_flags, tgd_b2ap = _parse_nav_orb_line(self.lines[8])
        tx_time_tow, _, _, iode = _parse_nav_orb_line(self.lines[9])

        return BdsCnv3NavBlock(Satellite.from_string(sat_str),
                               datetime.datetime(year, month, day, hour, min, sec),
                               clk_bias, clk_drift, clk_drift_rate,
                               adot, crs, deltan, M0,
                               cuc, e, cus, sqrtA,
                               toe_tow, cic, OMEGA0, cis,
                               i0, crc, omega, OMEGA_DOT,
                               idot, deltan_dot, int(sattype), t_op,
                               sisai_oe, sisai_ocb, sisai_oc1, sisai_oc2,
                               sismai, int(health), int(b2b_integrity_flags), tgd_b2ap,
                               tx_time_tow)


class Nav(object):
    """
    Class that holds Rinex Navigation data
    """

    RINEX4_EPH_BLOCK_LINES_LEO = 8

    RINEX4_EPH_BLOCK_LINES = {
        EphType.GPS_LNAV: 8,
        EphType.GPS_CNAV: 9,
        EphType.GPS_CNV2: 10,
        EphType.GAL_INAV: 8,
        EphType.GAL_FNAV: 8,
        EphType.GLO_FDMA: 5,
        EphType.QZS_LNAV: 8,
        EphType.QZS_CNAV: 9,
        EphType.QZS_CNV2: 10,
        EphType.BDS_D1: 8,
        EphType.BDS_D2: 8,
        EphType.BDS_CNV1: 10,
        EphType.BDS_CNV2: 10,
        EphType.BDS_CNV3: 9,
        EphType.SBS: 4,
        EphType.IRN_LNAV: 8,
        EphType.LEO: RINEX4_EPH_BLOCK_LINES_LEO,
        EphType.SPIRE: RINEX4_EPH_BLOCK_LINES_LEO,
        EphType.STARLINK: RINEX4_EPH_BLOCK_LINES_LEO,
        EphType.ONEWEB: RINEX4_EPH_BLOCK_LINES_LEO,
    }

    def __init__(self, file: Union[str, IO]):
        self.blocks = Nav._load(file)

    @staticmethod
    @process_filename_or_file_handler('r')
    def _load(fh: IO) -> List['Nav']:
        """
        Load from a stream of data
        """

        blocks = []

        # header
        line = fh.readline()

        if not line.startswith('     4'):
            raise ValueError(f'Unsupported RINEX Nav version [ {line[5:6]} ] ')

        # body
        while True:
            line = fh.readline().rstrip()
            if line is None or len(line) == 0:
                break

            if line.startswith('> EOP'):
                skip_lines(fh, 3)
                continue

            elif line.startswith('> STO'):
                skip_lines(fh, 2)
                continue

            elif line.startswith('> ION') and line.endswith('> LNAV'):
                skip_lines(fh, 3)
                continue

            elif line.startswith('> ION') and line.endswith('> D1D2'):
                skip_lines(fh, 3)
                continue

            elif line.startswith('> ION') and line.endswith('> CNVX'):
                skip_lines(fh, 3)
                continue

            elif line.startswith('> ION') and line.endswith('> IFNV'):
                skip_lines(fh, 2)
                continue

            if not line.startswith('> EPH'):
                continue

            fields = line.split()
            sat = fields[2]
            satellite = Satellite(sat[0], int(sat[1:]))  # satellite from eph

            eph_type = _get_eph_type_from_rinex4_nav_block_header(line)
            n_lines = Nav.RINEX4_EPH_BLOCK_LINES[eph_type]

            lines = [fh.readline().rstrip() for _ in range(n_lines)]

            epoch_line = lines[0]
            epoch = datetime.datetime.strptime(epoch_line[4:23], "%Y %m %d %H %M %S")

            block_lines = [line] + lines
            block = RawNavBlock(satellite, epoch, block_lines)

            blocks.append(block)

        return blocks

    def get(satellite: Satellite):
        pass

    def to_blocks(self) -> Dict[EphType, List[NavBlock]]:

        out = {}

        for block in self.blocks:

            try:
                nav_block = block.to_nav_block()
            except ValueError:
                pass

            eph_type = nav_block.get_type()

            if eph_type not in out:
                out[eph_type] = []

            out[eph_type].append(nav_block)

        return out

    def to_dataframes(self) -> Dict[EphType, pd.DataFrame]:

        blocks = self.to_blocks()

        dfs = {}

        for eph_type, nav_blocks in blocks.items():
            data = [block.to_dict() for block in nav_blocks]
            dfs[eph_type] = pd.DataFrame(data)

        return dfs

    def __len__(self):
        """
        Number of blocks of the Rinex Nav file
        """
        return len(self.blocks)

    def __iter__(self):
        """
        Iterator for the Rinex blocks, to be used in for loops
        """
        return iter(self.blocks)

    def __lt__(self, other):
        return self.blocks < other.blocks

    @staticmethod
    def create_header(pgm: str = "roktools") -> str:
        """
        Create a basic RINEX 4.99 header

        Subversion 99 stands for Rokubun implementation of Rinex 4 standard
        that supports LEO navigation blocks
        """

        epoch_str = datetime.datetime.utcnow().strftime('%Y%m%d %H%M%S UTC ')

        out = "     4.99           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n"
        out = out + f"{pgm.ljust(20)}rokubun             {epoch_str}PGM / RUN BY / DATE\n"
        out = out + "    18                                                      LEAP SECONDS\n"
        out = out + "                                                            END OF HEADER\n"
        return out

    @staticmethod
    def create_navblock(satellite: Satellite, orbit: Kepler, sat_clock: Clock, code_biases: CodeBiases) -> RawNavBlock:

        WRITERS = {
            ConstellationId.STARLINK: Nav.create_leo_navblock,
            ConstellationId.SPIRE: Nav.create_leo_navblock,
            ConstellationId.LEO: Nav.create_leo_navblock,
            ConstellationId.ONEWEB: Nav.create_leo_navblock
        }

        writer = WRITERS.get(satellite.constellation, None)
        if writer:
            return writer(satellite, orbit, sat_clock, code_biases)

    @staticmethod
    def create_leo_navblock(satellite: Satellite, orbit: Kepler, sat_clock: Clock, biases: CodeBiases) -> RawNavBlock:
        """
        Output a RINEX4 navigation block from a set of orbit and clock parameters
        """

        lines = []

        # Header line
        lines.append(f'> EPH {satellite.constellation.to_char()}{satellite.prn:05d}')

        # EPOCH - CLK epoch
        lines.append(orbit.toe.strftime('    %Y %m %d %H %M %S') +
                     f'{sat_clock.bias_s:19.12e}{sat_clock.drift_s_per_s:19.12e}{sat_clock.drift_rate_s_per_s2:19.12e}')

        # ORBIT 1
        adot = 0.0
        crs = 0.0
        delta_n0 = 0.0
        M0 = orbit.true_anomaly_rad
        lines.append(Nav._write_orbit_line(adot, crs, delta_n0, M0))

        # ORBIT 2
        cuc = 0.0
        e = orbit.eccentricity
        cus = 0.0
        sqrta = math.sqrt(orbit.a_m)
        lines.append(Nav._write_orbit_line(cuc, e, cus, sqrta))

        # ORBIT - 3
        weektow = time.to_week_tow(orbit.toe)
        void = 0
        cic = 0.0
        # The RAAN parameter of GPS orbits are referred to the start of the GPS week, not the epoch
        # epoch of the ephemeris. Therfore, we need to compensate it
        t_start_of_week = time.from_week_tow(weektow.week, 0.0)
        greenwich_raan_rad = compute_greenwich_ascending_node_rad(t_start_of_week)
        OMEGA0 = orbit.raan_rad - greenwich_raan_rad
        OMEGA0 = math.fmod(OMEGA0 + math.tau, math.tau)

        cis = 0.0
        lines.append(Nav._write_orbit_line(weektow.tow, cic, OMEGA0, cis))

        # ORBIT - 4
        i0 = orbit.inclination_rad
        crc = 0.0
        omega = orbit.arg_perigee_rad
        OMEGA_DOT = 0.0
        lines.append(Nav._write_orbit_line(i0, crc, omega, OMEGA_DOT))

        # ORBIT - 5
        void = 0.0
        idot = 0.0
        delta_n_dot = orbit.delta_n_dot_rad_per_s
        lines.append(Nav._write_orbit_line(idot, delta_n_dot, weektow.week, void))

        # ORBIT - 6
        lines.append(Nav._write_orbit_line(void, void, biases.get_base_tgd(), void))

        # ORBIT - 7
        channel = TrackingChannel.from_string('9C')  # May depend on the LEO constellation
        lines.append(Nav._write_orbit_line(biases.get_code_bias(channel), void, void, void))

        return RawNavBlock(satellite, orbit.toe, lines)

    @staticmethod
    @process_filename_or_file_handler('w')
    def write_from_tle(output, tle_list: List[TLE], rinex2=False, sat_clock_model=ZeroClockModel(), code_biases=ZeroCodeBiases()) -> None:

        output.write(Nav.create_header(pgm='rinex_from_file'))

        for tle in tle_list:

            try:
                satellite = tle.get_satellite()
                orbit = tle.to_kepler()
                sat_clock = sat_clock_model.get_clock(satellite, tle.toe)
                nav_block = Nav.create_navblock(satellite, orbit, sat_clock, code_biases)

                output.write(nav_block.to_rinex2() if rinex2 else str(nav_block))
                output.write('\n')
            except ValueError as e:
                logger.warning(e)
                continue

    @staticmethod
    @process_filename_or_file_handler('w')
    def write_from_dataframe(output, df: pd.DataFrame, rinex2=False, sat_clock_model=ZeroClockModel(), clock_biases=ZeroCodeBiases()) -> None:
        f"""
        Write a RINEX file from the orbital parameters contained in a DataFrame,
        which should have the following fields
        - {EPOCH_STR}, with the reference epoch (will be parsed to a datetime)
        - {SAT_STR}, with the satellite identifier
        - {A_M_STR}, with the semimajor axis
        - {ECCENTRICITY_STR} Eccentricity of the orbit (adimensional)
        - {INCLINATION_DEG_STR} Inclination of the orbit (degrees)
        - {RIGHT_ASCENSION_DEG_STR} Right ascension of the ascending node (degrees)
        - {ARG_PERIGEE_DEG_STR} Argument of the perigee (degrees)
        - {TRUE_ANOMALY_DEG_STR} True anomaly (degrees)

        """

        output.write(Nav.create_header(pgm='rinex_from_file'))

        df = df.sort_values(by=['epoch', 'sat'], ascending=[True, True])

        for _, row in df.iterrows():

            try:
                constellation = ConstellationId.from_string(row.sat[0])
                prn = int(row.sat[1:])
                satellite = Satellite(constellation=constellation, prn=prn)
                sat_clock = sat_clock_model.get_clock(satellite, row.epoch)
                orbit = Kepler(
                    row.epoch, row.a_m, row.eccentricity,
                    math.radians(row.inclination_deg), math.radians(row.raan_deg),
                    math.radians(row.arg_perigee_deg), math.radians(row.true_anomaly_deg))
                nav_block = Nav.create_navblock(satellite, orbit, sat_clock, clock_biases)

                output.write(nav_block.to_rinex2() if rinex2 else str(nav_block))
                output.write('\n')
            except Exception as e:
                logger.warning(e)
                continue

    def write(self, output_filename, rinex2=False) -> None:

        with open(output_filename, 'w') as fh:

            fh.write(Nav.create_header(pgm='rinex_from_file'))

            for block in self.blocks:
                try:
                    fh.write(block.to_rinex2() if rinex2 else str(block))
                    fh.write('\n')
                except Exception as e:
                    logger.warning(e)
                    continue

#    def to_csv(self, csv_filename:str) -> None:
#
#        with open(csv_filename, "w") as fh:
#
#            for block in self.blocks:
#                pass

    @staticmethod
    def _write_orbit_line(a: float, b: float, c: float, d: float) -> str:
        return f'    {a:19.12e}{b:19.12e}{c:19.12e}{d:19.12e}'


def compute_greenwich_ascending_node_rad(epoch_utc: datetime.datetime) -> float:
    """
    Compute the Ascending Node of the Greenwich meridian at the given UTC epoch

    >>> round(compute_greenwich_ascending_node_rad(datetime.datetime(2024, 2, 11)), 9)
    2.453307616
    """

    gmst_h = time.gmst(epoch_utc)

    gmst_rad = gmst_h * math.tau / 24.0
    gmst_rad = gmst_rad % math.tau

    return gmst_rad


def merge_nav(files: List[Union[str, IO]]) -> str:
    """
    Merge RINEX navigation files

    :param files: list of RINEX Nav files to merge

    :return: the merged RINEX NAV file as a string
    """

    # Read navigation blocks and place them into memory
    rinex_navs = [Nav(file) for file in files]

    sorted_blocks = sorted([block for rinex_nav in rinex_navs for block in rinex_nav])

    # Proceed to output all of them
    out = Nav.create_header(pgm="merge_rinex_nav")

    out += '\n'.join([str(block) for block in sorted_blocks])

    return out


def merge_nav_cli():

    parser = argparse.ArgumentParser(description="Tool to merge various RINEX 4 files",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # for verbatim

    parser.add_argument('files', metavar='FILE', type=str, nargs='+', help='input RINEX4 file(s)')

    args = parser.parse_args()

    merged_rinex_str = merge_nav(args.files)

    sys.stdout.write(merged_rinex_str)


def rinex_from_file():

    parser = argparse.ArgumentParser(description="Tool to convert an input file to RINEX navigation file",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # for verbatim

    # Define the mutually exclusive group
    input_options = parser.add_mutually_exclusive_group(required=True)

    input_options.add_argument('--celestrak_file', metavar='<filename>', type=str,
                               help='File from Celestrak with the TLE elements to convert to RINEX.' +
                                    'Based on https://arxiv.org/abs/2401.17767')

    input_options.add_argument('--csv', metavar='<filename>', type=str,
                               help='CSV file with the description of ')

    parser.add_argument('--clock-model', '-c', choices=[ZERO_STR, RANDOM_STR], required=False,
                        help="Choose the clock model to be applied to satellites where clock \
                            bias and drift has not been provided. Defaults to '{ZERO_STR}'")

    parser.add_argument('--rinex2', action='store_true',
                        help='Output the format in Rinex 2 GPS format. Will skip satellites \
                            with PRN larger than 99')

    # Parse the command-line arguments
    args = parser.parse_args()

    sat_clock_model = ZeroClockModel()
    if args.clock_model == RANDOM_STR:
        sat_clock_model = GnssRandomClock()

    code_biases = ZeroCodeBiases()
    if args.code_biases == RANDOM_STR:
        TGD_MAX_S = 10.0e-9
        tgd_s = np.random.uniform(low=-TGD_MAX_S, high=TGD_MAX_S)
        isc_s9c_s = np.random.uniform(low=-TGD_MAX_S, high=TGD_MAX_S)
        code_biases = LEOCodeBiases(tgd_s, isc_s9c_s)

    if args.celestrak_file:
        tle_list = read_celestrak(args.celestrak_file)
        Nav.write_from_tle(sys.stdout, tle_list, rinex2=args.rinex2, sat_clock_model=sat_clock_model, code_biases=code_biases)

    elif args.csv:
        df = pd.read_csv(args.csv, parse_dates=['epoch'])
        Nav.write_from_dataframe(sys.stdout, df, rinex2=args.rinex2, sat_clock_model=sat_clock_model, code_biases=code_biases)


_ACCURACY_STR = "accuracy"
_ADOT_STR = "Adot[m/s]"
_AODC_STR = "aodc"
_AODE_STR = "aode"
_B1C_INTEGRITY_FLAGS_STR = "b1c_integrity_flags"
_B2A_INTEGRITY_FLAGS_STR = "b2a_integrity_flags"
_B2B_INTEGRITY_FLAGS_STR = "b2b_integrity_flags"
_BGD_E5A_STR = "bgd_e5a_e1[s]"
_BGD_E5B_STR = "bgd_e5b_e1[s]"
_CIC_STR = "cic[rad]"
_CIS_STR = "cis[rad]"
_CLK_BIAS_STR = "clock_bias[s]"
_CLK_DRIFT_RATE_STR = "clock_drift_rate[s/s2]"
_CLK_DRIFT_STR = "clock_drift[s/s]"
_CODESL2_STR = "codesL2"
_CRC_STR = "crc[m]"
_CRS_STR = "crs[m]"
_CUC_STR = "cuc[rad]"
_CUS_STR = "cus[rad]"
_DATA_SOURCES_STR = "datasources"
_DELTAN_DOT_STR = "deltan_dot[r/s^2]"
_DELTAN_STR = "deltan[rad/s]"
_EPOCH_STR = "epoch"
_E_STR = "e"
_FIT_INTERVAL_STR = "fit_interval"
_HEALTH_STR = "health"
_I0_STR = "i0[rad]"
_IDOT_STR = "idot[rad/s]"
_IODC_STR = "iodc"
_IODE_STR = "iode"
_IODNAV_STR = "iodnav"
_ISC_B2AD_STR = "isc_b2ad[s]"
_ISC_B1CD_STR = "isc_b1cd[s]"
_ISC_L1CA_STR = "isc_L1CA[s]"
_ISC_L1CD_STR = "isc_L1CD[s]"
_ISC_L1CP_STR = "isc_L1CP[s]"
_ISC_L2C_STR = "isc_L2C[s]"
_ISC_L5I5_STR = "isc_L5I5[s]"
_ISC_L5Q5_STR = "isc_L5Q5[s]"
_L2PFLAG_STR = "l2p_flag"
_M0_STR = "M0[rad]"
_OMEGA0_STR = "OMEGA0[rad]"
_OMEGA_DOT_STR = "OMEGA_DOT[rad/s]"
_OMEGA_STR = "omega[rad]"
_SATH1_STR = "sat_H1"
_SAT_STR = "sat"
_SAT_TYPE_STR = "sat_type"
_SISAI_OC1_STR = "SISAI_oc1"
_SISAI_OC2_STR = "SISAI_oc2"
_SISAI_OCB_STR = "SISAI_ocb"
_SISAI_OE_STR = "SISAI_oe"
_SISA_STR = "sisa[m]"
_SISMAI_STR = "SISMAI"
_SQRTA_STR = "sqrtA[sqrt(m)]"
_TGD1_STR = "tgd1[s]"
_TGD2_STR = "tgd2[s]"
_TGD_B1CP = "tgd_b1cp[s]"
_TGD_B2AP = "tgd_b2ap[s]"
_TGD_B2BI = "tgd_b2bi[s]"
_TGD_STR = "tgd[s]"
_TOE_BDT_TOW_STR = "toe_bdt_tow[s]"
_TOE_BDT_WEEK_STR = "toe_bdt_week[week]"
_TOE_GAL_TOW_STR = "toe_gal_tow[s]"
_TOE_GAL_WEEK_STR = "toe_gal_week[week]"
_TOE_SOW_STR = "toe[s]"
_TOE_WEEK_STR = "toe[week]"
_TX_TIME_TOW_STR = "tx_time[s]"
_T_OP_STR = "t_op[s]"
_URAI_ED_STR = "urai_ed"
_URAI_NED0_STR = "urai_ned0"
_URAI_NED1_STR = "urai_ned1"
_URAI_NED2_STR = "urai_ned2"
_WN_OP_STR = "wn_op[week]"


def _parse_nav_epoch_line(line: str) -> tuple:
    """
    Parse a Rinex 4 Navigation line

    >>> _parse_nav_epoch_line("G01 2024 02 13 11 22 33 1.693181693554e-04 1.477928890381e-12 1.000000000000e+00")
    ('G01', 2024, 2, 13, 11, 22, 33, 0.0001693181693554, 1.477928890381e-12, 1.0)
    """

    return line[0:3], \
        int(line[4:8]), int(line[9:11]), int(line[12:14]),  \
        int(line[15:17]),  int(line[17:20]),  int(line[20:23]), \
        float(line[23:42]), float(line[42:61]), float(line[61:])


def _parse_nav_orb_line(line: str) -> tuple:
    """
    Parse a Rinex 4 Navigation line (broadcast orbit line)


    >>> _parse_nav_orb_line('     1.700595021248e-06')
    (1.700595021248e-06, None, None, None)

    >>> _parse_nav_orb_line('     1.700595021248e-06 1.270996686071e-02')
    (1.700595021248e-06, 0.01270996686071, None, None)

    >>> _parse_nav_orb_line('     1.700595021248e-06 1.270996686071e-02 1.259148120880e-05')
    (1.700595021248e-06, 0.01270996686071, 1.25914812088e-05, None)

    >>> _parse_nav_orb_line('     1.700595021248e-06 1.270996686071e-02 1.259148120880e-05 5.154011671066e+03')
    (1.700595021248e-06, 0.01270996686071, 1.25914812088e-05, 5154.011671066)
    """

    try:
        v1 = float(line[4:23])
    except ValueError:
        v1 = None

    try:
        v2 = float(line[23:42])
    except ValueError:
        v2 = None

    try:
        v3 = float(line[42:61])
    except ValueError:
        v3 = None

    try:
        v4 = float(line[61:])
    except ValueError:
        v4 = None

    return v1, v2, v3, v4


def _parse_obs_line(line: str, n_obs: int) -> Tuple[Satellite, List[ObservableValue]]:
    """

    >>> line = "C05  40058862.469 6 208597044.05206  40058858.572 7 161300483.44407  40058861.947 7 169502210.29507"
    >>> _parse_obs_line(line, 6)
    (C05, [ObservableValue(value=40058862.469, lli=0, snr=6), \
ObservableValue(value=208597044.052, lli=0, snr=6), \
ObservableValue(value=40058858.572, lli=0, snr=7), \
ObservableValue(value=161300483.444, lli=0, snr=7), \
ObservableValue(value=40058861.947, lli=0, snr=7), \
ObservableValue(value=169502210.295, lli=0, snr=7)])
    """

    satellite = Satellite.from_string(line[0:3])

    observable_values = []

    offset = 3
    for i_obs in range(n_obs):
        start = offset + i_obs * 16
        obs_str = line[start:start + 14]
        lli_str = line[start + 14:start + 14 + 1]
        snr_str = line[start + 15:start + 15 + 1]

        obs = float(obs_str) if obs_str and obs_str != '              ' and obs_str != '\n' else math.nan
        lli = int(lli_str) if lli_str and lli_str != ' ' and lli_str != '\n' else 0
        snr = int(snr_str) if snr_str and snr_str != ' ' and snr_str != '\n' else 0

        observable_values.append(ObservableValue(obs, lli, snr))

    return satellite, observable_values


def _parse_rnx3_epoch(line):
    """
    Parse a measurement epoch from a Rinex3 and return a tuple
    with the epochm event type and number of lines

    >>> _parse_rnx3_epoch("> 2017 08 03 11 22 30.1234000  0 29")
    (datetime.datetime(2017, 8, 3, 11, 22, 30, 123400), 0, 29)

    >>> _parse_rnx3_epoch("> 2021  2  5 15 51 30.2000000 0 22")
    (datetime.datetime(2021, 2, 5, 15, 51, 30, 200000), 0, 22)

    >>> _parse_rnx3_epoch("> 2020 11 18 21 43 30.0000000  0 28       0.000000000000")
    (datetime.datetime(2020, 11, 18, 21, 43, 30), 0, 28)

    >>> _parse_rnx3_epoch("> 2019 07 02 13 25  5.9999995  0 31")
    (datetime.datetime(2019, 7, 2, 13, 25, 5, 999999), 0, 31)
    """

    try:
        _, year, month, day, hour, minute, seconds, epoch_flag, n_lines, *b = line.split()
    except ValueError as e:
        raise ValueError(f"Invalid Rinex 3 epoch line [ {line} ]: {e}")

    seconds, microseconds, *b = seconds.split('.')

    t = datetime.datetime(int(year), int(month), int(day),
                          hour=int(hour), minute=int(minute), second=int(seconds),
                          microsecond=int(microseconds[0:6]))

    return t, int(epoch_flag), int(n_lines)


def _get_eph_type_from_rinex4_nav_block_header(block_header: str) -> EphType:
    """
    Get the type of ephemeris based on the header line of the Rinex 4 navigation block

    >>> _get_eph_type_from_rinex4_nav_block_header('> EPH G01 LNAV')
    <EphType.GPS_LNAV: 'G_LNAV'>

    >>> _get_eph_type_from_rinex4_nav_block_header('> EPH X02434')
    <EphType.STARLINK: 'X'>
    """

    fields = block_header.split()

    sat = Satellite.from_string(fields[2])

    sat.constellation.value
    key = sat.constellation.value
    if len(fields) == 4:
        key = f'{key}_{fields[-1]}'

    return EphType.from_string(key)
