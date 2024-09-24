from collections import namedtuple
import datetime
import math
import enum
from typing import List, Tuple
import numpy as np
import pandas as pd

GPS_TIME_START = datetime.datetime(1980, 1, 6, 0, 0, 0)
J2000_TIME_START = datetime.datetime(2000, 1, 1, 12, 0, 0)
SECONDS_IN_DAY = 24 * 60 * 60
SECONDS_IN_WEEK = 86400 * 7
GPS_AS_J2000 = -630763200

WeekTow = namedtuple('WeekTow', 'week tow day_of_week')


class TimeScale(enum.Enum):
    GPS = enum.auto()
    UTC = enum.auto()


def get_gps_leapseconds(utc_date: datetime.datetime) -> datetime.timedelta:

    if utc_date >= datetime.datetime(2017, 1, 1):
        return datetime.timedelta(seconds=18)
    elif utc_date >= datetime.datetime(2015, 7, 1):
        return datetime.timedelta(seconds=17)
    elif utc_date >= datetime.datetime(2012, 7, 1):
        return datetime.timedelta(seconds=16)
    elif utc_date >= datetime.datetime(2009, 1, 1):
        return datetime.timedelta(seconds=15)
    elif utc_date >= datetime.datetime(2006, 1, 1):
        return datetime.timedelta(seconds=14)
    elif utc_date >= datetime.datetime(1999, 1, 1):
        return datetime.timedelta(seconds=13)

    raise ValueError('No Leap second information for epochs prior to 1999-01-01')


class Timespan:
    def __init__(self, start: datetime.datetime, end: datetime.datetime):
        self.start = start
        self.end = end

    def __str__(self):
        return f"{self.start} - {self.end}"

    def is_overlaping(self, other: 'Timespan') -> bool:
        return (self.start <= other.end) and (self.end >= other.start)

    def duration(self) -> datetime.timedelta:
        return self.end - self.start

    def duration_seconds(self) -> int:
        return int((self.duration()).total_seconds())

    def duration_minutes(self) -> float:
        return self.duration_seconds() / 60

    def duration_hours(self) -> float:
        return self.duration_minutes() / 60

    def duration_days(self) -> float:
        return self.duration_hours() / 24

    def overlap(self, other: 'Timespan') -> 'Timespan':
        if not self.is_overlaping(other):
            raise ValueError('Timespans are not overlaped')

        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return Timespan(start, end)

    def as_tuple(self) -> tuple:
        return (self.start, self.end)

    def __repr__(self) -> str:
        return self.__str__


def to_week_tow(epoch: datetime.datetime, timescale: TimeScale = TimeScale.GPS) -> WeekTow:
    """
    Convert from datetime to GPS week (asumes datetime in GPS Timescale)

    >>> to_week_tow(datetime.datetime(1980, 1, 6))
    WeekTow(week=0, tow=0.0, day_of_week=0)
    >>> to_week_tow(datetime.datetime(2005, 1, 28, 13, 30))
    WeekTow(week=1307, tow=480600.0, day_of_week=5)

    Conversion method based on algorithm provided in this link
    http://www.novatel.com/support/knowledge-and-learning/published-papers-and-documents/unit-conversions/
    """

    timedelta = epoch - GPS_TIME_START
    leap_delta = get_gps_leapseconds(epoch) if timescale == TimeScale.UTC else datetime.timedelta(0)
    gpsw = int(timedelta.days / 7)
    day = timedelta.days - 7 * gpsw
    tow = timedelta.microseconds * 1e-6 + timedelta.seconds + day * SECONDS_IN_DAY + leap_delta.total_seconds()

    return WeekTow(gpsw, tow, day)


def from_week_tow(week: int, tow: float, timescale: TimeScale = TimeScale.GPS) -> datetime.datetime:
    """
    Convert from week tow to datetime in GPS scale

    >>> from_week_tow(0, 0.0)
    datetime.datetime(1980, 1, 6, 0, 0)
    >>> from_week_tow(1307, 480600.0)
    datetime.datetime(2005, 1, 28, 13, 30)
    """

    delta = datetime.timedelta(weeks=week, seconds=tow)

    gps_epoch = GPS_TIME_START + delta

    leap_delta = get_gps_leapseconds(gps_epoch) if timescale == TimeScale.UTC else datetime.timedelta(0)

    return gps_epoch - leap_delta


def weektow_to_datetime(tow: float, week: int) -> datetime.datetime:
    import warnings
    warnings.warn("This function will be replaced by 'from_week_tow'", DeprecationWarning, stacklevel=2)
    return from_week_tow(week, tow)


def weektow_to_j2000(tow: float, week: int) -> float:
    """
    Convert from GPS week and time of the week (in seconds) to j2000 seconds

    The week and tow values can be vectors, and thus it will return a vector of
    tuples.

    >>> weektow_to_j2000(0, 0.0)
    -630763200.0
    """

    j2000s = week * SECONDS_IN_WEEK
    j2000s += tow

    # Rebase seconds from GPS start origin to J2000 start origin
    j2000s += GPS_AS_J2000

    return j2000s


def to_j2000(epoch: datetime.datetime) -> float:
    """
    Convert from datetime toj2000 seconds

    >>> to_j2000(datetime.datetime(2005, 1, 28, 13, 30))
    160191000.0
    """
    week_tow = to_week_tow(epoch)
    return weektow_to_j2000(week_tow.tow, week_tow.week)


def from_j2000(j2000s: int, fraction_of_seconds: float = 0.0) -> datetime.datetime:
    """
    Convert from J2000 epoch to datetime

    >>> from_j2000(160191000)
    datetime.datetime(2005, 1, 28, 13, 30)

    >>> from_j2000(160191000, fraction_of_seconds = 0.1)
    datetime.datetime(2005, 1, 28, 13, 30, 0, 100000)
    """

    microseconds = int(fraction_of_seconds * 1.0e6)
    epoch = J2000_TIME_START + datetime.timedelta(seconds=j2000s, microseconds=microseconds)
    return epoch


def epoch_range(start_epoch, end_epoch, interval_s):
    """
    Iterate between 2 epochs with a given interval

    >>> import datetime
    >>> st = datetime.datetime(2015, 10, 1,  0,  0,  0)
    >>> en = datetime.datetime(2015, 10, 1,  0, 59, 59)
    >>> interval_s = 15 * 60
    >>> ','.join([str(d) for d in epoch_range(st, en, interval_s)])
    '2015-10-01 00:00:00,2015-10-01 00:15:00,2015-10-01 00:30:00,2015-10-01 00:45:00'
    >>> st = datetime.datetime(2015, 10, 1,  0,  0,  0)
    >>> en = datetime.datetime(2015, 10, 1,  1,  0,  0)
    >>> interval_s = 15 * 60
    >>> ','.join([str(d) for d in epoch_range(st, en, interval_s)])
    '2015-10-01 00:00:00,2015-10-01 00:15:00,2015-10-01 00:30:00,2015-10-01 00:45:00,2015-10-01 01:00:00'
    """

    total_seconds = (end_epoch - start_epoch).total_seconds() + interval_s / 2.0
    n_intervals_as_float = total_seconds / interval_s
    n_intervals = int(n_intervals_as_float)
    if math.fabs(n_intervals - n_intervals_as_float) >= 0.5:
        n_intervals = n_intervals + 1

    for q in range(n_intervals):
        yield start_epoch + datetime.timedelta(seconds=interval_s * q)


def round_to_interval(epoch: datetime, interval: int) -> datetime:
    """
        >>> dt = datetime.datetime(2023, 4, 20, 10, 48, 52, 794000)
        >>> interval = 0.1
        >>> round_to_interval(dt, interval)
        datetime.datetime(2023, 4, 20, 10, 48, 52, 800000)

        >>> interval = 1.0
        >>> round_to_interval(dt, interval)
        datetime.datetime(2023, 4, 20, 10, 48, 53)

        >>> interval = 0.5
        >>> round_to_interval(dt, interval)
        datetime.datetime(2023, 4, 20, 10, 48, 53)

        >>> interval = 2.0
        >>> round_to_interval(dt, interval)
        datetime.datetime(2023, 4, 20, 10, 48, 52)

        >>> interval = 0.05
        >>> round_to_interval(dt, interval)
        datetime.datetime(2023, 4, 20, 10, 48, 52, 800000)

        >>> interval = 0.01
        >>> round_to_interval(dt, interval)
        datetime.datetime(2023, 4, 20, 10, 48, 52, 790000)
    """
    timestamp = epoch.timestamp()
    rounded_timestamp = round(timestamp / interval) * interval
    return datetime.datetime.fromtimestamp(rounded_timestamp)


def get_interval(epochs: List[datetime.datetime], target_intervals: Tuple[float] = (2.0, 1.0, 0.5, 0.1, 0.05, 0.01)) -> float:
    """
    Finds the closest possible interval from a predefined set of intervals to the computed inteval
    from the pvt.

    Args:
        epochs: List of datetimes
        interval: The target intervals for which the closest possible interval is to be found.

    Returns:
        The closest possible interval from the predefined set.

    >>> t0 = datetime.datetime(2023, 6, 1, 12, 0, 0)
    >>> epochs = [t0, datetime.datetime(2023, 6, 1, 12, 0, 2), datetime.datetime(2023, 6, 1, 12, 0, 3)]
    >>> get_interval(epochs)
    2.0

    >>> t2 = datetime.datetime(2023, 6, 1, 12, 0, 1)
    >>> epochs = [t0, t2, datetime.datetime(2023, 6, 1, 12, 0, 2)]
    >>> get_interval(epochs)
    1.0

    >>> t1 = datetime.datetime(2023, 6, 1, 12, 0, 0, 500000)
    >>> epochs = [t0, t1, t2]
    >>> get_interval(epochs)
    0.5

    >>> t1 = datetime.datetime(2023, 6, 1, 12, 0, 0, 100000)
    >>> t2 = datetime.datetime(2023, 6, 1, 12, 0, 0, 200000)
    >>> epochs = [t0, t1, t2]
    >>> get_interval(epochs)
    0.1

    >>> t1 = datetime.datetime(2023, 6, 1, 12, 0, 0, 10000)
    >>> t2 = datetime.datetime(2023, 6, 1, 12, 0, 0, 11000)
    >>> epochs = [t0, t1, t2]
    >>> get_interval(epochs)
    0.01

    """
    interval = np.median(np.ediff1d(epochs))
    differences = [abs(interval.total_seconds() - target_interval) for target_interval in target_intervals]
    return target_intervals[differences.index(min(differences))]


def to_julian_date(epoch: datetime.datetime) -> float:
    """
    Convert an epoch to Julian Date

    >>> to_julian_date(datetime.datetime(2024, 2, 11))
    2460351.5
    >>> round(to_julian_date(datetime.datetime(2019, 1, 1, 8)), 2)
    2458484.83
    """

    # Convert datetime object to Julian Date
    dt = epoch - datetime.datetime(2000, 1, 1, 12, 0, 0)
    julian_date = 2451545.0 + dt.total_seconds() / 86400.0

    return julian_date


def seconds_of_day(epoch: datetime.datetime) -> float:
    """
    Compute the seconds of the day

    >>> seconds_of_day(datetime.datetime(2024, 4, 1))
    0.0
    >>> seconds_of_day(datetime.datetime(2024, 4, 1, 23, 59, 59))
    86399.0
    """

    return epoch.hour * 3600 + epoch.minute * 60 + epoch.second + epoch.microsecond / 1.0e6


def gmst(epoch: datetime.datetime) -> float:
    """
    Compute the Greenwich Mean Sidereal Time (in hours)

    https://astronomy.stackexchange.com/questions/21002/how-to-find-greenwich-mean-sideral-time


    >>> round(gmst(datetime.datetime(2019, 1, 1, 8)), 6)
    14.712605
    >>> gmst_hour = gmst(datetime.datetime(2024, 2, 11))
    >>> round(gmst_hour * math.tau / 24, 9)
    2.453307616
    """

    julian_date = to_julian_date(epoch)

    midnight = math.floor(julian_date) + 0.5
    days_since_midnight = julian_date - midnight
    hours_since_midnight = days_since_midnight * 24.0
    days_since_epoch = julian_date - 2451545.0
    centuries_since_epoch = days_since_epoch / 36525
    whole_days_since_epoch = midnight - 2451545.0

    GMST_hours = 6.697374558 + 0.06570982441908 * whole_days_since_epoch \
        + 1.00273790935 * hours_since_midnight \
        + 0.000026 * centuries_since_epoch**2

    return GMST_hours % 24


def compute_elapsed_seconds(epochs: pd.Series) -> pd.Series:
    return (epochs - epochs.iloc[0]).dt.total_seconds()


def compute_decimal_hours(epochs: pd.Series) -> pd.Series:
    return epochs.apply(lambda x: x.hour + x.minute / 60 + x.second / 3600)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
