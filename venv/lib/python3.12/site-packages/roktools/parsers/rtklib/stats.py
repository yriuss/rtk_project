import datetime
import numpy as np
import pandas as pd

from roktools import logger
from roktools.file import grep_lines
from roktools.time import from_week_tow
from roktools.gnss.residuals import Residuals
from roktools.gnss.types import TrackingChannel, INVALID_TRACKING_CHANNEL


def parse(filename: str) -> Residuals:
    """
    Loads $SAT lines of a.stat file from a rtklib solution
    The format of those files are the following:
    Residuals of pseudorange and carrier-phase observables. The format
    of a record is as follows.
    $SAT,week,tow,sat,frq,az,el,resp,resc,vsat,snr,fix,slip,lock,outc,slipc,rejc,icbias,bias,bias_var,lambda
       - week/tow : gps week no/time of week (s)
       - sat/frq : satellite id/frequency (1:L1,2:L2,3:L5,...)
       - az/el : azimuth/elevation angle (deg)
       - resp : pseudorange residual (m)
       - resc : carrier-phase residual (m)
       - vsat : valid data flag (0:invalid,1:valid)
       - snr : signal strength (dbHz)
       - fix : ambiguity flag (0:no data,1:not part of AR set,2:part of AR set,3:part of hold set)
       - slip : cycle-slip flag (bit1:slip,bit2:parity unknown)
       - lock : carrier-lock count
       - outc : data outage count
       - slipc : cycle-slip count
       - rejc : data reject (outlier) count
       - icbias : interchannel bias (GLONASS)
       - bias : phase bias
       - bias_var : variance of phase bias
       - lambda : wavelength


    Example:
    $SAT,2215,152783.000,G03,1,82.8,35.9,16.3975,0.0018,1,36,0,0,0,0,0,0,-117643836.98,182.318392,0.00000

    Args:
        stat_file (str): .stat file path

    Returns:
        np.array: A numpy array with the content of the file
    """

    DTYPE = [
        ('week', 'i8'),
        ('tow', 'f8'),
        ('sat', 'S3'),
        ('freq', 'i8'),
        ('az', 'f8'),
        ('el', 'f8'),
        ('res_code_m', 'f8'),
        ('res_phase_m', 'f8'),
        ('vsat', 'i8'),
        ('snr_dbHz', 'f8'),
        ('fix', 'f8'),
        ('slip', 'f8'),
        ('lock', 'f8'),
        ('outc', 'f8'),
        ('slipc', 'f8'),
        ('rejc', 'f8'),
        # ('icbias', 'f8'),
        # ('bias', 'f8'),
        # ('bias_var', 'f8'),
        # ('lambda', 'f8'),
    ]

    USECOLS = list(range(1, len(DTYPE) + 1))

    generator = grep_lines(filename, "$SAT")

    data = np.loadtxt(generator, delimiter=',', usecols=USECOLS, dtype=DTYPE)

    data = __add_field_in_numpy_array(data, [('epoch', datetime.datetime)])
    data['epoch'] = [from_week_tow(int(row['week']), float(row['tow'])) for row in data]

    data = __add_field_in_numpy_array(data, [('processing_direction', 'S8')])
    data_length = len(data)
    data['processing_direction'] = ['forward'] * data_length
    if (np.unique(data['epoch']).size > 1) and (data['epoch'][0] == data['epoch'][-1]):
        data['processing_direction'][int(data_length / 2):] = 'backward'
    df = pd.DataFrame(data)
    df['sat'] = df['sat'].str.decode('utf-8')
    df['processing_direction'] = df['processing_direction'].str.decode('utf-8')
    df['constellation'] = df['sat'].str[0]
    df['channel'] = [__compute_channel(constellation, frequency)
                     for constellation, frequency in zip(df['constellation'], df['freq'])]
    df['signal'] = [str(sat) + str(channel) for sat, channel in zip(df['sat'], df['channel'])]

    return Residuals(df)


def __compute_channel(constellation: str, frequency: int) -> TrackingChannel:

    CHANNEL_1C = TrackingChannel(1, 'C')
    CHANNEL_2C = TrackingChannel(2, 'C')
    CHANNEL_5Q = TrackingChannel(5, 'Q')

    conversion_rule = {
        'G': {
            1: CHANNEL_1C,
            2: CHANNEL_2C,
            3: CHANNEL_5Q,
        },
        'E': {
            1: CHANNEL_1C,
            2: TrackingChannel(7, 'Q'),
            3: CHANNEL_5Q,
        },
        'C': {
            1: TrackingChannel(2, 'I'),
            2: TrackingChannel(6, 'I'),
            3: TrackingChannel(7, 'I'),
        },
        'R': {
            1: CHANNEL_1C,
            2: CHANNEL_2C,
            3: INVALID_TRACKING_CHANNEL,
        },
        'J': {
            1: CHANNEL_1C,
            2: CHANNEL_2C,
            3: CHANNEL_5Q,
        }
    }

    try:
        return conversion_rule[constellation][frequency]
    except Exception:
        logger.debug(f"Unknown channel for {constellation} with frequency {frequency}")
        return INVALID_TRACKING_CHANNEL


def __add_field_in_numpy_array(a: np.array, descr) -> np.array:
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], dtype=[('id', '<i8'), ('name', '|S3')])
    >>> sa.dtype.descr
    [('id', '<i8'), ('name', '|S3')]

    >>> sb = __add_field_in_numpy_array(sa, [('score', '<f8')])
    >>> sb.dtype.descr
    [('id', '<i8'), ('name', '|S3'), ('score', '<f8')]

    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError("'A' must be a structured numpy array")
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b
