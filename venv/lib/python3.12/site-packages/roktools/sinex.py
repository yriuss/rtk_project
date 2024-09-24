from dataclasses import dataclass
import datetime
import typing

TAG_SAT_PRN = "SATELLITE/PRN"


@dataclass
class _SatPrnItem():
    svid: str
    valid_from: datetime.datetime
    valid_to: datetime.datetime
    prn: str

    def in_period(self, epoch: datetime):
        return epoch >= self.valid_from and (self.valid_to is None or epoch <= self.valid_to)


class _SatPrn():

    def __init__(self, sat_prns: typing.List[_SatPrnItem]):
        self.items = {}

        for sat_prn in sat_prns:
            key = sat_prn.prn
            if key not in self.items:
                self.items[key] = []

            self.items[key].append(sat_prn)

    def to_svid(self, prn: str, epoch: datetime) -> str:
        """
        Get the Space Vehicle ID for a given Satellite PRN assignement at a
        specific epoch
        """

        items = self.items[prn]

        for item in items:
            if item.in_period(epoch):
                return item.svid

        raise ValueError(f'Could not find SVID for [ {prn} ] at [ {epoch} ]')


def to_sat_prn(sinex_filename: str) -> _SatPrn:

    with open(sinex_filename, 'r') as fh:
        lines = _extract_section(fh, TAG_SAT_PRN)
        sat_prns = []

        for line in lines:
            if line.startswith(('*', '+')):
                continue
            sat_prns.append(_parse_sat_prn_line(line))

    return _SatPrn(sat_prns)


def _parse_epoch(epoch_str: str) -> datetime.datetime:
    """
    Parse an epoch expressed in SINEX format %Y:%j:<seconds_of_day>

    >>> _parse_epoch("2024:029:43200")
    datetime.datetime(2024, 1, 29, 12, 0)

    >>> _parse_epoch("0000:000:00000")
    """

    NO_EPOCH = "0000:000:00000"
    if epoch_str == NO_EPOCH:
        return None

    fields = epoch_str.split(":")
    if len(fields) < 3:
        raise ValueError(f'Input [ {epoch_str} ] does not seem to conform to SINEX epoch format and cannot be parsed')

    epoch = datetime.datetime.strptime(f'{fields[0]}:{fields[1]}', "%Y:%j")
    seconds = float(fields[2])

    epoch = epoch + datetime.timedelta(seconds=seconds)

    return epoch


def _parse_sat_prn_line(line: str) -> _SatPrnItem:
    """
    Parse a SATELLITE/PRN line and extract its fields

    >>> _parse_sat_prn_line('G001 1978:053:00000 1985:199:00000 G04')
    _SatPrnItem(svid='G001', valid_from=datetime.datetime(1978, 2, 22, 0, 0), valid_to=datetime.datetime(1985, 7, 18, 0, 0), prn='G04')
    """

    fields = line.split(' ')
    if len(fields) < 4:
        raise ValueError(f'The input line [ {line} ] does not seem to be a SATELLITE/PRN line')

    svid, valid_from, valid_to, prn = fields[0:4]

    return _SatPrnItem(svid, _parse_epoch(valid_from), _parse_epoch(valid_to), prn)


def _extract_section(fh, tag: str) -> typing.List[str]:
    """
    Extract a SINEX section
    """

    lines = []
    in_block = False

    for line in fh:

        if f'+{tag}' in line:
            in_block = True
        elif f'-{tag}' in line:
            break

        if in_block:
            lines.append(line.strip())

    return lines
