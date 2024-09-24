from dataclasses import dataclass
import datetime
import math

from ..constants import EARTH_GRAVITATION_PARAM_MU


@dataclass
class Kepler(object):
    """
    Data class to represent a satellite orbit in an inertial reference frame
    """

    """ Time of Ephemeris """
    toe: datetime.datetime

    """ Semi-major axis [m] """
    a_m: float

    """ Eccentricity """
    eccentricity: float

    """ Inclination [rad] """
    inclination_rad: float

    """ Right ascension of the ascending node [rad] """
    raan_rad: float

    """ Argument of the perigee [rad] """
    arg_perigee_rad: float

    """ True anomaly at the time of ephemeris [rad] """
    true_anomaly_rad: float

    delta_n_dot_rad_per_s: float = 0.0

    def __repr__(self) -> str:
        out = f"""
        toe: {self.toe}
        semi-major axis[deg]: {math.degrees(self.a_m)}
        eccentricity: {self.eccentricity}
        inclination[deg]: {math.degrees(self.inclination_rad)}
        RAAN[deg]: {math.degrees(self.raan_rad)}
        arg_perigee[deg]: {math.degrees(self.arg_perigee_rad)}
        mean anomaly[deg]: {math.degrees(self.true_anomaly_rad)}
        Delta N [deg/s]: {math.degrees(self.delta_n_dot_rad_per_s)}
        """
        return out


def compute_semi_major_axis(mean_motion_rad_per_s: float) -> float:
    """
    Compute the semi major axis (a) in meters from the mean motion

    Using the equation $$n = sqrt(mu) / (sqrt(A))^3$$

    >>> mean_motion_rev_per_day = 15.5918272
    >>> mean_motion_rad_per_s = mean_motion_rev_per_day * math.tau / 86400.0
    >>> compute_semi_major_axis(mean_motion_rad_per_s)
    6768158.4970976645
    """

    return math.pow(EARTH_GRAVITATION_PARAM_MU/math.pow(mean_motion_rad_per_s, 2.0), 1.0 / 3.0)
