#!/usr/bin/env python
"""
Module with several geodetic related methods
"""
from dataclasses import astuple, dataclass
import math
from typing import List, Union, Iterable

import numpy as np

import roktools.logger

# WGS84 constants
WGS84_A = 6378137.0
WGS84_E = 8.1819190842622e-2


@dataclass
class ElAz:
    elevation: float
    azimuth: float

    def __iter__(self):
        return iter(astuple(self))

    def __repr__(self):
        return f'{self.elevation:.3f} {self.azimuth:.3f}'


@dataclass
class DMS:
    degrees: int
    minutes: int
    seconds: float

    def __iter__(self):
        return iter(astuple(self))

    def __repr__(self):
        return f"{self.degrees}º {self.minutes}' {self.seconds:.3f}\""


@dataclass
class XYZ:
    x: float
    y: float
    z: float

    def is_valid(self):
        EPS_THRESHOLD = 1.0e-3
        return abs(self) > EPS_THRESHOLD

    def __iter__(self):
        return iter(astuple(self))

    def __repr__(self):
        return f'{self.x:.3f} {self.y:.3f} {self.z:.3f}'

    def __sub__(self, ref: 'XYZ') -> 'XYZ':
        return XYZ(self.x - ref.x, self.y - ref.y, self.z - ref.z)

    def __add__(self, ref: 'XYZ') -> 'XYZ':
        return XYZ(self.x + ref.x, self.y + ref.y, self.z + ref.z)

    def __abs__(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def __truediv__(self, scale: float) -> 'XYZ':
        return XYZ(self.x / scale, self.y / scale, self.z / scale)

    def __mul__(self, scale: float) -> 'XYZ':
        return XYZ(self.x * scale, self.y * scale, self.z * scale)


@dataclass
class LLA:
    longitude: float  # In degrees
    latitude: float  # In degrees
    altitude: float  # In meters

    def is_valid(self):
        ALT_THRESHOLD_M = 10.0e3  # meters above/under surface level
        return self.altitude < ALT_THRESHOLD_M and self.altitude > -ALT_THRESHOLD_M

    def __iter__(self):
        return iter(astuple(self))

    def __repr__(self):
        return f'{self.longitude:.3f} {self.latitude:.3f} {self.altitude:.3f}'


@dataclass
class LLA_DMS:
    longitude: DMS
    latitude: DMS
    altitude: float

    def is_valid(self):
        ALT_THRESHOLD_M = 10.0e3  # meters above/under surface level
        return self.altitude < ALT_THRESHOLD_M and self.altitude > -ALT_THRESHOLD_M

    def __iter__(self):
        return iter(astuple(self))

    def __repr__(self):
        return f'{self.longitude} {self.latitude} {self.altitude}'


@dataclass
class ENU:
    east: float
    north: float
    up: float

    def __iter__(self):
        return iter(astuple(self))

    def __repr__(self):
        return f'{self.east:.3f} {self.north:.3f} {self.up:.3f}'

    def get_horizontal_error(self):
        return get_horizontal_error(self.east, self.north)

    def get_vertical_error(self):
        return get_vertical_error(self.east, self.north, self.up)

    def __mul__(self, scale: float) -> 'ENU':
        return ENU(self.east * scale, self.north * scale, self.up * scale)


@dataclass
class AXU:
    along: float
    cross: float
    up: float

    def __iter__(self):
        return iter(astuple(self))

    def __repr__(self):
        return f'{self.along:.3f} {self.cross:.3f} {self.up:.3f}'


class Coordinates(object):
    """
    Class to store coordinates and transform among various coordinate systems
    (cartesian and geodetic)
    """

    def __init__(self, x: float, y: float, z: float) -> None:
        self.point = XYZ(x, y, z)

    def __eq__(self, other) -> bool:
        """
        >>> a = Coordinates(0, 0, 0)
        >>> a == a
        True

        >>> b = Coordinates(100, 0, 0)
        >>> a == b
        False
        """

        if isinstance(other, Coordinates):
            equal_x = math.isclose(self.point.x, other.point.x)
            equal_y = math.isclose(self.point.y, other.point.y)
            equal_z = math.isclose(self.point.z, other.point.z)
            return (equal_x and equal_y and equal_z)

        return False

    def is_valid(self) -> bool:
        """
        >>> coordinates = Coordinates(0, 0, 0)
        >>> coordinates.is_valid()
        False
        """
        return self.xyz().is_valid()

    def xyz(self) -> XYZ:
        """
        >>> coordinates = Coordinates(4889803.653, 170755.706, 4078049.851)
        >>> coordinates.xyz()
        4889803.653 170755.706 4078049.851
        """
        return self.point

    def lla(self, a: float = WGS84_A, e: float = WGS84_E) -> LLA:
        """
        >>> coordinates = Coordinates(4889803.653, 170755.706, 4078049.851)
        >>> "lon: {:.0f} lat: {:.0f} height: {:.0f}".format(*coordinates.lla())
        'lon: 2 lat: 40 height: 100'
        """
        return LLA(*xyz_to_lla(*self.point, a=a, e=e))

    def lla_dms(self, a: float = WGS84_A, e: float = WGS84_E) -> LLA_DMS:
        """
        >>> coordinates = Coordinates(4889803.653, 170755.706, 4078049.851)
        >>> coord = coordinates.lla_dms()
        >>> coord.longitude.degrees
        1
        >>> coord.longitude.minutes
        59
        """
        lla = self.lla(a=a, e=e)

        longitude = convert_decimal_units_to_dms(lla.longitude)
        latitude = convert_decimal_units_to_dms(lla.latitude)

        return LLA_DMS(longitude, latitude, lla.altitude)

    def apply_displacement(self, enu: ENU) -> 'Coordinates':
        """
        >>> coordinates = Coordinates(4787369.149, 183275.989, 4196362.621)
        >>> enu = ENU(0.096,0.010,0.144)
        >>> coordinates.apply_displacement(enu).xyz()
        4787369.247 183276.089 4196362.724
        """
        displacement = CoordinatesDisplacement(enu=enu)
        displacement.set_reference_position(self)
        xyz = displacement.get_xyz()
        return Coordinates(self.xyz().x + xyz.x, self.xyz().y + xyz.y, self.xyz().z + xyz.z)

    def __repr__(self) -> str:
        """
        >>> Coordinates(4889803.653, 170755.706, 4078049.851)
        4889803.653 170755.706 4078049.851
        """
        return f'{self.point.x:.3f} {self.point.y:.3f} {self.point.z:.3f}'

    def print_lla(self, a=WGS84_A, e=WGS84_E) -> str:
        """
        >>> coordinates = Coordinates(4889803.653, 170755.706, 4078049.851)
        >>> coordinates.print_lla()
        '40.0000000N 2.0000000E 100.000m'
        """
        lla = LLA(*xyz_to_lla(*self.point, a=a, e=e))
        lat_letter = 'N' if lla.latitude else 'S'
        latitude = abs(lla.latitude)
        lon_letter = 'E' if lla.longitude else 'W'
        longitude = abs(lla.longitude)
        return f"{latitude:.7f}{lat_letter} {longitude:.7f}{lon_letter} {lla.altitude:.3f}m"

    def __sub__(self: 'Coordinates', ref: 'Coordinates') -> 'CoordinatesDisplacement':
        """
        >>> dest_position = Coordinates(4889804, 170755, 4078049)
        >>> orig_position = Coordinates(4889803, 170757, 4078049)
        >>> (dest_position - orig_position).get_xyz()
        1.000 -2.000 0.000
        """

        xyz_self = self.xyz()
        xyz_ref = ref.xyz()

        xyz = XYZ(x=xyz_self.x - xyz_ref.x, y=xyz_self.y - xyz_ref.y, z=xyz_self.z - xyz_ref.z)
        return CoordinatesDisplacement(xyz=xyz, ref_position=ref)

    @staticmethod
    def from_lla(longitude_deg: float, latitude_deg: float, height_m: float,
                 a: float = WGS84_A, e: float = WGS84_E) -> 'Coordinates':
        """
        >>> coordinates = Coordinates.from_lla(2, 40, 100)
        >>> coordinates.is_valid()
        True
        """

        xyz = lla_to_xyz(longitude_deg, latitude_deg, height_m, a=a, e=e)

        return Coordinates(*xyz)

    @staticmethod
    def from_xyz(x: float, y: float, z: float) -> 'Coordinates':
        """
        >>> coordinates = Coordinates.from_lla(4889803.653, 170755.706, 4078049.851)
        >>> coordinates.is_valid()
        True
        """

        return Coordinates(x, y, z)


class CoordinatesDisplacement(object):
    """
    Representation of a displacement relative to a certain point (Coordinates).
    Use this class to represent deviations such as error vector, lever arms, ...
    """

    def __init__(self, xyz: XYZ = None, enu: ENU = None, ref_position: Coordinates = None):
        """
        >>> c = CoordinatesDisplacement(xyz=XYZ(0.1, 0.1, 0.1))
        >>> c.is_valid()
        True
        """
        self.xyz = xyz
        self.enu = enu
        self.ref_lla = None
        if ref_position:
            self.set_reference_position(ref_position)

    def __repr__(self) -> str:
        if self.xyz:
            return f'XYZ: {str(self.xyz)}'
        elif self.enu:
            return f'ENU: {str(self.enu)}'
        else:
            raise Exception('Format not defined')

    def set_reference_position(self, position: Coordinates) -> None:

        if not position:
            raise ValueError("Invalid reference position")

        self.ref_lla = position.lla()

        if not self.enu:
            self.enu = self.get_enu()

    def is_valid(self) -> bool:
        """
        >>> c = CoordinatesDisplacement()
        >>> c.is_valid()
        False
        """
        return self.xyz is not None or self.enu is not None

    def check_validity(self) -> None:
        if not self.is_valid():
            raise ValueError("CoordinatesDisplacement object is not valid")

    def get_xyz(self) -> XYZ:
        """
        >>> enu = ENU(-0.416, 5.664, -0.081)
        >>> reference_position = Coordinates.from_lla(1.52160871, 41.26531742, 127.0)
        >>> c = CoordinatesDisplacement(enu=enu)
        >>> c.set_reference_position(reference_position)
        >>> xyz = c.get_xyz()
        >>> '{:.03f} {:.03f} {:.03f}'.format(*xyz)
        '-3.784 -0.517 4.204'
        """

        self.check_validity()

        if not self.xyz:
            if not self.ref_lla:
                raise ValueError("Missing reference position in CoordinatesDisplacement to convert frame (ENU to XYZ)")
            xyz_list = enu_to_ecef(self.ref_lla.longitude, self.ref_lla.latitude, *self.enu)
            self.xyz = XYZ(*xyz_list)

        return self.xyz

    def get_enu(self) -> ENU:
        """
        >>> xyz = XYZ(0.1, 0.1, 0.1)
        >>> ref_position = Coordinates(4889803.653, 170755.706, 4078049.851)
        >>> c = CoordinatesDisplacement(xyz=xyz, ref_position=ref_position)
        >>> enu = c.get_enu()
        >>> '{:.03f} {:.03f} {:.03f}'.format(*enu)
        '0.096 0.010 0.144'
        """

        self.check_validity()

        if not self.enu:
            if not self.ref_lla:
                raise ValueError("Missing reference position in CoordinatesDisplacement to convert frame (XYZ to ENU)")
            enu_list = ecef_to_enu(self.ref_lla.longitude, self.ref_lla.latitude, *self.xyz)
            self.enu = ENU(*enu_list)

        return self.enu

    def get_absolute_position(self) -> Coordinates:
        """
        >>> enu = ENU(0.096,0.010,0.144)
        >>> ref_position = Coordinates(4889803.653, 170755.706, 4078049.851)
        >>> c = CoordinatesDisplacement(enu=enu, ref_position=ref_position)
        >>> position = c.get_absolute_position()
        >>> '{:.3f} {:.3f} {:.3f}'.format(*position.xyz())
        '4889803.753 170755.806 4078049.951'
        >>> c_displacement = CoordinatesDisplacement.from_positions(ref_position, position)
        >>> '{:.1f} {:.1f} {:.1f}'.format(*c_displacement.get_xyz())
        '0.1 0.1 0.1'
        """

        xyz = lla_to_xyz(self.ref_lla.longitude, self.ref_lla.latitude, self.ref_lla.altitude)

        dxyz = self.get_xyz()

        return Coordinates(*(xyz + dxyz))

    def get_axu(self, direction: ENU) -> AXU:
        """
        Convert to along-track, cross-track and up for a given direction

        >>> velocity = ENU(1.0, 1.0, 1)
        >>> error = ENU(1.0, 1.0, 0)
        >>> error = CoordinatesDisplacement(enu=error)
        >>> error.get_axu(velocity)
        1.414 0.000 0.000

        >>> error = ENU(1.0, 0.0, 0)
        >>> error = CoordinatesDisplacement(enu=error)
        >>> error.get_axu(velocity)
        0.707 -0.707 0.000

        >>> error = ENU(1.0, -1.0, 0)
        >>> error = CoordinatesDisplacement(enu=error)
        >>> error.get_axu(velocity)
        0.000 -1.414 0.000

        >>> error = ENU(-1.0, -1.0, 0)
        >>> error = CoordinatesDisplacement(enu=error)
        >>> error.get_axu(velocity)
        -1.414 0.000 0.000

        >>> error = ENU(0.0, -1.0, 0)
        >>> error = CoordinatesDisplacement(enu=error)
        >>> error.get_axu(velocity)
        -0.707 -0.707 0.000
        """

        position = self.get_enu()

        # Compute angle between the horizontal coordinates vector and the horizontal direction
        v_hor = [direction.east, direction.north]
        v_hor_norm = np.linalg.norm(v_hor)

        u_v = np.array(v_hor) / v_hor_norm

        p_hor = [position.east, position.north]
        p_hor_norm = np.linalg.norm(p_hor)

        theta_rad = math.acos(np.dot(v_hor, p_hor) / v_hor_norm / p_hor_norm)

        signum = +1 if np.cross(u_v, p_hor) >= 0 else -1
        cross_track = p_hor_norm * math.sin(theta_rad) * signum

        along_track = p_hor_norm * math.cos(theta_rad)

        return AXU(along_track, cross_track, position.up)

    def __abs__(self) -> float:
        """
        >>> c = CoordinatesDisplacement.from_xyz(0.1, 0.1, 0.1)
        >>> f'{abs(c):.4f}'
        '0.1732'
        """

        if not self.is_valid():
            raise ValueError("Cannot compute the magnitude of an empty array")

        vector = self.xyz or self.enu

        return math.sqrt(sum([v * v for v in vector]))

    @staticmethod
    def from_xyz(x: float, y: float, z: float):
        """
        >>> c = CoordinatesDisplacement.from_xyz(0.1, 0.1, 0.1)
        >>> c.is_valid()
        True
        """
        return CoordinatesDisplacement(xyz=XYZ(x, y, z))

    @staticmethod
    def from_positions(from_position: Coordinates, to_positions: Coordinates) -> 'CoordinatesDisplacement':
        """
        Compute the displacement vector between two positions

        >>> from_position = Coordinates(0, 0, 0.1)
        >>> to_position = Coordinates(0, 0, 0)
        >>> c = CoordinatesDisplacement.from_positions(from_position, to_position)
        >>> c.is_valid()
        True
        >>> abs(c)
        0.1
        """

        from_xyz = from_position.xyz()
        to_xyz = to_positions.xyz()
        dxyz = XYZ(to_xyz.x - from_xyz.x, to_xyz.y - from_xyz.y, to_xyz.z - from_xyz.z)

        return CoordinatesDisplacement(xyz=dxyz, ref_position=from_position)

    @staticmethod
    def from_enu(e: float, n: float, u: float):
        """
        >>> c = CoordinatesDisplacement.from_enu(0.1, 0.1, 0.1)
        >>> c.is_valid()
        True
        """
        return CoordinatesDisplacement(enu=ENU(e, n, u))


def get_horizontal_error(x: float, y: float) -> float:
    return math.sqrt(x**2 + y**2)


def get_vertical_error(x: float, y: float, z: float):
    return math.sqrt(x**2 + y**2 + z**2)


def lla_to_xyz(longitude_deg: Union[float, Iterable], latitude_deg: Union[float, Iterable],
               height_m: Union[float, Iterable], a: float = WGS84_A, e: float = WGS84_E) -> XYZ:
    """
    Convert from geodetic coordinates (relative to a reference ellipsoid which
    defaults to WGS84) to Cartesian XYZ-ECEF coordinates.

    The longitude, latitude and height values can be vectors

    >>> xyz = lla_to_xyz(9.323302567, 48.685064919, 373.2428)
    >>> [float('{0:.3f}'.format(v)) for v in xyz]
    [4163316.145, 683507.935, 4767789.479]

    >>> lons = np.array([9.323302567, 9.323335545, 9.323368065])
    >>> lats = np.array([48.685064919, 48.685050295, 48.685036011])
    >>> hgts = np.array([373.2428, 373.2277, 373.2078])
    >>> x, y, z = lla_to_xyz(lons, lats, hgts)
    >>> "{0:.6f}".format(x[0])
    '4163316.144693'
    >>> "{0:.6f}".format(x[1])
    '4163316.946837'
    >>> "{0:.6f}".format(x[2])
    '4163317.723291'
    >>> "{0:.6f}".format(y[0])
    '683507.934738'
    >>> "{0:.6f}".format(y[1])
    '683510.527317'
    >>> "{0:.6f}".format(y[2])
    '683513.081502'
    >>> "{0:.6f}".format(z[0])
    '4767789.478699'
    >>> "{0:.6f}".format(z[1])
    '4767788.393654'
    >>> "{0:.6f}".format(z[2])
    '4767787.329966'
    """

    # Convert from degrees to radians if necessary
    lon = np.deg2rad(longitude_deg)
    lat = np.deg2rad(latitude_deg)

    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    sinlon = np.sin(lon)
    coslon = np.cos(lon)
    e2 = e * e

    # Intermediate calculation (prime vertical radius of curvature)
    N = a / np.sqrt(1.0 - e2 * sinlat * sinlat)
    nalt = N + height_m

    x = nalt * coslat * coslon
    y = nalt * coslat * sinlon
    z = ((1.0 - e2) * N + height_m) * sinlat

    return XYZ(x, y, z)


def convert_decimal_units_to_dms(value: float) -> DMS:
    """
    Convert from longitude/latitude to Degrees, minutes and seconds
    >>> latitude_deg = 41.528442316
    >>> dms = convert_decimal_units_to_dms(latitude_deg)
    >>> "{:.0f} {:.0f} {:.7f}".format(*dms)
    '41 31 42.3923376'
    >>> longitude_deg = 2.434318900
    >>> dms = convert_decimal_units_to_dms(longitude_deg)
    >>> "{:.0f} {:.0f} {:.7f}".format(*dms)
    '2 26 3.5480400'
    """

    degrees = int(value)
    minutes_float = (value - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60

    return DMS(degrees, minutes, seconds)


def xyz_to_lla(x, y, z, a=WGS84_A, e=WGS84_E) -> LLA:
    """
    Convert from Cartesian XYZ-ECEF coordinates to geodetic coordinates
    (relative to a reference ellipsoid which defaults to WGS84).

    Output longitude and latitude are expressed in degrees

    The x, y and z values can be vectors

    >>> x, y, z = xyz_to_lla(4807314.3520, 98057.0330, 4176767.6160)
    >>> assert math.isclose(x, 1.1685266980613551)
    >>> assert math.isclose(y, 41.170001652314625)
    >>> assert math.isclose(z, 173.4421242615208)

    >>> xs = np.array([4807314.3520, 4807315.3520, 4807316.3520])
    >>> ys = np.array([98057.0330, 98058.0330, 98059.0330])
    >>> zs = np.array([4176767.6160, 4176768.6160, 4176769.6160])
    >>> lon, lat, height = xyz_to_lla(xs, ys, zs)
    >>> "{0:.7f}".format(lon[0])
    '1.1685267'
    >>> "{0:.7f}".format(lon[1])
    '1.1685384'
    >>> "{0:.7f}".format(lon[2])
    '1.1685500'
    >>> "{0:.7f}".format(lat[0])
    '41.1700017'
    >>> "{0:.7f}".format(lat[1])
    '41.1700024'
    >>> "{0:.7f}".format(lat[2])
    '41.1700031'
    >>> "{0:.7f}".format(height[0])
    '173.4421243'
    >>> "{0:.7f}".format(height[1])
    '174.8683741'
    >>> "{0:.7f}".format(height[2])
    '176.2946241'
    """

    a2 = a ** 2  # Squared of radius, for convenience
    e2 = e ** 2  # Squared of eccentricity, for convenience

    b = np.sqrt(a2 * (1 - e2))
    b2 = b ** 2

    ep = np.sqrt((a2 - b2) / b2)

    p = np.sqrt(x ** 2 + y ** 2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lon = np.fmod(lon, 2 * np.pi)

    sinth = np.sin(th)
    costh = np.cos(th)

    lat = np.arctan2((z + b * (ep ** 2) * (sinth ** 3)), (p - e2 * a * costh ** 3))
    sinlat = np.sin(lat)

    N = a / np.sqrt(1 - e2 * (sinlat ** 2))
    alt = p / np.cos(lat) - N

    return LLA(np.rad2deg(lon), np.rad2deg(lat), alt)


def body_to_enu_matrix(yaw_deg, pitch_deg, roll_deg):
    """
    Computes the matrix that transforms from Body frame to ENU

    This accepts scalars or vector of scalars.

    Also, the yaw, pitch and roll can be expressed as radians (radians=True)
    or degrees (radians=False)
    """

    # Convert from degrees to radians if necessary
    ya = np.deg2rad(yaw_deg)
    pi = np.deg2rad(pitch_deg)
    ro = np.deg2rad(roll_deg)

    # Compute the sines and cosines, used later to compute the elements
    # of the transformation matrix
    siny = np.sin(ya)
    cosy = np.cos(ya)
    sinp = np.sin(pi)
    cosp = np.cos(pi)
    sinr = np.sin(ro)
    cosr = np.cos(ro)

    cosy_cosr = cosy * cosr
    siny_cosr = siny * cosr
    sinp_sinr = sinp * sinr
    sinr_siny = sinr * siny

    # Compute the matrix coefficients
    m = np.zeros((3, 3), dtype=object)

    m[0, 0] = cosp * cosy
    m[0, 1] = sinp_sinr * cosy - siny_cosr
    m[0, 2] = cosy_cosr * sinp + sinr_siny

    m[1, 0] = cosp * siny
    m[1, 1] = sinp_sinr * siny + cosr * cosy
    m[1, 2] = siny_cosr * sinp - sinr * cosy

    m[2, 0] = - sinp
    m[2, 1] = sinr * cosp
    m[2, 2] = cosr * cosp

    return m


def body_to_enu(yaw_deg, pitch_deg, roll_deg, x, y, z, matrix=None):
    """
    Converts from Body reference frame to ENU reference point.

    This accepts scalars or vector of scalars.

    Also, the yaw, pitch and roll can be expressed as radians (radians=True)
    or degrees (radians=False)

    TEST RATIONALE HAS TO BE CHANGED ACCORDING TO THE CHANGE IN THE BODY_TO_ENU_MATRIX RESULT INTERPRETATION:
    BODY_TO_ENU_MATRIX IS ACTUALLY IMPLEMENTING BODY2NED_MATRIX!!!!!!

    >>> body_to_enu(0, 0, 0, 1, 1, 1)
    (1.0, 1.0, -1.0)

    >>> enu = body_to_enu(90, 0, 0, 1, 1, 1)
    >>> np.round(enu)
    array([ 1., -1., -1.])

    >>> enu = body_to_enu(0, 90, 0, 1, 1, 1)
    >>> np.round(enu)
    array([1., 1., 1.])

    >>> enu = body_to_enu(0, 0, 90, 1, 1, 1)
    >>> np.round(enu)
    array([-1.,  1., -1.])

    >>> yaws = [0] * 3; pitches = [0] * 3; rolls = [0] * 3
    >>> xs = [1] * 3; ys = [1] * 3; zs = [1]*3
    >>> es, ns, us = body_to_enu(yaws, pitches, rolls, xs, ys, zs)
    >>> es
    array([1., 1., 1.])
    >>> ns
    array([1., 1., 1.])
    >>> us
    array([-1., -1., -1.])
    """

    if matrix is None:
        m = body_to_enu_matrix(yaw_deg, pitch_deg, roll_deg)

    n = m[0, 0] * x + m[0, 1] * y + m[0, 2] * z
    e = m[1, 0] * x + m[1, 1] * y + m[1, 2] * z
    d = m[2, 0] * x + m[2, 1] * y + m[2, 2] * z

    return e, n, -d


def enu_to_body(yaw_deg, pitch_deg, roll_deg, e, n, u, matrix=None):
    """
    Converts from ENU to XYZ Body reference frame

    This accepts scalars or vector of scalars.

    Also, the yaw, pitch and roll can be expressed as radians (radians=True)
    or degrees (radians=False)


    >>> np.round(enu_to_body(0, 0, 0, *body_to_enu(0, 0, 0, 0.2, 0.2, 1.5)), decimals=1)
    array([0.2, 0.2, 1.5])
    """

    if matrix is None:
        m = body_to_enu_matrix(yaw_deg, pitch_deg, roll_deg)

    x = m[0, 0] * n + m[1, 0] * e + m[2, 0] * -u
    y = m[0, 1] * n + m[1, 1] * e + m[2, 1] * -u
    z = m[0, 2] * n + m[1, 2] * e + m[2, 2] * -u

    return x, y, z


def enu_to_ecef_matrix(longitude_deg, latitude_deg):
    """
    Computes the transformation matrix from ENU to XYZ (ECEF) and returns
    the matrix coefficients stored as a a numpy two-dimensional array

    This accepts scalars or vector of scalars.

    The latitude and longitude can be expressed either in radians (radians=True)
    or degrees (radians = True)
    """

    n_samples = np.size(longitude_deg)

    # Compute the sine and cos to avoid recalculations
    sinlon = np.sin(np.deg2rad(longitude_deg))
    coslon = np.cos(np.deg2rad(longitude_deg))
    sinlat = np.sin(np.deg2rad(latitude_deg))
    coslat = np.cos(np.deg2rad(latitude_deg))

    m = np.zeros((3, 3), dtype=object)

    # Compute the elemnts of the matrix

    m[0, 0] = -sinlon
    m[1, 0] = coslon
    m[2, 0] = 0 if n_samples == 1 else np.zeros(n_samples)

    m[0, 1] = -sinlat * coslon
    m[1, 1] = -sinlat * sinlon
    m[2, 1] = coslat

    m[0, 2] = coslat * coslon
    m[1, 2] = coslat * sinlon
    m[2, 2] = sinlat

    return m


def enu_to_ecef(longitude_deg, latitude_deg, e, n, u, matrix=None):
    """
    Converts from East North and Up to Cartesian XYZ ECEF

    This accepts scalars or vector of scalars.

    The latitude and longitude can be expressed either in radians (radians=True)
    or degrees (radians = True)

    >>> enu = (0.5, 0.5, 1.0)
    >>> enu_to_ecef(0, 0, *enu)
    (1.0, 0.5, 0.5)

    >>> enu = (0.0, 0.0, 1.0)
    >>> enu_to_ecef(0, 0, *enu)
    (1.0, 0.0, 0.0)

    >>> lons = [90, 180, 270]
    >>> lats = [0, 0, 0]
    >>> es = [-1, -1, +1]
    >>> ns = [1, 1, 1]
    >>> us = [1, -1, -1]
    >>> dxyz = enu_to_ecef(lons, lats, es, ns, us)
    >>> round(dxyz[0][0], 8)
    1.0
    >>> round(dxyz[0][1], 8)
    1.0
    >>> round(dxyz[0][2], 8)
    1.0
    >>> round(dxyz[1][0], 8)
    1.0
    >>> round(dxyz[1][1], 8)
    1.0
    >>> round(dxyz[1][2], 8)
    1.0
    >>> round(dxyz[2][0], 8)
    1.0
    >>> round(dxyz[2][1], 8)
    1.0
    >>> round(dxyz[2][2], 8)
    1.0
    """

    if matrix is None:
        m = enu_to_ecef_matrix(longitude_deg, latitude_deg)

    x = m[0, 0] * e + m[0, 1] * n + m[0, 2] * u
    y = m[1, 0] * e + m[1, 1] * n + m[1, 2] * u
    z = m[2, 0] * e + m[2, 1] * n + m[2, 2] * u

    return x, y, z


def xyz_to_enu(ref_pos, x, y, z, a=WGS84_A, e=WGS84_E):
    """
    Converts from Cartesian XYZ ECEF to East North and Up given a reference
    position in Cartesian absolute XYZ ECEF.

    >>> xyz = [WGS84_A, 0.0, 0.0]   # Reference position
    >>> dxyz = (1.0, 0.0, 0.0)      # Deviation relative to reference position
    >>> enu = xyz_to_enu(xyz, *dxyz)  # Conversion to ENU at reference position
    >>> enu
    (0.0, 0.0, 1.0)

    >>> enu = (0.5, 0.5, 1.0)
    >>> dxyz = enu_to_ecef(0, 0, *enu)
    >>> dxyz
    (1.0, 0.5, 0.5)
    >>> xyz_to_enu(xyz, *dxyz) == enu
    True
    """

    lla = xyz_to_lla(*ref_pos, a=a, e=e)

    return ecef_to_enu(lla.longitude, lla.latitude, x, y, z)


def ecef_to_enu(longitude_deg, latitude_deg, x, y, z, matrix=None):
    """
    Converts from Cartesian XYZ ECEF to East North and Up

    This accepts scalars or vector of scalars.

    The latitude and longitude can be expressed either in radians (radians=True)
    or degrees (radians = True)

    >>> dxyz = (1, 0, 0)
    >>> ecef_to_enu(0, 0, *dxyz)
    (0.0, 0.0, 1.0)

    >>> dxyz = (1, 0.5, 0.5)
    >>> ecef_to_enu(0, 0, *dxyz)
    (0.5, 0.5, 1.0)

    >>> enu = (0.5, 0.5, 1)
    >>> dxyz = enu_to_ecef(0, 0, *enu)
    >>> dxyz
    (1.0, 0.5, 0.5)
    >>> ecef_to_enu(0, 0, *dxyz) == enu
    True
    """

    if matrix is None:
        m = enu_to_ecef_matrix(longitude_deg, latitude_deg)

    e = m[0, 0] * x + m[1, 0] * y + m[2, 0] * z
    n = m[0, 1] * x + m[1, 1] * y + m[2, 1] * z
    u = m[0, 2] * x + m[1, 2] * y + m[2, 2] * z

    return e, n, u


def body_to_ecef(longitude, latitude, yaw, pitch, roll, x, y, z):
    """
    Transform from body fixed reference frame to Cartesian XYZ-ECEF
    """

    enu = body_to_enu(yaw, pitch, roll, x, y, z)

    return enu_to_ecef(longitude, latitude, *enu)


def transpose_pos_geodetic(longitude, latitude, height, yaw, pitch, roll, lx, ly, lz):
    """
    Apply lever arm (l=[lx,ly,lz]) to position (longitude,latitude, height) according to rover attitude (yaw,pitch,roll)
    """

    leverArmEnu = body_to_enu(yaw, pitch, roll, lx, ly, lz)
    leverArmNed = (leverArmEnu[1], leverArmEnu[0], -1 * leverArmEnu[2])

    deltaPos = small_perturbation_cart_to_geodetic(leverArmNed[0], leverArmNed[1], leverArmNed[2], latitude, height)

    return latitude + deltaPos[0], longitude + deltaPos[1], height + deltaPos[2]


def small_perturbation_cart_to_geodetic(n, e, d, latitude_deg, h):
    """
    Transform small perturbations from cartesian (in local navigation frame NED)  to geodetic, small angle
    aproximation has to apply, which means norm(n,e,d) <<< Earth Radius
    """

    lat = np.deg2rad(latitude_deg)

    sinlat = np.sin(lat)

    RE = WGS84_A / np.sqrt((1 - (sinlat ** 2) * (WGS84_E ** 2)) ** 3)
    RN = WGS84_A * (1 - WGS84_E ** 2) / np.sqrt(1 - (sinlat ** 2) * (WGS84_E ** 2))

    deltaLat = n / (RN + h)
    deltaLon = e / ((RE + h) * np.cos(lat))
    deltaH = -1 * d

    return np.rad2deg(deltaLat), np.rad2deg(deltaLon), deltaH


def haversine(lon1_deg, lat1_deg, lon2_deg, lat2_deg, r=6371):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) using the Haversine formula

    Return units: kilometers

    Extracted from http://stackoverflow.com/questions/4913349

    >>> haversine(0, 0, 0, 0)
    0.0

    Example extracted from https://rosettacode.org/wiki/Haversine_formula

    >>> haversine(-86.67, 36.12, -118.40, 33.94, r=6371.8)
    2886.8068907353736
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1_deg, lat1_deg, lon2_deg, lat2_deg])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return c * r


def area_triangle_in_sphere(coords, r=6371):
    """

    Area = pi*R^2*E/180

    R = radius of sphere
    E = spherical excess of triangle, E = A + B + C - 180
    A, B, C = angles of spherical triangle in degrees

    tan(E/4) = sqrt(tan(s/2)*tan((s-a)/2)*tan((s-b)/2)*tan((s-c)/2))

    where

    a, b, c = sides of spherical triangle
    s = (a + b + c)/2

    http://mathforum.org/library/drmath/view/65316.html

    # Compute approximate area of India (3287263 km2)
    >>> pWest = (68.752441, 23.483401)
    >>> pEast = (90.153809, 22.87744)
    >>> pNorth = (76.723022, 32.87036)
    >>> pSouth = (77.45636, 8.119053)
    >>> triangle1 = (pWest,pEast,pNorth)
    >>> triangle2 = (pWest,pEast,pSouth)
    >>> area_triangle_in_sphere(triangle1) + area_triangle_in_sphere(triangle2)
    3028215.293314756
    """

    # Compute the spherical areas using the haversine method
    a = haversine(coords[0][0], coords[0][1], coords[1][0], coords[1][1], r=1)
    b = haversine(coords[1][0], coords[1][1], coords[2][0], coords[2][1], r=1)
    c = haversine(coords[0][0], coords[0][1], coords[2][0], coords[2][1], r=1)

    s = (a + b + c) / 2.0

    tanE = math.sqrt(math.tan(s / 2) * math.tan((s - a) / 2) * math.tan((s - b) / 2) * math.tan((s - c) / 2))

    E = math.atan(tanE * 4.0)

    area = r * r * E

    return area


class Dop(object):
    """
    Class that implements computation of Dilution of Precision (DOP)


    Check that dop from NMEA GPGSV is the same as the one in NMEA GPGSA message
    (further investigations need to be conducted if NMEA generated by u-blox NEO 6P,
    discrepancies have been found)

    $GPGSA,A,3,10,07,05,02,29,04,08,13,,,,,1.72,1.03,1.38*0A
    $GPGSV,3,1,11,10,63,137,17,07,61,098,15,05,59,290,20,08,54,157,30*70
    $GPGSV,3,2,11,02,39,223,19,13,28,070,17,26,23,252,,04,14,186,14*79
    $GPGSV,3,3,11,29,09,301,24,16,09,020,,36,,,*76

    >>> elAz = [ElAz(63,137),
    ...         ElAz(61, 98),
    ...         ElAz(59,290),
    ...         ElAz(39,223),
    ...         ElAz( 9,301),
    ...         ElAz(14,186),
    ...         ElAz(54,157),
    ...         ElAz(28, 70)]
    >>> dop = Dop(elAz)
    >>> f'{dop.pdop():.2f}'
    '1.72'
    >>> f'{dop.vdop():.2f}'
    '1.38'
    >>> f'{dop.hdop():.2f}'
    '1.04'


    Verification through the example of DOP computation proposed
    in Exercise 6-5 (page 228) of:

    Misra, P., Enge, P., "Global Positioning System: Signals,
    Measurements and Performance", 2nd Edition. 2006

    Increasing 30 degrees the azimuth position of 3 sats (+ one in zenith), doubles the vdop value
    leaving pdop approximately constant

    >>> el_az_list = [ElAz( 0,   0),
    ...               ElAz( 0, 120),
    ...               ElAz( 0, 240),
    ...               ElAz(90,   0)]
    >>> dop = Dop(el_az_list)
    >>> f'{dop.vdop():.2f}'
    '1.15'
    >>> f'{dop.hdop():.2f}'
    '1.15'

    >>> el_az_list = [ElAz(30,   0),
    ...               ElAz(30, 120),
    ...               ElAz(30, 240),
    ...               ElAz(90,  0)]
    >>> dop = Dop(el_az_list)
    >>> f'{dop.vdop():.2f}'
    '2.31'
    >>> f'{dop.hdop():.2f}'
    '1.33'
    """

    def __init__(self, el_az_list: List[ElAz]) -> None:

        self.el_az_list = el_az_list
        self.__G()  # generate G matrix
        self.__Q()  # generate Q matrix

    def __G(self) -> None:
        """
        Construct geometry matrix G
        """

        elevations = np.radians([v.elevation for v in self.el_az_list])
        azimuths = np.radians([v.azimuth for v in self.el_az_list])

        n_satellites = len(self.el_az_list)

        cosel = np.cos(elevations)
        gx = cosel * np.sin(azimuths)
        gy = cosel * np.cos(azimuths)
        gz = np.sin(elevations)

        rows = list(zip(gx, gy, gz, [1] * n_satellites))

        self.G = np.array(rows)

    def __Q(self) -> None:
        """
        Construct co-factor matrix
        """
        try:
            self.Q = np.linalg.inv(np.dot(np.matrix.transpose(self.G), self.G))
        except np.linalg.linalg.LinAlgError as e:
            self.Q = np.full([4, 4], np.nan)
            roktools.logger.warning("Could not compute DOP. Reason {}".format(e))

    def gdop(self) -> float:
        """
        Geometric DOP
        """

        return np.sqrt(self.Q[0, 0] + self.Q[1, 1] + self.Q[2, 2] + self.Q[3, 3])

    def pdop(self) -> float:
        """
        Position DOP
        """

        return np.sqrt(self.Q[0, 0] + self.Q[1, 1] + self.Q[2, 2])

    def tdop(self) -> float:
        """
        Time DOP
        """

        return np.sqrt(self.Q[3, 3])

    def vdop(self) -> float:
        """
        Vertical DOP
        """

        return np.sqrt(self.Q[2, 2])

    def hdop(self) -> float:
        """
        Horizontal DOP
        """

        return np.sqrt(self.Q[0, 0] + self.Q[1, 1])

    def __repr__(self) -> str:

        return f'DOP - position: {self.pdop():.3f} horizontal:  {self.hdop():.3f} vertical: {self.vdop():.3f}'


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
