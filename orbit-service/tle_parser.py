#!/usr/bin/env python3
"""
TLE Parser Implementation
Parses Two-Line Element (TLE) sets according to AAS 06-675 paper specifications
Includes Long-Period Periodic (LPP) terms for secular effects computation
"""

import re
import math
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple
from sgp4.api import Satrec, WGS72, WGS84

# Earth gravitational constants for LPP and SPP calculations
EARTH_RADIUS_KM = 6378.137  # Earth equatorial radius in km
J2 = 1.08262668e-3  # Earth second zonal harmonic
J3 = -2.53215306e-6  # Earth third zonal harmonic
J4 = -1.61098761e-6  # Earth fourth zonal harmonic


class TLEParser:
    """Canonical TLE parser implementing AAS 06-675 specifications"""
    
    def __init__(self):
        self.line1_pattern = re.compile(r'^1\s(\d{5})[A-Z]\s{2}(\d{2})(\d{3}\.\d{8})\s(.{10})\s(\d)\s(\d{5})\s(.{1})\s(\d{10})\s(.{1})\s(\d{10})\s(.{1})\s(\d{10})\s(.{5})\s(\d{4})([+-]\d{4})')
        self.line2_pattern = re.compile(r'^2\s(\d{5})\s(\d{8})\s(\d{8})\s(\d{7})\s(\d{8})\s(\d{8})\s(\d{11})\s(\d{5})')
        
    def calculate_nodal_precession(self, inclination_deg: float, eccentricity: float, 
                                 mean_motion_rev_per_day: float) -> float:
        """
        Calculate secular nodal precession rate (dot(Ω))
        
        Formula: dot(Ω) = -1.5 * J2 * (Re / p)^2 * n0 * cos(i)
        where p = a * (1 - e^2) is the semi-latus rectum
        and a = (Re * (n0 * 1440 / 2π)^(2/3)) is the semi-major axis
        
        Args:
            inclination_deg: Inclination in degrees
            eccentricity: Eccentricity (unitless)
            mean_motion_rev_per_day: Mean motion in revolutions per day
            
        Returns:
            Nodal precession rate in degrees per day
        """
        # Convert to radians
        inclination_rad = math.radians(inclination_deg)
        
        # Convert mean motion to radians per minute
        n0 = mean_motion_rev_per_day * 2 * math.pi / 1440  # rev/day to rad/min
        
        # Calculate semi-major axis
        # For TLE mean motion, we use the relationship with orbital period
        # n = sqrt(μ/a^3) => a = (μ/n^2)^(1/3)
        # Using simplified model with Earth's gravitational parameter
        mu = 398600.4418  # Earth's gravitational parameter in km^3/min^2
        a = (mu / (n0 ** 2)) ** (1/3) if n0 != 0 else 0
        
        # Calculate semi-latus rectum
        p = a * (1 - eccentricity ** 2) if a != 0 else 0
        
        # Avoid division by zero
        if p == 0:
            return 0
        
        # Calculate nodal precession rate
        nodal_precession = -1.5 * J2 * (EARTH_RADIUS_KM / p) ** 2 * n0 * math.cos(inclination_rad)
        
        # Convert back to degrees per day
        return math.degrees(nodal_precession) * 1440

    def solve_kepler_equation(self, mean_anomaly_rad: float, eccentricity: float, tolerance: float = 1e-12, max_iterations: int = 100) -> float:
        """
        Solve Kepler's equation: M = E - e*sin(E) for eccentric anomaly E
        Using Newton-Raphson iteration method
        
        Args:
            mean_anomaly_rad: Mean anomaly in radians
            eccentricity: Eccentricity (unitless)
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            
        Returns:
            Eccentric anomaly in radians
        """
        # Initial guess for eccentric anomaly
        if eccentricity < 0.8:
            E = mean_anomaly_rad
        else:
            E = math.pi if mean_anomaly_rad > math.pi else mean_anomaly_rad
        
        # Newton-Raphson iteration
        for _ in range(max_iterations):
            f = E - eccentricity * math.sin(E) - mean_anomaly_rad
            df = 1 - eccentricity * math.cos(E)
            
            if abs(df) < 1e-15:  # Avoid division by zero
                break
                
            delta_E = f / df
            E = E - delta_E
            
            if abs(delta_E) < tolerance:
                break
        
        return E

    def calculate_true_anomaly(self, eccentric_anomaly_rad: float, eccentricity: float) -> float:
        """
        Calculate true anomaly from eccentric anomaly
        
        Args:
            eccentric_anomaly_rad: Eccentric anomaly in radians
            eccentricity: Eccentricity (unitless)
            
        Returns:
            True anomaly in radians
        """
        # Formula: tan(f/2) = sqrt((1+e)/(1-e)) * tan(E/2)
        if abs(eccentricity - 1.0) < 1e-10:  # Avoid division by zero for parabolic orbits
            return eccentric_anomaly_rad
            
        sqrt_factor = math.sqrt((1 + eccentricity) / (1 - eccentricity))
        true_anomaly = 2 * math.atan(sqrt_factor * math.tan(eccentric_anomaly_rad / 2))
        
        return true_anomaly

    def calculate_gmst(self, julian_date: float) -> float:
        """
        Calculate Greenwich Mean Sidereal Time (GMST) in hours
        
        Formula: GMST = 18.697374558 + 0.06570982441908 * d + 1.00273790935 * h + 0.000026 * t^2
        where:
        - d = days from J2000.0 (Julian Date 2451545.0)
        - h = UT1 hours from midnight
        - t = centuries from J2000.0
        
        Args:
            julian_date: Julian date (UT1)
            
        Returns:
            GMST in hours (0-24)
        """
        # J2000.0 epoch
        j2000 = 2451545.0
        
        # Days from J2000.0
        d = julian_date - j2000
        
        # UT1 hours from midnight (fractional part of julian date * 24)
        h = (julian_date - int(julian_date)) * 24
        
        # Centuries from J2000.0
        t = d / 36525.0
        
        # Calculate GMST using the specified formula
        gmst_hours = 18.697374558 + 0.06570982441908 * d + 1.00273790935 * h + 0.000026 * t * t
        
        # Normalize to 0-24 hours
        gmst_hours = gmst_hours % 24.0
        if gmst_hours < 0:
            gmst_hours += 24.0
            
        return gmst_hours

    def datetime_to_julian_date(self, dt: datetime) -> float:
        """
        Convert datetime to Julian Date
        
        Args:
            dt: Datetime object (assumed to be UTC/UT1)
            
        Returns:
            Julian Date
        """
        # Ensure datetime is in UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            dt = dt.astimezone(timezone.utc)
        
        # Julian Date calculation
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute
        second = dt.second + dt.microsecond / 1e6
        
        # Convert to fractional day
        fractional_day = day + (hour + minute/60.0 + second/3600.0) / 24.0
        
        # Adjust for January and February
        if month <= 2:
            year -= 1
            month += 12
        
        # Julian Date formula
        a = int(year / 100)
        b = 2 - a + int(a / 4)
        
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + fractional_day + b - 1524.5
        
        return jd

    def teme_to_ecef(self, r_teme: np.ndarray, epoch_datetime: datetime) -> np.ndarray:
        """
        Convert TEME (True Equator Mean Equinox) coordinates to ECEF (Earth-Centered, Earth-Fixed)
        
        Args:
            r_teme: Position vector in TEME coordinates [x, y, z] (km)
            epoch_datetime: Epoch datetime for GMST calculation
            
        Returns:
            Position vector in ECEF coordinates [x, y, z] (km)
        """
        # Convert epoch to Julian Date
        jd = self.datetime_to_julian_date(epoch_datetime)
        
        # Calculate GMST in hours
        gmst_hours = self.calculate_gmst(jd)
        
        # Convert GMST to radians
        gmst_rad = math.radians(gmst_hours * 15.0)  # 15 degrees per hour
        
        # Create rotation matrix for TEME to ECEF conversion
        # R = [[cos(θ), sin(θ), 0], [-sin(θ), cos(θ), 0], [0, 0, 1]]
        # where θ = GMST
        cos_gmst = math.cos(gmst_rad)
        sin_gmst = math.sin(gmst_rad)
        
        rotation_matrix = np.array([
            [cos_gmst, sin_gmst, 0],
            [-sin_gmst, cos_gmst, 0],
            [0, 0, 1]
        ])
        
        # Transform TEME to ECEF
        r_ecef = rotation_matrix @ r_teme
        
        return r_ecef

    def propagate_orbit(self, tle_data: Dict[str, Any], tsince_minutes: float) -> Dict[str, Any]:
        """
        Propagate orbital elements and compute position in TEME coordinates
        
        Args:
            tle_data: Dictionary containing parsed TLE elements
            tsince_minutes: Time since epoch in minutes
            
        Returns:
            Dictionary containing propagated elements and position vector
        """
        # Extract orbital elements
        inclination_deg = tle_data['inclination_deg']
        raan_deg = tle_data['raan_deg']
        eccentricity = tle_data['eccentricity']
        arg_perigee_deg = tle_data['arg_perigee_deg']
        mean_anomaly_deg = tle_data['mean_anomaly_deg']
        mean_motion_rev_per_day = tle_data['mean_motion_rev_per_day']
        
        # Convert to radians
        inclination_rad = math.radians(inclination_deg)
        raan_rad = math.radians(raan_deg)
        arg_perigee_rad = math.radians(arg_perigee_deg)
        mean_anomaly_rad = math.radians(mean_anomaly_deg)
        
        # Convert mean motion to radians per minute
        mean_motion_rad_per_min = mean_motion_rev_per_day * 2 * math.pi / 1440
        
        # Calculate semi-major axis using Kepler's third law: n^2 * a^3 = μ
        mu = 398600.4418  # Earth's gravitational parameter in km^3/s^2
        # Convert mean motion to rad/s for proper units
        mean_motion_rad_per_sec = mean_motion_rad_per_min / 60
        semi_major_axis_km = (mu / (mean_motion_rad_per_sec ** 2)) ** (1/3)
        
        # Update mean anomaly for time tsince
        updated_mean_anomaly_rad = mean_anomaly_rad + mean_motion_rad_per_min * tsince_minutes
        
        # Solve Kepler's equation for eccentric anomaly
        eccentric_anomaly_rad = self.solve_kepler_equation(updated_mean_anomaly_rad, eccentricity)
        
        # Calculate true anomaly
        true_anomaly_rad = self.calculate_true_anomaly(eccentric_anomaly_rad, eccentricity)
        
        # Compute position in orbital plane
        r = semi_major_axis_km * (1 - eccentricity**2) / (1 + eccentricity * math.cos(true_anomaly_rad))
        x_orbital = r * math.cos(true_anomaly_rad)
        y_orbital = r * math.sin(true_anomaly_rad)
        
        # Transform to TEME coordinates
        # Position vector in orbital plane (z=0)
        r_orbital = np.array([x_orbital, y_orbital, 0])
        
        # Rotation matrices
        cos_raan = math.cos(raan_rad)
        sin_raan = math.sin(raan_rad)
        cos_inc = math.cos(inclination_rad)
        sin_inc = math.sin(inclination_rad)
        cos_arg = math.cos(arg_perigee_rad)
        sin_arg = math.sin(arg_perigee_rad)
        
        # Combined rotation matrix from orbital plane to TEME
        R11 = cos_raan * cos_arg - sin_raan * sin_arg * cos_inc
        R12 = -cos_raan * sin_arg - sin_raan * cos_arg * cos_inc
        R13 = sin_raan * sin_inc
        
        R21 = sin_raan * cos_arg + cos_raan * sin_arg * cos_inc
        R22 = -sin_raan * sin_arg + cos_raan * cos_arg * cos_inc
        R23 = -cos_raan * sin_inc
        
        R31 = sin_arg * sin_inc
        R32 = cos_arg * sin_inc
        R33 = cos_inc
        
        # Transform to TEME
        r_teme = np.array([
            R11 * x_orbital + R12 * y_orbital,
            R21 * x_orbital + R22 * y_orbital,
            R31 * x_orbital + R32 * y_orbital
        ])
        
        # Calculate orbital radius
        orbital_radius_km = np.linalg.norm(r_teme)
        
        # Calculate current epoch time for ECEF conversion
        epoch_datetime = tle_data['epoch_datetime']
        current_time = epoch_datetime + timedelta(minutes=tsince_minutes)
        
        # Convert TEME to ECEF coordinates
        r_ecef = self.teme_to_ecef(r_teme, current_time)
        
        return {
            'tsince_minutes': tsince_minutes,
            'semi_major_axis_km': semi_major_axis_km,
            'anomalies': {
                'mean_anomaly_deg': math.degrees(updated_mean_anomaly_rad),
                'eccentric_anomaly_deg': math.degrees(eccentric_anomaly_rad),
                'true_anomaly_deg': math.degrees(true_anomaly_rad)
            },
            'position_teme_km': {
                'x': float(r_teme[0]),
                'y': float(r_teme[1]),
                'z': float(r_teme[2])
            },
            'position_ecef_km': {
                'x': float(r_ecef[0]),
                'y': float(r_ecef[1]),
                'z': float(r_ecef[2])
            },
            'orbital_radius_km': orbital_radius_km,
            'orbital_plane_coords_km': {
                'x': x_orbital,
                'y': y_orbital
            },
            'gmst_hours': self.calculate_gmst(self.datetime_to_julian_date(current_time))
        }
        
    def parse_tle(self, line1: str, line2: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse TLE according to AAS 06-675 specifications
        
        Args:
            line1: First line of TLE
            line2: Second line of TLE
            name: Satellite name (optional)
            
        Returns:
            Dictionary with parsed TLE elements
        """
        # Parse Line 1
        line1_data = self._parse_line1(line1)
        
        # Parse Line 2
        line2_data = self._parse_line2(line2)
        
        # Combine data
        tle_data = {
            'name': name or 'Unknown Satellite',
            'norad_id': line1_data['norad_id'],
            'line1': line1,
            'line2': line2,
            **line1_data,
            **line2_data
        }
        
        # Add secular effects
        tle_data['nodal_precession'] = self.calculate_nodal_precession(
            tle_data['inclination_deg'], tle_data['eccentricity'], tle_data['mean_motion_rev_per_day']
        )
        
        return tle_data
        
    def _parse_line1(self, line1: str) -> Dict[str, Any]:
        """Parse TLE Line 1"""
        # Extract NORAD ID (columns 3-7)
        norad_id = int(line1[2:7])
        
        # Extract classification (column 8)
        classification = line1[7]
        
        # Extract international designator (columns 10-17)
        int_designator = line1[9:17].strip()
        
        # Parse international designator
        if len(int_designator) >= 5:
            launch_year = int(int_designator[:2])
            launch_num = int(int_designator[2:5])
            launch_piece = int_designator[5:].strip() if len(int_designator) > 5 else ""
        else:
            launch_year = launch_num = None
            launch_piece = ""
        
        # Extract epoch (columns 19-32)
        epoch_str = line1[18:32]
        epoch_year = int(epoch_str[:2])
        epoch_day = float(epoch_str[2:])
        
        # Y2K fix for epoch year
        if epoch_year >= 57:
            full_year = 1900 + epoch_year
        else:
            full_year = 2000 + epoch_year
        
        # Convert epoch to datetime
        epoch_datetime = datetime(full_year, 1, 1, tzinfo=timezone.utc) + timedelta(days=epoch_day - 1)
        
        # Extract first derivative of mean motion (columns 34-43)
        first_deriv_str = line1[33:43].strip()
        first_deriv_mean_motion = float(first_deriv_str) if first_deriv_str else 0.0
        
        # Extract second derivative of mean motion (columns 45-52)
        second_deriv_str = line1[44:52].strip()
        if second_deriv_str:
            # Handle scientific notation format
            if 'E' in second_deriv_str or 'e' in second_deriv_str:
                second_deriv_mean_motion = float(second_deriv_str)
            else:
                # TLE format: assume decimal point before first digit
                mantissa = second_deriv_str[:-2]
                exponent = int(second_deriv_str[-2:])
                second_deriv_mean_motion = float('0.' + mantissa) * (10 ** exponent)
        else:
            second_deriv_mean_motion = 0.0
        
        # Extract BSTAR drag term (columns 54-61)
        bstar_str = line1[53:61].strip()
        if bstar_str:
            # Handle scientific notation format
            if 'E' in bstar_str or 'e' in bstar_str:
                bstar_drag = float(bstar_str)
            else:
                # TLE format: assume decimal point before first digit
                mantissa = bstar_str[:-2]
                exponent = int(bstar_str[-2:])
                bstar_drag = float('0.' + mantissa) * (10 ** exponent)
        else:
            bstar_drag = 0.0
        
        # Extract ephemeris type (column 63)
        ephemeris_type = int(line1[62]) if line1[62].isdigit() else 0
        
        # Extract element number (columns 65-68)
        element_num = int(line1[64:68])
        
        return {
            'norad_id': norad_id,
            'classification': classification,
            'int_designator_launch_year': launch_year,
            'int_designator_launch_num': launch_num,
            'int_designator_launch_piece': launch_piece,
            'epoch_year': epoch_year,
            'epoch_datetime': epoch_datetime,
            'first_deriv_mean_motion': first_deriv_mean_motion,
            'second_deriv_mean_motion': second_deriv_mean_motion,
            'bstar_drag': bstar_drag,
            'ephemeris_type': ephemeris_type,
            'element_num': element_num
        }
    
    def _parse_line2(self, line2: str) -> Dict[str, Any]:
        """Parse TLE Line 2"""
        # Extract inclination (columns 9-16)
        inclination_deg = float(line2[8:16])
        
        # Extract RAAN (columns 18-25)
        raan_deg = float(line2[17:25])
        
        # Extract eccentricity (columns 27-33)
        eccentricity_str = line2[26:33]
        eccentricity = float('0.' + eccentricity_str)
        
        # Extract argument of perigee (columns 35-42)
        arg_perigee_deg = float(line2[34:42])
        
        # Extract mean anomaly (columns 44-51)
        mean_anomaly_deg = float(line2[43:51])
        
        # Extract mean motion (columns 53-63)
        mean_motion_rev_per_day = float(line2[52:63])
        
        # Extract revolution number (columns 64-68)
        revolution_num = int(line2[63:68])
        
        return {
            'inclination_deg': inclination_deg,
            'raan_deg': raan_deg,
            'eccentricity': eccentricity,
            'arg_perigee_deg': arg_perigee_deg,
            'mean_anomaly_deg': mean_anomaly_deg,
            'mean_motion_rev_per_day': mean_motion_rev_per_day,
            'revolution_num': revolution_num
        }
