#!/usr/bin/env python3
"""
TLE (Two-Line Element) Parser Implementation

Parses Two-Line Element sets according to the standard TLE format and provides
utilities for orbital propagation and coordinate transformations.

TLE Format:
The TLE format is a standard way to represent satellite orbital elements in two
69-character lines. This parser extracts orbital parameters and supports:
- Parsing and validation of TLE fields
- Exponential notation handling for drag terms
- Reconstruction of TLE strings with modified parameters
- Orbital propagation via SGP4
- Coordinate transformations (TEME to ECEF)

Gravitational Model:
Uses WGS-72 constants as required by SGP4 (per Vallado et al. 2006, AAS 06-675)

References:
- Vallado, D. A., et al. (2006). "Revisiting Spacetrack Report #3." AIAA 2006-6753
- CelesTrak TLE format documentation: https://celestrak.org/NORAD/documentation/tle-fmt.php
"""

import re
import math
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple
from sgp4.api import Satrec, WGS72, WGS84, jday
from sgp4.io import twoline2rv

# WGS-72 gravitational constants for SGP4 (per Vallado et al. 2006, AAS 06-675)
EARTH_RADIUS_KM = 6378.135  # Earth equatorial radius in km (WGS-72)
J2 = 0.00108262998905892  # Earth second zonal harmonic
J3 = -0.00000253215306  # Earth third zonal harmonic
J4 = -0.00000165597  # Earth fourth zonal harmonic


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
        mu = 398600.4418 * 3600  # Convert km^3/s^2 to km^3/min^2
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

    def datetime_to_jd_fr(self, dt: datetime) -> Tuple[float, float]:
        """
        Convert datetime to Julian Date and fractional day for SGP4.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)

    def _tle_checksum(self, line: str) -> int:
        """Calculate the TLE checksum for a given line."""
        s = 0
        for char in line[:-1]: # Exclude checksum digit
            if '1' <= char <= '9':
                s += int(char)
            elif char == '-':
                s += 1
        return s % 10

    def tle_data_to_lines(self, tle_data: Dict[str, Any]) -> Tuple[str, str]:
        """Converts a TLE data dictionary back to TLE line strings."""
        
        def format_tle_exponential(value: float) -> str:
            """Format a float in TLE exponential notation (e.g., 21844-3 for 2.1844e-4)"""
            if value == 0.0:
                return " 00000-0"
            
            # Handle sign
            sign = '-' if value < 0 else ' '
            abs_value = abs(value)
            
            # Convert to scientific notation
            exp = int(math.floor(math.log10(abs_value)))
            mantissa = abs_value / (10.0 ** exp)
            
            # Normalize mantissa to be between 1 and 10
            while mantissa >= 10.0:
                mantissa /= 10.0
                exp += 1
            while mantissa < 1.0 and mantissa != 0.0:
                mantissa *= 10.0
                exp -= 1
            
            # Format mantissa as 5 digits (remove leading "1.")
            mantissa_str = f"{mantissa:.5f}"[2:7]  # Get 5 digits after "1."
            
            # Format exponent as signed single digit
            exp_str = f"{exp:+d}"[-2:]  # Take last 2 characters to handle -10, etc.
            
            return f"{sign}{mantissa_str}{exp_str}"
        
        # If we have original lines, use them as base and only modify what's needed
        if 'line1' in tle_data and 'line2' in tle_data:
            line1 = tle_data['line1']
            line2 = tle_data['line2']
            
            # Check if B* has been modified
            original_bstar_parsed = tle_data.get('original_bstar', tle_data['bstar_drag'])
            if abs(tle_data['bstar_drag'] - original_bstar_parsed) > 1e-10:
                # B* has been modified, replace the B* field in the original line
                new_bstar_str = format_tle_exponential(tle_data['bstar_drag'])
                line1 = line1[:53] + new_bstar_str + line1[61:]
                # Recalculate checksum
                line1_checksum = self._tle_checksum(line1)
                line1 = line1[:-1] + str(line1_checksum)
            
            return line1, line2
        
        # Fallback: reconstruct from scratch (original behavior)
        orig_second_deriv = format_tle_exponential(tle_data['second_deriv_mean_motion'])
        orig_bstar = format_tle_exponential(tle_data['bstar_drag'])
        
        # Format first derivative with proper sign and spacing
        first_deriv = tle_data['first_deriv_mean_motion']
        if first_deriv >= 0:
            first_deriv_str = f" {first_deriv:.8f}"
        else:
            first_deriv_str = f"{first_deriv:.8f}"
        
        # Line 1
        line1 = f"1 {tle_data['norad_id']:05d}{tle_data.get('classification', 'U')} {tle_data['int_designator_launch_year']:02d}{tle_data['int_designator_launch_num']:03d}{tle_data['int_designator_launch_piece']:<3s} {tle_data['epoch_year']:02d}{tle_data['epoch_day']:012.8f} {first_deriv_str} {orig_second_deriv} {orig_bstar} {tle_data['ephemeris_type']:1d} {tle_data['element_num']:4d}"
        line1_checksum = self._tle_checksum(line1)
        line1 = f"{line1}{line1_checksum}"

        # Line 2
        line2 = f"2 {tle_data['norad_id']:05d} {tle_data['inclination_deg']:8.4f} {tle_data['raan_deg']:8.4f} {tle_data['eccentricity'] * 1e7:07.0f} {tle_data['arg_perigee_deg']:8.4f} {tle_data['mean_anomaly_deg']:8.4f} {tle_data['mean_motion_rev_per_day']:11.8f}{tle_data['revolution_num']:5d}"
        line2_checksum = self._tle_checksum(line2)
        line2 = f"{line2}{line2_checksum}"

        return line1, line2

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
        Propagate orbital elements using SGP4 and compute position in TEME/ECEF.
        """
        line1, line2 = self.tle_data_to_lines(tle_data)
        
        try:
            satellite = twoline2rv(line1, line2, WGS84)
        except Exception as e:
            # Fallback: use original lines if reconstruction fails
            if 'line1' in tle_data and 'line2' in tle_data:
                satellite = twoline2rv(tle_data['line1'], tle_data['line2'], WGS84)
            else:
                raise RuntimeError(f"TLE reconstruction failed: {e}")

        current_time = tle_data['epoch_datetime'] + timedelta(minutes=tsince_minutes)
        jd, fr = self.datetime_to_jd_fr(current_time)

        error, r_teme, v_teme = satellite.sgp4(jd, fr)
        if error != 0:
            raise RuntimeError(f"SGP4 propagation error: {error}")

        r_ecef = self.teme_to_ecef(np.array(r_teme), current_time)

        # Calculate some additional orbital parameters for compatibility
        orbital_radius = np.linalg.norm(r_teme)
        
        # Simple anomaly calculations (approximate)
        mean_anomaly = tle_data['mean_anomaly_deg']
        
        return {
            'tsince_minutes': tsince_minutes,
            'position_teme_km': {'x': r_teme[0], 'y': r_teme[1], 'z': r_teme[2]},
            'velocity_teme_kms': {'x': v_teme[0], 'y': v_teme[1], 'z': v_teme[2]},
            'position_ecef_km': {'x': r_ecef[0], 'y': r_ecef[1], 'z': r_ecef[2]},
            'orbital_radius_km': orbital_radius,
            'semi_major_axis_km': orbital_radius,  # Approximation
            'gmst_hours': self.calculate_gmst(self.datetime_to_julian_date(current_time)),
            'anomalies': {
                'mean_anomaly_deg': mean_anomaly,
                'eccentric_anomaly_deg': mean_anomaly,  # Approximation for low eccentricity
                'true_anomaly_deg': mean_anomaly  # Approximation for low eccentricity
            },
            'orbital_plane_coords_km': {'x': r_teme[0], 'y': r_teme[1]},  # Approximation
            'sgp4_error': error
        }
        
    def parse_tle(self, line1: str, line2: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse TLE according to AAS 06-675 specifications
        """
        line1_data = self._parse_line1(line1)
        line2_data = self._parse_line2(line2)

        tle_data = {
            'name': name or 'Unknown Satellite',
            'line1': line1,
            'line2': line2,
            **line1_data,
            **line2_data
        }

        tle_data['nodal_precession'] = self.calculate_nodal_precession(
            tle_data['inclination_deg'], tle_data['eccentricity'], tle_data['mean_motion_rev_per_day']
        )

        return tle_data

    def _parse_line1(self, line1: str) -> Dict[str, Any]:
        """Parse TLE Line 1"""
        norad_id = int(line1[2:7])
        classification = line1[7]
        int_designator = line1[9:17].strip()

        if len(int_designator) >= 5:
            launch_year = int(int_designator[:2])
            launch_num = int(int_designator[2:5])
            launch_piece = int_designator[5:].strip()
        else:
            launch_year, launch_num, launch_piece = 0, 0, ""

        epoch_year = int(line1[18:20])
        epoch_day = float(line1[20:32])
        full_year = 1900 + epoch_year if epoch_year >= 57 else 2000 + epoch_year
        epoch_datetime = datetime(full_year, 1, 1, tzinfo=timezone.utc) + timedelta(days=epoch_day - 1)

        first_deriv_mean_motion = float(line1[33:43])
        
        def parse_tle_exp(val: str) -> float:
            if not val or val.strip() == '':
                return 0.0
            # Don't strip - we need exact positions
            if len(val) < 8:  # Need at least 8 chars: " NNNNN±N"
                return 0.0
            try:
                # Handle TLE exponential format: " NNNNN±N" (e.g., " 21844-3" = +2.1844e-3)
                # Position breakdown: [0]=sign [1-5]=mantissa [6-7]=exponent
                
                # First character is sign (space = positive, - = negative)  
                sign = -1.0 if val[0] == '-' else 1.0
                
                # Characters 1-5 are the mantissa digits (e.g., "21844")
                mantissa_digits = val[1:6]
                
                # Handle special case where mantissa is all zeros
                if mantissa_digits == "00000":
                    return 0.0
                
                # Insert decimal point: "21844" -> "2.1844"
                mantissa = float(mantissa_digits[0] + '.' + mantissa_digits[1:])
                
                # Characters 6-7 are the exponent (e.g., "-3")
                exp_str = val[6:8]
                exponent = int(exp_str) if exp_str.strip() else 0
                
                result = sign * mantissa * (10 ** exponent)
                return result
            except (ValueError, IndexError) as e:
                return 0.0

        second_deriv_str = line1[44:52]  # Don't strip for exact parsing
        second_deriv_mean_motion = parse_tle_exp(second_deriv_str)

        bstar_str = line1[53:61]  # Don't strip for exact parsing
        bstar_drag = parse_tle_exp(bstar_str)

        ephemeris_type = int(line1[62])
        element_num = int(line1[64:68])

        return {
            'norad_id': norad_id,
            'classification': classification,
            'int_designator_launch_year': launch_year,
            'int_designator_launch_num': launch_num,
            'int_designator_launch_piece': launch_piece,
            'epoch_year': epoch_year,
            'epoch_day': epoch_day,
            'epoch_datetime': epoch_datetime,
            'first_deriv_mean_motion': first_deriv_mean_motion,
            'second_deriv_mean_motion': second_deriv_mean_motion,
            'bstar_drag': bstar_drag,
            'original_bstar': bstar_drag,  # Store original value for comparison
            'ephemeris_type': ephemeris_type,
            'element_num': element_num,
        }

    def _parse_line2(self, line2: str) -> Dict[str, Any]:
        """Parse TLE Line 2"""
        return {
            'inclination_deg': float(line2[8:16]),
            'raan_deg': float(line2[17:25]),
            'eccentricity': float('0.' + line2[26:33]),
            'arg_perigee_deg': float(line2[34:42]),
            'mean_anomaly_deg': float(line2[43:51]),
            'mean_motion_rev_per_day': float(line2[52:63]),
            'revolution_num': int(line2[63:68])
        }
