"""
TLE Parser Module

Provides utilities for parsing Two-Line Element (TLE) sets, extracting orbital
parameters, and propagating satellite orbits using the SGP4 model.

This module serves as a high-level interface to the sgp4 library with added
functionality for TLE manipulation and coordinate transformations.
"""

import math
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, Any
from sgp4.api import Satrec


class TLEParser:
    """
    Parser and utilities for Two-Line Element (TLE) sets.
    
    Provides methods for:
    - Parsing TLE data into structured format
    - Reconstructing TLE strings from parsed data
    - Orbital propagation using SGP4
    - Coordinate transformations (TEME to ECEF/Geodetic)
    """
    
    def __init__(self):
        """Initialize TLE parser."""
        pass
    
    def parse_tle(self, line1: str, line2: str, name: str = "") -> Dict[str, Any]:
        """
        Parse TLE lines into structured data.
        
        Args:
            line1: First line of TLE
            line2: Second line of TLE
            name: Optional satellite name
            
        Returns:
            Dictionary containing parsed TLE data
        """
        # Parse using sgp4 library
        satellite = Satrec.twoline2rv(line1, line2)
        
        # Convert mean motion from rad/min to rev/day
        mean_motion_rev_day = satellite.no_kozai * 1440.0 / (2.0 * math.pi)
        
        # Calculate epoch datetime
        epoch_year = satellite.epochyr
        epoch_days = satellite.epochdays
        
        year = 1900 + epoch_year if epoch_year >= 57 else 2000 + epoch_year
        epoch_datetime = self._epoch_to_datetime(year, epoch_days)
        
        return {
            "name": name,
            "norad_id": satellite.satnum,
            "classification": getattr(satellite, 'classification', 'U'),
            "epoch_year": epoch_year,
            "epoch_days": epoch_days,
            "epoch_datetime": epoch_datetime,
            "ndot": satellite.ndot,
            "nddot": satellite.nddot,
            "bstar_drag": satellite.bstar,
            "element_number": getattr(satellite, 'elnum', 0),
            "inclination_deg": math.degrees(satellite.inclo),
            "raan_deg": math.degrees(satellite.nodeo),
            "eccentricity": satellite.ecco,
            "arg_perigee_deg": math.degrees(satellite.argpo),
            "mean_anomaly_deg": math.degrees(satellite.mo),
            "mean_motion_rev_per_day": mean_motion_rev_day,
            "revolution_number": getattr(satellite, 'revnum', 0),
            "line1": line1,
            "line2": line2,
        }
    
    def tle_data_to_lines(self, tle_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Reconstruct TLE lines from parsed data.
        
        Args:
            tle_data: Dictionary containing TLE parameters
            
        Returns:
            Tuple of (line1, line2) strings
        """
        # Use original lines if available and no modifications
        if "line1" in tle_data and "line2" in tle_data:
            return tle_data["line1"], tle_data["line2"]
        
        # Otherwise construct new TLE lines
        norad_id = tle_data["norad_id"]
        classification = tle_data.get("classification", "U")
        epoch_year = tle_data["epoch_year"]
        epoch_days = tle_data["epoch_days"]
        ndot = tle_data.get("ndot", 0.0)
        nddot = tle_data.get("nddot", 0.0)
        bstar = tle_data["bstar_drag"]
        element_num = tle_data.get("element_number", 0)
        
        incl = tle_data["inclination_deg"]
        raan = tle_data["raan_deg"]
        ecc = tle_data["eccentricity"]
        argp = tle_data["arg_perigee_deg"]
        mean_anom = tle_data["mean_anomaly_deg"]
        mean_motion = tle_data["mean_motion_rev_per_day"]
        rev_num = tle_data.get("revolution_number", 0)
        
        # Format line 1
        line1 = f"1 {norad_id:05d}{classification} "
        line1 += f"{epoch_year:02d}{epoch_days:012.8f} "
        line1 += f"{ndot:10.8f} "
        line1 += self._format_exponential(nddot)
        line1 += " "
        line1 += self._format_exponential(bstar)
        line1 += f" 0 {element_num:4d}"
        line1 = line1[:68] + str(self._checksum(line1))
        
        # Format line 2
        ecc_str = f"{int(ecc * 10000000):07d}"
        line2 = f"2 {norad_id:05d} "
        line2 += f"{incl:8.4f} "
        line2 += f"{raan:8.4f} "
        line2 += ecc_str + " "
        line2 += f"{argp:8.4f} "
        line2 += f"{mean_anom:8.4f} "
        line2 += f"{mean_motion:11.8f}"
        line2 += f"{rev_num:5d}"
        line2 = line2[:68] + str(self._checksum(line2))
        
        return line1, line2
    
    def propagate_orbit(self, tle_data: Dict[str, Any], tsince_minutes: float) -> Dict[str, Any]:
        """
        Propagate orbit using SGP4.
        
        Args:
            tle_data: Parsed TLE data dictionary
            tsince_minutes: Time since epoch in minutes
            
        Returns:
            Dictionary with position, velocity, and derived quantities
        """
        # Reconstruct TLE lines
        line1, line2 = self.tle_data_to_lines(tle_data)
        
        # Create satellite object
        satellite = Satrec.twoline2rv(line1, line2)
        
        # Propagate
        current_time = tle_data["epoch_datetime"] + timedelta(minutes=tsince_minutes)
        jd, fr = self.datetime_to_jd_fr(current_time)
        
        error, r_teme, v_teme = satellite.sgp4(jd, fr)
        
        # Convert to ECEF coordinates
        r_ecef, v_ecef = self.teme_to_ecef_precise(
            np.array(r_teme), np.array(v_teme), current_time
        )
        
        # Convert to geodetic
        lat, lon, alt = self.ecef_to_geodetic_precise(r_ecef)
        
        return {
            "timestamp": current_time.isoformat(),
            "tsince_minutes": tsince_minutes,
            "sgp4_error": error,
            "position_teme_km": {"x": r_teme[0], "y": r_teme[1], "z": r_teme[2]},
            "velocity_teme_kms": {"x": v_teme[0], "y": v_teme[1], "z": v_teme[2]},
            "position_ecef_km": {"x": r_ecef[0], "y": r_ecef[1], "z": r_ecef[2]},
            "velocity_ecef_kms": {"x": v_ecef[0], "y": v_ecef[1], "z": v_ecef[2]},
            "latitude_deg": lat,
            "longitude_deg": lon,
            "altitude_km": alt,
            "orbital_radius_km": np.linalg.norm(r_teme),
        }
    
    def datetime_to_jd_fr(self, dt: datetime) -> Tuple[float, float]:
        """
        Convert datetime to Julian date and fraction.
        
        Args:
            dt: Datetime object
            
        Returns:
            Tuple of (julian_day, fraction)
        """
        if dt.tzinfo is not None:
            dt = dt.utctimetuple()
        else:
            dt = dt.timetuple()
        
        year, month, day = dt.tm_year, dt.tm_mon, dt.tm_mday
        hour, minute, second = dt.tm_hour, dt.tm_min, dt.tm_sec
        
        # Julian day calculation
        if month <= 2:
            year -= 1
            month += 12
        
        a = int(year / 100)
        b = 2 - a + int(a / 4)
        
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        
        # Fractional part
        fr = (hour + minute / 60.0 + second / 3600.0) / 24.0
        
        return jd, fr
    
    def epoch_to_datetime(self, epoch_year: int, epoch_days: float) -> datetime:
        """
        Convert TLE epoch to datetime.
        
        Args:
            epoch_year: Two-digit year
            epoch_days: Day of year with fractional part
            
        Returns:
            Datetime object in UTC
        """
        year = 1900 + epoch_year if epoch_year >= 57 else 2000 + epoch_year
        return self._epoch_to_datetime(year, epoch_days)
    
    def _epoch_to_datetime(self, year: int, days: float) -> datetime:
        """Convert year and fractional days to datetime."""
        # Start of year
        dt = datetime(year, 1, 1, tzinfo=timezone.utc)
        
        # Add days (accounting for fractional part)
        dt += timedelta(days=days - 1.0)  # -1 because day 1 is Jan 1
        
        return dt
    
    def teme_to_ecef_precise(self, r_teme: np.ndarray, v_teme: np.ndarray, 
                             epoch_datetime: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precise TEME to ECEF transformation with proper Earth rotation.
        
        Args:
            r_teme: Position vector in TEME coordinates [x, y, z] (km)
            v_teme: Velocity vector in TEME coordinates [vx, vy, vz] (km/s)
            epoch_datetime: Epoch datetime for transformation
            
        Returns:
            Tuple of (r_ecef, v_ecef) in ECEF coordinates
        """
        # Calculate Julian centuries from J2000
        jd, fr = self.datetime_to_jd_fr(epoch_datetime)
        T = (jd - 2451545.0 + fr) / 36525.0
        
        # GMST with higher order terms (IAU 2000B)
        gmst_sec = (
            67310.54841 +
            (876600.0 * 3600.0 + 8640184.812866) * T +
            0.093104 * T * T -
            6.2e-6 * T * T * T
        )
        
        # Convert to radians and normalize
        gmst_rad = (gmst_sec % 86400.0) * (2.0 * math.pi / 86400.0)
        
        # Include equation of equinoxes for better accuracy
        omega = 125.04452 - 1934.136261 * T
        delta_psi = -0.000319 * math.sin(math.radians(omega))
        eqeq = delta_psi * math.cos(math.radians(23.4393))
        
        # Greenwich Apparent Sidereal Time
        gast = gmst_rad + eqeq
        
        # Rotation matrix from TEME to ECEF
        cos_gast = math.cos(gast)
        sin_gast = math.sin(gast)
        
        # Transform position
        r_ecef = np.array([
            cos_gast * r_teme[0] + sin_gast * r_teme[1],
            -sin_gast * r_teme[0] + cos_gast * r_teme[1],
            r_teme[2]
        ])
        
        # Transform velocity (includes Earth rotation rate)
        omega_earth = 7.2921159e-5  # rad/s
        
        v_ecef = np.array([
            cos_gast * v_teme[0] + sin_gast * v_teme[1] + omega_earth * r_ecef[1],
            -sin_gast * v_teme[0] + cos_gast * v_teme[1] - omega_earth * r_ecef[0],
            v_teme[2]
        ])
        
        return r_ecef, v_ecef
    
    def ecef_to_geodetic_precise(self, r_ecef: np.ndarray) -> Tuple[float, float, float]:
        """
        Accurate ECEF to geodetic conversion using Bowring's method.
        
        Args:
            r_ecef: Position vector in ECEF coordinates [x, y, z] (km)
            
        Returns:
            Tuple of (latitude_deg, longitude_deg, altitude_km)
        """
        # WGS84 parameters
        a = 6378.137  # km
        f = 1.0 / 298.257223563
        b = a * (1.0 - f)
        e2 = 2.0 * f - f * f
        ep2 = e2 / (1.0 - e2)
        
        x, y, z = r_ecef
        
        # Longitude
        lon = math.atan2(y, x)
        
        # Distance from z-axis
        p = math.sqrt(x * x + y * y)
        
        # Handle pole cases
        if p < 1e-10:
            lat = math.pi / 2.0 if z > 0 else -math.pi / 2.0
            alt = abs(z) - b
            return math.degrees(lat), math.degrees(lon), alt
        
        # Initial estimate using Bowring's formula
        theta = math.atan2(z * a, p * b)
        
        # Iterate for latitude (usually converges in 2-3 iterations)
        for _ in range(5):
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            
            lat = math.atan2(
                z + ep2 * b * sin_theta * sin_theta * sin_theta,
                p - e2 * a * cos_theta * cos_theta * cos_theta
            )
            
            # Update theta for next iteration
            sin_lat = math.sin(lat)
            N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
            
            # Check convergence
            new_theta = math.atan2(z + e2 * N * sin_lat, p)
            if abs(new_theta - theta) < 1e-12:
                break
            theta = new_theta
        
        # Calculate altitude
        cos_lat = math.cos(lat)
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        
        if cos_lat > 1e-10:
            alt = p / cos_lat - N
        else:
            alt = z / sin_lat - N * (1.0 - e2)
        
        return math.degrees(lat), math.degrees(lon), alt
    
    def _format_exponential(self, value: float) -> str:
        """Format a number in TLE exponential notation."""
        if value == 0.0:
            return " 00000-0"
        
        sign = "-" if value < 0 else " "
        abs_val = abs(value)
        
        # Find exponent
        if abs_val == 0:
            return " 00000-0"
        
        exp = int(math.floor(math.log10(abs_val)))
        mantissa = abs_val / (10 ** exp)
        
        # Format mantissa (5 digits)
        mant_str = f"{int(mantissa * 100000):05d}"
        
        # Format exponent
        exp_sign = "-" if exp < 0 else "+"
        exp_str = f"{abs(exp):d}"
        
        return f"{sign}{mant_str}{exp_sign}{exp_str}"
    
    def _checksum(self, line: str) -> int:
        """Calculate TLE checksum."""
        checksum = 0
        for char in line[:68]:
            if char.isdigit():
                checksum += int(char)
            elif char == "-":
                checksum += 1
        return checksum % 10


if __name__ == "__main__":
    # Simple test
    parser = TLEParser()
    
    line1 = "1 25544U 98067A   23259.57580000  .00012022  00000-0  21844-3 0  9995"
    line2 = "2 25544  51.6416 220.9944 0004263 122.0101 312.2755 15.49541986415598"
    
    tle_data = parser.parse_tle(line1, line2, "ISS")
    print(f"Parsed TLE for {tle_data['name']}")
    print(f"NORAD ID: {tle_data['norad_id']}")
    print(f"Inclination: {tle_data['inclination_deg']:.4f}°")
    print(f"Mean Motion: {tle_data['mean_motion_rev_per_day']:.8f} rev/day")
    
    # Test propagation
    result = parser.propagate_orbit(tle_data, 0.0)
    print(f"\nPosition at epoch:")
    print(f"  TEME: {result['position_teme_km']}")
    print(f"  Lat/Lon/Alt: {result['latitude_deg']:.2f}°, {result['longitude_deg']:.2f}°, {result['altitude_km']:.1f} km")
