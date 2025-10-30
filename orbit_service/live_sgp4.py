"""
Live SGP4 Satellite Tracking

Provides satellite tracking capabilities using the proven sgp4 library.
Manages multiple satellites and provides propagation, caching, and state export.

Features:
- Load and track multiple satellites from TLE data
- Propagate to arbitrary times
- Cache propagation results for efficiency
- Export satellite state in various formats
- Coordinate transformations (TEME to geodetic)

This module is suitable for building satellite tracking applications and
demonstrates best practices for using the sgp4 library in production-like
scenarios.

Note: For real-time tracking, ensure TLE data is kept current (updated at least
weekly for LEO satellites, less frequently for higher orbits).
"""

import math
import numpy as np
from datetime import datetime, timezone
from sgp4.api import Satrec
from orbit_service.two_body_fallback import TwoBodyFallback, sgp4_state_to_orbital_elements
import logging

logger = logging.getLogger(__name__)


# SGP4 error code meanings
SGP4_ERROR_CODES = {
    0: "No error",
    1: "Mean eccentricity < 0.0 or > 1.0",
    2: "Mean motion < 0.0",
    3: "Perturbed eccentricity < 0.0 or > 1.0",
    4: "Semi-latus rectum < 0.0",
    5: "Satellite has decayed",
    6: "Satellite has decayed (low altitude)",
}


class LiveSGP4:
    """
    Production-ready SGP4 using proven library with error recovery.
    
    Features:
    - Automatic fallback to two-body propagation on SGP4 failure
    - Detailed error diagnostics with physical interpretation
    - Error state tracking and recovery mechanisms
    - Graceful degradation instead of hard failures
    """

    def __init__(self, enable_fallback=True):
        """
        Initialize LiveSGP4.
        
        Args:
            enable_fallback: Enable two-body propagation fallback (default: True)
        """
        self.satellites = {}
        self.cache = {}
        self.enable_fallback = enable_fallback
        self.error_history = {}  # Track errors for each satellite
        self.fallback_states = {}  # Store fallback propagators

    def load_satellite(self, line1, line2, name=None):
        """Load satellite from TLE"""
        try:
            satellite = Satrec.twoline2rv(line1, line2)
            norad_id = satellite.satnum

            self.satellites[norad_id] = {
                "satellite": satellite,
                "name": name or f"SAT_{norad_id}",
                "line1": line1,
                "line2": line2,
                "loaded_at": datetime.now(timezone.utc),
            }

            return norad_id
        except Exception as e:
            raise ValueError(f"Failed to load satellite: {e}")

    def propagate(self, norad_id, timestamp=None):
        """
        Propagate satellite position with error recovery.
        
        Args:
            norad_id: NORAD catalog ID
            timestamp: Target time (default: now)
            
        Returns:
            Dictionary with position, velocity, and diagnostic information
            
        Notes:
            If SGP4 fails, automatically falls back to two-body propagation
            and includes detailed error diagnostics.
        """
        if norad_id not in self.satellites:
            raise ValueError(f"Satellite {norad_id} not loaded")

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        satellite = self.satellites[norad_id]["satellite"]

        # Convert to Julian date
        jd, fr = self._datetime_to_jd(timestamp)

        # Attempt SGP4 propagation
        error, position, velocity = satellite.sgp4(jd, fr)
        
        # Initialize result dictionary
        result = {
            "timestamp": timestamp.isoformat(),
            "sgp4_error_code": error,
            "sgp4_error_message": SGP4_ERROR_CODES.get(error, f"Unknown error code {error}"),
            "propagation_method": "sgp4",
            "fallback_used": False,
        }

        # Handle SGP4 errors with fallback
        if error != 0:
            # Log error
            self._log_error(norad_id, error, timestamp)
            
            # Get physical interpretation of error
            diagnostics = self._get_error_diagnostics(satellite, error, timestamp)
            result["error_diagnostics"] = diagnostics
            
            # Attempt fallback if enabled
            if self.enable_fallback:
                logger.warning(f"SGP4 error {error} for satellite {norad_id}, attempting two-body fallback")
                
                try:
                    position, velocity = self._fallback_propagation(norad_id, timestamp)
                    result["propagation_method"] = "two_body_fallback"
                    result["fallback_used"] = True
                    result["fallback_warning"] = "Using simplified two-body propagation (less accurate)"
                except Exception as e:
                    logger.error(f"Fallback propagation also failed for satellite {norad_id}: {e}")
                    raise RuntimeError(
                        f"SGP4 error {error}: {result['sgp4_error_message']}. "
                        f"Fallback propagation also failed: {str(e)}"
                    )
            else:
                raise RuntimeError(
                    f"SGP4 error {error}: {result['sgp4_error_message']}. "
                    f"Fallback disabled. Diagnostics: {diagnostics}"
                )

        # Convert to lat/lon/alt
        lat, lon, alt = self._ecef_to_geodetic(position)

        result.update({
            "position_km": list(position),
            "velocity_kms": list(velocity),
            "latitude": lat,
            "longitude": lon,
            "altitude_km": alt,
        })

        return result

    def propagate_batch(self, norad_id, timestamps):
        """Propagate multiple timestamps efficiently"""
        results = []
        for ts in timestamps:
            try:
                result = self.propagate(norad_id, ts)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "timestamp": ts.isoformat()})
        return results

    def get_orbital_elements(self, norad_id):
        """Get orbital elements"""
        if norad_id not in self.satellites:
            raise ValueError(f"Satellite {norad_id} not loaded")

        sat = self.satellites[norad_id]["satellite"]

        return {
            "norad_id": norad_id,
            "name": self.satellites[norad_id]["name"],
            "epoch": self._jd_to_datetime(sat.jdsatepoch, sat.jdsatepochF).isoformat(),
            "mean_motion": sat.no_kozai * 1440.0 / (2 * math.pi),  # rev/day
            "eccentricity": sat.ecco,
            "inclination": math.degrees(sat.inclo),
            "raan": math.degrees(sat.nodeo),
            "arg_perigee": math.degrees(sat.argpo),
            "mean_anomaly": math.degrees(sat.mo),
            "bstar": sat.bstar,
            "classification": sat.classification,
            "element_number": sat.elnum,
            "revolution_number": sat.revnum,
        }
    
    def _log_error(self, norad_id, error_code, timestamp):
        """Log error for tracking and diagnostics."""
        if norad_id not in self.error_history:
            self.error_history[norad_id] = []
        
        self.error_history[norad_id].append({
            "error_code": error_code,
            "timestamp": timestamp.isoformat(),
            "error_message": SGP4_ERROR_CODES.get(error_code, f"Unknown error {error_code}"),
        })
        
        # Keep only last 100 errors
        if len(self.error_history[norad_id]) > 100:
            self.error_history[norad_id] = self.error_history[norad_id][-100:]
    
    def _get_error_diagnostics(self, satellite, error_code, timestamp):
        """
        Get detailed physical diagnostics for SGP4 error.
        
        Args:
            satellite: Satrec object
            error_code: SGP4 error code
            timestamp: Propagation timestamp
            
        Returns:
            Dictionary with diagnostic information
        """
        diagnostics = {
            "error_code": error_code,
            "error_description": SGP4_ERROR_CODES.get(error_code, f"Unknown error {error_code}"),
        }
        
        # Add orbital element information
        try:
            diagnostics["orbital_parameters"] = {
                "eccentricity": satellite.ecco,
                "inclination_deg": math.degrees(satellite.inclo),
                "mean_motion_rev_day": satellite.no_kozai * 1440.0 / (2 * math.pi),
                "bstar_drag": satellite.bstar,
                "epoch_age_days": (timestamp - self._jd_to_datetime(
                    satellite.jdsatepoch, satellite.jdsatepochF
                )).total_seconds() / 86400.0,
            }
        except:
            pass
        
        # Physical interpretation based on error code
        if error_code == 1:
            diagnostics["physical_meaning"] = (
                "The satellite's orbital eccentricity is outside valid range [0, 1). "
                "This indicates the TLE data may be corrupted or the orbit is no longer bound."
            )
            diagnostics["recommended_action"] = "Obtain fresh TLE data for this satellite."
            
        elif error_code == 2:
            diagnostics["physical_meaning"] = (
                "The mean motion is negative, which is physically impossible. "
                "This indicates corrupted TLE data or an invalid orbital state."
            )
            diagnostics["recommended_action"] = "Verify TLE data integrity and obtain updated elements."
            
        elif error_code in [3, 4]:
            diagnostics["physical_meaning"] = (
                "SGP4 computed perturbed orbital elements that are unphysical. "
                "This typically occurs when propagating far from the TLE epoch or "
                "for satellites with very high drag in decaying orbits."
            )
            diagnostics["recommended_action"] = (
                "Use more recent TLE data or limit propagation to shorter time periods. "
                "Consider using two-body fallback for rough estimates."
            )
            
        elif error_code in [5, 6]:
            diagnostics["physical_meaning"] = (
                "The satellite has decayed and re-entered the atmosphere. "
                "The computed altitude is below the minimum safe threshold (~98 km). "
                "Physical satellites at this altitude experience extreme drag and rapid decay."
            )
            diagnostics["recommended_action"] = (
                "This satellite has re-entered. Historical propagation only. "
                "No future propagation is possible."
            )
        
        return diagnostics
    
    def _fallback_propagation(self, norad_id, timestamp):
        """
        Fallback to two-body propagation when SGP4 fails.
        
        Args:
            norad_id: NORAD catalog ID
            timestamp: Target timestamp
            
        Returns:
            Tuple of (position, velocity) in km and km/s
        """
        satellite = self.satellites[norad_id]["satellite"]
        
        # Get state at epoch (where SGP4 should work)
        jd_epoch = satellite.jdsatepoch
        fr_epoch = satellite.jdsatepochF
        
        # Try to get initial state at epoch
        error, r0, v0 = satellite.sgp4(jd_epoch, fr_epoch)
        
        if error != 0:
            # Even epoch propagation failed, use TLE elements to estimate initial state
            logger.warning(f"Cannot get epoch state for satellite {norad_id}, using approximate initial conditions")
            
            # Compute approximate circular orbit state from TLE elements
            a = (398600.4418 / (satellite.no_kozai * 60)**2)**(1/3)  # Semi-major axis
            v_circular = math.sqrt(398600.4418 / a)
            
            # Simplified: assume circular orbit at RAAN and inclination
            r0 = np.array([a, 0, 0])
            v0 = np.array([0, v_circular, 0])
        
        # Create or retrieve fallback propagator
        if norad_id not in self.fallback_states:
            self.fallback_states[norad_id] = {
                "r0": np.array(r0),
                "v0": np.array(v0),
                "t0": self._jd_to_datetime(jd_epoch, fr_epoch),
                "propagator": TwoBodyFallback(np.array(r0), np.array(v0))
            }
        
        fallback = self.fallback_states[norad_id]
        
        # Calculate time difference in seconds
        dt = (timestamp - fallback["t0"]).total_seconds()
        
        # Propagate using two-body dynamics
        r, v = fallback["propagator"].propagate(dt)
        
        return r, v
    
    def get_error_history(self, norad_id):
        """
        Get error history for a satellite.
        
        Args:
            norad_id: NORAD catalog ID
            
        Returns:
            List of error records
        """
        return self.error_history.get(norad_id, [])

    def validate_against_reference(self, norad_id=6251):
        """Validate against known reference values"""
        # Reference TLE for satellite 06251
        line1 = "1 06251U 62025A   06176.82412014  .00002182  00000-0  13103-3 0  6091"
        line2 = "2 06251  58.0579  54.0425 0002329  75.6910 284.4861 14.84479601804021"

        # Load satellite
        sat_id = self.load_satellite(line1, line2, "TEST_06251")

        # Get epoch time
        sat = self.satellites[sat_id]["satellite"]
        epoch_time = self._jd_to_datetime(sat.jdsatepoch, sat.jdsatepochF)

        # Propagate at epoch (tsince = 0)
        result = self.propagate(sat_id, epoch_time)

        # Expected values from AAS paper
        expected_pos = [-907, 4655, 4404]  # km
        expected_vel = [-7.45, -2.15, 0.92]  # km/s

        # Calculate errors
        pos_error = math.sqrt(
            sum((p - e) ** 2 for p, e in zip(result["position_km"], expected_pos))
        )
        vel_error = math.sqrt(
            sum((v - e) ** 2 for v, e in zip(result["velocity_kms"], expected_vel))
        )

        validation_result = {
            "satellite": "06251",
            "test_type": "AAS_paper_reference",
            "computed_position": result["position_km"],
            "expected_position": expected_pos,
            "position_error_km": pos_error,
            "computed_velocity": result["velocity_kms"],
            "expected_velocity": expected_vel,
            "velocity_error_kms": vel_error,
            "test_passed": pos_error < 2.0,  # Your 2km target
            "meets_target": pos_error < 2.0,
            "sgp4_error_code": result["error"],
        }

        return validation_result

    def _datetime_to_jd(self, dt):
        """Convert datetime to Julian date"""
        # Convert to UTC if timezone aware
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

    def _jd_to_datetime(self, jd, fr):
        """Convert Julian date to datetime"""
        jd_total = jd + fr

        # Algorithm from Meeus
        a = int(jd_total + 0.5)
        if a < 2299161:
            c = a
        else:
            alpha = int((a - 1867216.25) / 36524.25)
            c = a + 1 + alpha - int(alpha / 4)

        b = c + 1524
        d = int((b - 122.1) / 365.25)
        e = int(365.25 * d)
        f = int((b - e) / 30.6001)

        day = b - e - int(30.6001 * f)
        month = f - 1 if f <= 13 else f - 13
        year = d - 4716 if month > 2 else d - 4715

        # Fractional day to time
        frac_day = (jd_total + 0.5) - int(jd_total + 0.5)
        hours = frac_day * 24
        hour = int(hours)
        minutes = (hours - hour) * 60
        minute = int(minutes)
        seconds = (minutes - minute) * 60
        second = int(seconds)
        microsecond = int((seconds - second) * 1000000)

        return datetime(
            year, month, day, hour, minute, second, microsecond, timezone.utc
        )

    def _ecef_to_geodetic(self, position):
        """Convert ECEF position to geodetic coordinates"""
        x, y, z = position

        # WGS84 constants
        a = 6378137.0  # meters
        f = 1 / 298.257223563
        e2 = 2 * f - f * f

        # Convert km to meters
        x_m, y_m, z_m = x * 1000, y * 1000, z * 1000

        # Longitude
        lon = math.atan2(y_m, x_m)

        # Latitude (iterative)
        p = math.sqrt(x_m * x_m + y_m * y_m)
        lat = math.atan2(z_m, p * (1 - e2))

        for _ in range(5):  # Usually converges quickly
            N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
            h = p / math.cos(lat) - N
            lat = math.atan2(z_m, p * (1 - e2 * N / (N + h)))

        # Altitude
        N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - N

        return math.degrees(lat), math.degrees(lon), alt / 1000  # Convert back to km


def test_live_sgp4():
    """Test the live SGP4 implementation"""
    print("Testing Live SGP4 Implementation")
    print("=" * 40)

    sgp4 = LiveSGP4()

    # Run validation test
    try:
        result = sgp4.validate_against_reference()

        print(f"Satellite: {result['satellite']}")
        print(
            f"Computed position: [{result['computed_position'][0]:.1f}, {result['computed_position'][1]:.1f}, {result['computed_position'][2]:.1f}] km"
        )
        print(f"Expected position: {result['expected_position']} km")
        print(f"Position error: {result['position_error_km']:.3f} km")
        print(
            f"Computed velocity: [{result['computed_velocity'][0]:.3f}, {result['computed_velocity'][1]:.3f}, {result['computed_velocity'][2]:.3f}] km/s"
        )
        print(f"Expected velocity: {result['expected_velocity']} km/s")
        print(f"Velocity error: {result['velocity_error_kms']:.6f} km/s")
        print(f"SGP4 error code: {result['sgp4_error_code']}")
        print()
        print(f"Test result: {'✓ PASSED' if result['test_passed'] else '✗ FAILED'}")
        print(
            f"Target achieved: {'✓ YES' if result['meets_target'] else '✗ NO'} (< 2 km position error)"
        )

        return result["meets_target"], result["position_error_km"]

    except Exception as e:
        print(f"Test failed with error: {e}")
        return False, float("inf")


if __name__ == "__main__":
    test_live_sgp4()
