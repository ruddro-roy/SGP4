"""
SGP4 Configuration and Constants

This module contains fallback TLE data and physical constants used throughout the project.

Constants:
    WGS-72 gravitational constants as specified by Vallado et al. (2006, AAS 06-675)
    for use with SGP4 orbital propagation.

Fallback TLE Data:
    Hardcoded ISS TLE data for demonstrations and testing when live data is unavailable.
    
    IMPORTANT: Update this TLE data periodically for accuracy.
    - Low Earth Orbit (LEO) satellites: Update weekly
    - Medium Earth Orbit (MEO): Update monthly  
    - Geostationary (GEO): Update quarterly
    
    Current TLE epoch: 2025-08-18
    Next recommended update: 2025-08-25 or later
    
    Sources for updated TLEs:
    - Space-Track.org (requires free registration)
    - CelesTrak.org (public access)

References:
    Vallado, D. A., Crawford, P., Hujsak, R., & Kelso, T. S. (2006).
    Revisiting Spacetrack Report #3. AIAA 2006-6753.
"""

from typing import Dict, Any

# WGS-72 Gravitational Constants (per Vallado et al. 2006, AAS 06-675)
EARTH_RADIUS_KM: float = 6378.135  # Earth equatorial radius (km)
GRAVITATIONAL_PARAMETER: float = 398600.8  # Earth gravitational parameter (km³/s²)
J2: float = 0.00108262998905892  # Second zonal harmonic coefficient
J3: float = -0.00000253215306  # Third zonal harmonic coefficient
J4: float = -0.00000165597  # Fourth zonal harmonic coefficient

# Fallback ISS TLE for demonstrations and testing
# Last updated: 2025-08-18
# Update recommended by: 2025-08-25
FALLBACK_ISS_TLE: Dict[str, Any] = {
    'name': 'ISS (ZARYA)',
    'norad_id': 25544,
    'line1': '1 25544U 98067A   25230.51041667  .00002182  00000-0  13103-3 0  9991',
    'line2': '2 25544  51.6416  45.1234 0002329  75.6910 284.4861 15.50000000123456',
    'epoch': '2025-08-18T12:15:00Z',
    'mean_motion': 15.5,
    'inclination': 51.6416,
    'eccentricity': 0.0002329
}
