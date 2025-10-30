"""
Constants and Reference Data

This module provides fallback TLE data for demonstrations and testing.
The hardcoded ISS TLE should be updated periodically for accuracy.

Note: For operational use, always fetch current TLE data from authoritative
sources like Space-Track.org or CelesTrak.
"""

# Hardcoded ISS fallback TLE (as of August 2025)
# This is used as a fallback for the demo endpoint when live TLE data is unavailable.
ISS_TLE = {
    'name': 'ISS (ZARYA)',
    'norad_id': 25544,
    'line1': '1 25544U 98067A   25230.51041667  .00002182  00000-0  13103-3 0  9991',
    'line2': '2 25544  51.6416  45.1234 0002329  75.6910 284.4861 15.50000000123456',
    'epoch': '2025-08-18T12:15:00Z',
    'mean_motion': 15.5,
    'inclination': 51.6416,
    'eccentricity': 0.0002329
}
