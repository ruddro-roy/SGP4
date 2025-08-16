#!/usr/bin/env python3
"""
Test suite for TLE Parser
"""

import unittest
import sys
import os
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tle_parser import TLEParser

class TestTLEParser(unittest.TestCase):
    """Test cases for TLE parser functionality"""
    
    def setUp(self):
        """Set up test parser"""
        self.parser = TLEParser()
        
        # ISS TLE data for testing
        self.iss_line1 = "1 25544U 98067A   23259.57580000  .00012022  00000-0  21844-3 0  9995"
        self.iss_line2 = "2 25544  51.6416 220.9944 0004263 122.0101 312.2755 15.49541986415598"
        self.iss_name = "ISS (ZARYA)"
        
    def test_parse_tle_basic(self):
        """Test basic TLE parsing"""
        tle_data = self.parser.parse_tle(self.iss_line1, self.iss_line2, self.iss_name)
        
        self.assertEqual(tle_data['norad_id'], 25544)
        self.assertEqual(tle_data['name'], self.iss_name)
        self.assertAlmostEqual(tle_data['inclination_deg'], 51.6416, places=4)
        self.assertAlmostEqual(tle_data['eccentricity'], 0.0004263, places=6)
        
    def test_kepler_equation_solver(self):
        """Test Kepler equation solver"""
        # Test with known values
        mean_anomaly = 1.0  # radians
        eccentricity = 0.1
        
        eccentric_anomaly = self.parser.solve_kepler_equation(mean_anomaly, eccentricity)
        
        # Verify Kepler's equation: M = E - e*sin(E)
        calculated_mean = eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly)
        self.assertAlmostEqual(calculated_mean, mean_anomaly, places=10)
        
    def test_true_anomaly_calculation(self):
        """Test true anomaly calculation"""
        import math
        
        eccentric_anomaly = 1.0  # radians
        eccentricity = 0.1
        
        true_anomaly = self.parser.calculate_true_anomaly(eccentric_anomaly, eccentricity)
        
        # Verify the calculation is reasonable
        self.assertGreater(true_anomaly, 0)
        self.assertLess(true_anomaly, 2 * math.pi)
        
    def test_gmst_calculation(self):
        """Test GMST calculation"""
        # Test with J2000.0 epoch
        j2000_jd = 2451545.0
        gmst = self.parser.calculate_gmst(j2000_jd)
        
        # GMST at J2000.0 should be approximately 18.697 hours
        self.assertAlmostEqual(gmst, 18.697374558, places=2)
        
    def test_orbital_propagation(self):
        """Test orbital propagation"""
        tle_data = self.parser.parse_tle(self.iss_line1, self.iss_line2, self.iss_name)
        
        # Propagate at epoch (t=0)
        result = self.parser.propagate_orbit(tle_data, 0.0)
        
        self.assertIn('position_teme_km', result)
        self.assertIn('position_ecef_km', result)
        self.assertIn('anomalies', result)
        
        # Check that position vectors are reasonable
        pos_teme = result['position_teme_km']
        radius = (pos_teme['x']**2 + pos_teme['y']**2 + pos_teme['z']**2)**0.5
        
        # ISS should be at ~400km altitude (6778km from Earth center)
        self.assertGreater(radius, 6700)
        self.assertLess(radius, 6900)
        
    def test_teme_to_ecef_conversion(self):
        """Test TEME to ECEF coordinate conversion"""
        import numpy as np
        
        # Test vector
        r_teme = np.array([6778.0, 0.0, 0.0])  # km
        epoch = datetime(2023, 9, 16, 12, 0, 0, tzinfo=timezone.utc)
        
        r_ecef = self.parser.teme_to_ecef(r_teme, epoch)
        
        # ECEF vector should have same magnitude
        teme_mag = np.linalg.norm(r_teme)
        ecef_mag = np.linalg.norm(r_ecef)
        
        self.assertAlmostEqual(teme_mag, ecef_mag, places=6)
        
    def test_nodal_precession_calculation(self):
        """Test nodal precession calculation"""
        # ISS-like parameters
        inclination = 51.6416
        eccentricity = 0.0004263
        mean_motion = 15.49541986
        
        precession = self.parser.calculate_nodal_precession(inclination, eccentricity, mean_motion)
        
        # ISS nodal precession should be negative (westward)
        self.assertLess(precession, 0)
        # Should be on the order of -5 to -7 degrees per day
        self.assertGreater(precession, -10)
        self.assertLess(precession, -3)

if __name__ == '__main__':
    unittest.main()
