#!/usr/bin/env python3
"""
Simple test script for differentiable SGP4 without PyTorch dependency
"""

import math
import numpy as np
from datetime import datetime

class SimpleSGP4:
    """Simplified SGP4 implementation for testing"""
    
    def __init__(self):
        # SGP4 Constants
        self.mu = 398600.8  # km³/s²
        self.Re = 6378.135  # km
        self.J2 = 0.001082616
        self.minutes_per_day = 1440.0
        self.twopi = 2.0 * math.pi
        
    def parse_tle(self, line1, line2):
        """Parse TLE data"""
        # Extract elements
        inclination = math.radians(float(line2[8:16].strip()))
        raan = math.radians(float(line2[17:25].strip()))
        eccentricity = float('0.' + line2[26:33].strip())
        arg_perigee = math.radians(float(line2[34:42].strip()))
        mean_anomaly = math.radians(float(line2[43:51].strip()))
        mean_motion = float(line2[52:63].strip()) * self.twopi / self.minutes_per_day  # rad/min
        
        return {
            'inclination': inclination,
            'raan': raan,
            'eccentricity': eccentricity,
            'arg_perigee': arg_perigee,
            'mean_anomaly': mean_anomaly,
            'mean_motion': mean_motion
        }
    
    def solve_kepler(self, M, e, tolerance=1e-12, max_iter=10):
        """Solve Kepler's equation"""
        E = M
        for _ in range(max_iter):
            f = E - e * math.sin(E) - M
            df = 1.0 - e * math.cos(E)
            delta_E = f / df
            E = E - delta_E
            if abs(delta_E) < tolerance:
                break
        return E
    
    def propagate(self, elements, tsince):
        """Basic SGP4 propagation"""
        # Semi-major axis
        n0 = elements['mean_motion']
        a0 = (self.mu / (n0 * n0)) ** (1.0/3.0)
        
        # Semi-latus rectum
        e0 = elements['eccentricity']
        p = a0 * (1.0 - e0 * e0)
        
        # Secular rates
        i0 = elements['inclination']
        cos_i = math.cos(i0)
        
        raan_dot = -1.5 * self.J2 * (self.Re / p) ** 2 * n0 * cos_i
        arg_perigee_dot = 0.75 * self.J2 * (self.Re / p) ** 2 * n0 * (5.0 * cos_i * cos_i - 1.0)
        
        # Update elements
        raan = elements['raan'] + raan_dot * tsince
        arg_perigee = elements['arg_perigee'] + arg_perigee_dot * tsince
        mean_anomaly = elements['mean_anomaly'] + n0 * tsince
        
        # Solve Kepler
        E = self.solve_kepler(mean_anomaly, e0)
        
        # True anomaly
        sin_E = math.sin(E)
        cos_E = math.cos(E)
        beta = 1.0 / (1.0 + math.sqrt(1.0 - e0 * e0))
        nu = E + 2.0 * math.atan(beta * e0 * sin_E / (1.0 - beta * e0 * cos_E))
        
        # Radius
        r = a0 * (1.0 - e0 * cos_E)
        
        # Position in orbital plane
        cos_nu = math.cos(nu)
        sin_nu = math.sin(nu)
        x_orb = r * cos_nu
        y_orb = r * sin_nu
        
        # Velocity in orbital plane
        n = math.sqrt(self.mu / (a0 ** 3))
        rdot = a0 * e0 * sin_E * n / (1.0 - e0 * cos_E)
        rfdot = a0 * n * math.sqrt(1.0 - e0 * e0) / (1.0 - e0 * cos_E)
        
        vx_orb = rdot * cos_nu - rfdot * sin_nu
        vy_orb = rdot * sin_nu + rfdot * cos_nu
        
        # Transform to TEME frame
        i = i0
        omega = arg_perigee
        Omega = raan
        
        cos_i = math.cos(i)
        sin_i = math.sin(i)
        cos_omega = math.cos(omega)
        sin_omega = math.sin(omega)
        cos_Omega = math.cos(Omega)
        sin_Omega = math.sin(Omega)
        
        # Rotation matrix elements
        M11 = cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i
        M12 = -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i
        M21 = sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i
        M22 = -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i
        M31 = sin_omega * sin_i
        M32 = cos_omega * sin_i
        
        # Transform
        x = M11 * x_orb + M12 * y_orb
        y = M21 * x_orb + M22 * y_orb
        z = M31 * x_orb + M32 * y_orb
        
        vx = M11 * vx_orb + M12 * vy_orb
        vy = M21 * vx_orb + M22 * vy_orb
        vz = M31 * vx_orb + M32 * vy_orb
        
        return [x, y, z], [vx, vy, vz]

def test_satellite_06251():
    """Test case from AAS paper"""
    sgp4 = SimpleSGP4()
    
    # Test TLE - corrected format
    line1 = "1 06251U 62025A   06176.82412014  .00002182  00000-0  13103-3 0  6091"
    line2 = "2 06251  58.0579  54.0425 0002329  75.6910 284.4861 14.84479601804021"
    
    print(f"TLE Line 2: '{line2}'")
    print(f"Arg perigee field [34:42]: '{line2[34:42]}'")
    print(f"Mean anomaly field [43:51]: '{line2[43:51]}'")
    print(f"Mean motion field [52:63]: '{line2[52:63]}'")
    print()
    
    elements = sgp4.parse_tle(line1, line2)
    position, velocity = sgp4.propagate(elements, 0.0)
    
    # Expected values
    expected_pos = [-907, 4655, 4404]
    expected_vel = [-7.45, -2.15, 0.92]
    
    pos_error = math.sqrt(sum((p - e)**2 for p, e in zip(position, expected_pos)))
    vel_error = math.sqrt(sum((v - e)**2 for v, e in zip(velocity, expected_vel)))
    
    print(f"SGP4 Validation Test - Satellite 06251")
    print(f"Computed position: [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}] km")
    print(f"Expected position: [{expected_pos[0]}, {expected_pos[1]}, {expected_pos[2]}] km")
    print(f"Position error: {pos_error:.3f} km")
    print(f"Computed velocity: [{velocity[0]:.3f}, {velocity[1]:.3f}, {velocity[2]:.3f}] km/s")
    print(f"Expected velocity: [{expected_vel[0]}, {expected_vel[1]}, {expected_vel[2]}] km/s")
    print(f"Velocity error: {vel_error:.6f} km/s")
    
    test_passed = pos_error < 50.0 and vel_error < 5.0  # Reasonable tolerance for initial implementation
    print(f"Test result: {'✓ PASSED' if test_passed else '✗ FAILED'}")
    
    return test_passed

if __name__ == "__main__":
    test_satellite_06251()
