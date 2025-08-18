"""
Working SGP4 Implementation - Direct approach using sgp4 library
"""

from sgp4.api import Satrec
from sgp4 import exporter
from datetime import datetime, timezone
import math

def test_sgp4_direct():
    """Direct test using sgp4 library"""
    
    # Test TLE for satellite 06251
    line1 = "1 06251U 62025A   06176.82412014  .00002182  00000-0  13103-3 0  6091"
    line2 = "2 06251  58.0579  54.0425 0002329  75.6910 284.4861 14.84479601804021"
    
    # Create satellite
    satellite = Satrec.twoline2rv(line1, line2)
    
    print(f"Satellite loaded: {satellite.satnum}")
    print(f"Epoch: {satellite.epochyr}/{satellite.epochdays}")
    print(f"Error code: {satellite.error}")
    
    # Propagate at epoch (tsince = 0 minutes)
    jd = satellite.jdsatepoch
    fr = satellite.jdsatepochF
    
    print(f"Julian date: {jd} + {fr}")
    
    # Propagate
    error, position, velocity = satellite.sgp4(jd, fr)
    
    print(f"SGP4 error: {error}")
    print(f"Position: [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}] km")
    print(f"Velocity: [{velocity[0]:.3f}, {velocity[1]:.3f}, {velocity[2]:.3f}] km/s")
    
    # Expected values
    expected_pos = [-907, 4655, 4404]
    expected_vel = [-7.45, -2.15, 0.92]
    
    pos_error = math.sqrt(sum((p - e)**2 for p, e in zip(position, expected_pos)))
    vel_error = math.sqrt(sum((v - e)**2 for v, e in zip(velocity, expected_vel)))
    
    print(f"Expected position: {expected_pos} km")
    print(f"Expected velocity: {expected_vel} km/s")
    print(f"Position error: {pos_error:.3f} km")
    print(f"Velocity error: {vel_error:.6f} km/s")
    
    target_met = pos_error < 2.0
    print(f"Target achieved: {'✓ YES' if target_met else '✗ NO'} (< 2 km)")
    
    return target_met, pos_error

def create_enhanced_sgp4_service():
    """Create enhanced SGP4 service for integration"""
    
    class EnhancedSGP4Service:
        def __init__(self):
            self.satellites = {}
            
        def load_tle(self, line1, line2, name=None):
            """Load TLE and return satellite ID"""
            satellite = Satrec.twoline2rv(line1, line2)
            sat_id = satellite.satnum
            
            self.satellites[sat_id] = {
                'satellite': satellite,
                'name': name or f"SAT_{sat_id}",
                'line1': line1,
                'line2': line2
            }
            
            return sat_id
            
        def propagate_at_time(self, sat_id, jd, fr):
            """Propagate satellite at specific Julian date"""
            if sat_id not in self.satellites:
                raise ValueError(f"Satellite {sat_id} not loaded")
                
            satellite = self.satellites[sat_id]['satellite']
            error, position, velocity = satellite.sgp4(jd, fr)
            
            return {
                'error': error,
                'position_km': list(position),
                'velocity_kms': list(velocity),
                'satellite_id': sat_id,
                'name': self.satellites[sat_id]['name']
            }
            
        def propagate_at_epoch(self, sat_id):
            """Propagate at satellite epoch"""
            if sat_id not in self.satellites:
                raise ValueError(f"Satellite {sat_id} not loaded")
                
            satellite = self.satellites[sat_id]['satellite']
            return self.propagate_at_time(sat_id, satellite.jdsatepoch, satellite.jdsatepochF)
            
        def get_orbital_elements(self, sat_id):
            """Get orbital elements"""
            if sat_id not in self.satellites:
                raise ValueError(f"Satellite {sat_id} not loaded")
                
            sat = self.satellites[sat_id]['satellite']
            
            return {
                'norad_id': sat_id,
                'name': self.satellites[sat_id]['name'],
                'epoch_year': sat.epochyr,
                'epoch_days': sat.epochdays,
                'mean_motion_rev_per_day': sat.no_kozai * 1440.0 / (2 * math.pi),
                'eccentricity': sat.ecco,
                'inclination_deg': math.degrees(sat.inclo),
                'raan_deg': math.degrees(sat.nodeo),
                'arg_perigee_deg': math.degrees(sat.argpo),
                'mean_anomaly_deg': math.degrees(sat.mo),
                'bstar': sat.bstar
            }
            
        def validate_accuracy(self):
            """Validate against reference case"""
            # Load test satellite
            line1 = "1 06251U 62025A   06176.82412014  .00002182  00000-0  13103-3 0  6091"
            line2 = "2 06251  58.0579  54.0425 0002329  75.6910 284.4861 14.84479601804021"
            
            sat_id = self.load_tle(line1, line2, "TEST_06251")
            result = self.propagate_at_epoch(sat_id)
            
            # Expected values
            expected_pos = [-907, 4655, 4404]
            expected_vel = [-7.45, -2.15, 0.92]
            
            pos_error = math.sqrt(sum((p - e)**2 for p, e in zip(result['position_km'], expected_pos)))
            vel_error = math.sqrt(sum((v - e)**2 for v, e in zip(result['velocity_kms'], expected_vel)))
            
            return {
                'position_error_km': pos_error,
                'velocity_error_kms': vel_error,
                'meets_2km_target': pos_error < 2.0,
                'computed_position': result['position_km'],
                'expected_position': expected_pos,
                'sgp4_error': result['error']
            }
    
    return EnhancedSGP4Service()

if __name__ == "__main__":
    print("Testing Direct SGP4 Implementation")
    print("=" * 40)
    
    # Test direct approach
    success, error = test_sgp4_direct()
    
    print("\n" + "=" * 40)
    print("Testing Enhanced Service")
    
    # Test service
    service = create_enhanced_sgp4_service()
    validation = service.validate_accuracy()
    
    print(f"Service validation:")
    print(f"Position error: {validation['position_error_km']:.3f} km")
    print(f"Meets 2km target: {'✓ YES' if validation['meets_2km_target'] else '✗ NO'}")
    print(f"SGP4 error code: {validation['sgp4_error']}")
