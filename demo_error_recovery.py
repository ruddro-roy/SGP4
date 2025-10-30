"""
Error Recovery Demonstration

This script demonstrates the error recovery and fallback mechanisms
implemented in the SGP4 propagation system.

Showcases:
1. Normal propagation with error handling enabled
2. Automatic fallback to two-body propagation on SGP4 failure
3. Detailed error diagnostics with physical interpretation
4. Error history tracking
5. Graceful degradation instead of crashes

Usage:
    python demo_error_recovery.py
"""

import sys
from datetime import datetime, timedelta, timezone
import numpy as np

from orbit_service.live_sgp4 import LiveSGP4
from orbit_service.two_body_fallback import TwoBodyFallback


# Sample TLE data (September 2023) - For demonstration purposes only
# Real applications should use current TLE data from reliable sources
ISS_LINE1 = "1 25544U 98067A   23259.57580000  .00012022  00000-0  21844-3 0  9995"
ISS_LINE2 = "2 25544  51.6416 220.9944 0004263 122.0101 312.2755 15.49541986415598"

# High-drag satellite TLE for testing error conditions
DECAY_LINE1 = "1 44444U 19999A   23259.50000000  .10000000  00000-0  50000-2 0  9999"
DECAY_LINE2 = "2 44444  51.6400 100.0000 0005000  90.0000 270.0000 16.50000000 99999"

# Display configuration constants
MAX_HISTORY_DISPLAY = 5  # Maximum number of error history entries to show
TIMESTAMP_WIDTH = 20      # Width for timestamp column
MESSAGE_WIDTH = 30        # Width for message column


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_normal_propagation():
    """Demonstrate normal propagation with error handling."""
    print_section("1. Normal Propagation with Error Handling")
    
    # Create SGP4 instance with fallback enabled
    sgp4 = LiveSGP4(enable_fallback=True)
    norad_id = sgp4.load_satellite(ISS_LINE1, ISS_LINE2, "ISS")
    
    print(f"\nLoaded satellite: ISS (NORAD ID: {norad_id})")
    print("Fallback mechanism: ENABLED")
    
    # Propagate at epoch (should succeed)
    timestamp = datetime(2023, 9, 16, 13, 49, 0, tzinfo=timezone.utc)
    result = sgp4.propagate(norad_id, timestamp)
    
    print(f"\nPropagation at epoch:")
    print(f"  Status: ✓ SUCCESS")
    print(f"  Method: {result['propagation_method']}")
    print(f"  Error code: {result['sgp4_error_code']}")
    print(f"  Position: [{result['position_km'][0]:.1f}, {result['position_km'][1]:.1f}, {result['position_km'][2]:.1f}] km")
    print(f"  Velocity: [{result['velocity_kms'][0]:.3f}, {result['velocity_kms'][1]:.3f}, {result['velocity_kms'][2]:.3f}] km/s")
    print(f"  Latitude: {result['latitude']:.2f}°")
    print(f"  Longitude: {result['longitude']:.2f}°")
    print(f"  Altitude: {result['altitude_km']:.1f} km")


def demo_error_diagnostics():
    """Demonstrate detailed error diagnostics."""
    print_section("2. Detailed Error Diagnostics")
    
    sgp4 = LiveSGP4(enable_fallback=False)  # Disable fallback to see errors
    norad_id = sgp4.load_satellite(DECAY_LINE1, DECAY_LINE2, "DECAY_TEST")
    
    print(f"\nLoaded high-drag satellite: DECAY_TEST")
    print("Fallback mechanism: DISABLED (to demonstrate error diagnostics)")
    
    # Try to propagate far into future (will likely fail)
    future_time = datetime(2023, 9, 16, tzinfo=timezone.utc) + timedelta(days=365)
    
    print(f"\nAttempting propagation 1 year into future...")
    
    try:
        result = sgp4.propagate(norad_id, future_time)
        
        if result['sgp4_error_code'] != 0:
            print(f"\n  Status: ⚠ SGP4 ERROR (but recovered)")
            print(f"  Error code: {result['sgp4_error_code']}")
            print(f"  Error message: {result['sgp4_error_message']}")
            
            if 'error_diagnostics' in result:
                diag = result['error_diagnostics']
                print(f"\n  Error Diagnostics:")
                print(f"    Description: {diag['error_description']}")
                
                if 'physical_meaning' in diag:
                    print(f"\n    Physical Meaning:")
                    for line in diag['physical_meaning'].split('. '):
                        if line.strip():
                            print(f"      {line.strip()}.")
                
                if 'recommended_action' in diag:
                    print(f"\n    Recommended Action:")
                    for line in diag['recommended_action'].split('. '):
                        if line.strip():
                            print(f"      {line.strip()}.")
        else:
            print(f"  Status: ✓ SUCCESS (no error)")
    
    except RuntimeError as e:
        print(f"\n  Status: ✗ FAILED")
        print(f"  Exception: {str(e)[:200]}")


def demo_fallback_mechanism():
    """Demonstrate automatic fallback to two-body propagation."""
    print_section("3. Automatic Fallback to Two-Body Propagation")
    
    sgp4 = LiveSGP4(enable_fallback=True)  # Enable fallback
    norad_id = sgp4.load_satellite(DECAY_LINE1, DECAY_LINE2, "DECAY_FALLBACK")
    
    print(f"\nLoaded high-drag satellite: DECAY_FALLBACK")
    print("Fallback mechanism: ENABLED")
    
    # Try various propagation times
    times = [0, 30, 90, 180, 365]  # days
    
    print(f"\nPropagating to multiple future times:")
    print(f"{'Days':>6} | {'Method':^20} | {'Status':^10} | {'Altitude (km)':>13}")
    print("-" * 60)
    
    base_time = datetime(2023, 9, 16, tzinfo=timezone.utc)
    
    for days in times:
        target_time = base_time + timedelta(days=days)
        
        try:
            result = sgp4.propagate(norad_id, target_time)
            
            method = result['propagation_method']
            status = "SUCCESS" if result['sgp4_error_code'] == 0 else "FALLBACK"
            alt = result.get('altitude_km', 0)
            
            fallback_marker = " *" if result.get('fallback_used', False) else ""
            
            print(f"{days:6d} | {method:^20} | {status:^10} | {alt:13.1f}{fallback_marker}")
        
        except Exception as e:
            print(f"{days:6d} | {'N/A':^20} | {'FAILED':^10} | {'N/A':>13}")
    
    print("\n* Indicates fallback was used")


def demo_error_history():
    """Demonstrate error history tracking."""
    print_section("4. Error History Tracking")
    
    sgp4 = LiveSGP4(enable_fallback=True)
    norad_id = sgp4.load_satellite(DECAY_LINE1, DECAY_LINE2, "ERROR_TRACK")
    
    print(f"\nLoaded satellite: ERROR_TRACK")
    
    # Propagate to many times (some may fail)
    base_time = datetime(2023, 9, 16, tzinfo=timezone.utc)
    times = [0, 30, 60, 90, 120, 180, 270, 365]
    
    print(f"\nPropagating to {len(times)} different times...")
    
    for days in times:
        try:
            sgp4.propagate(norad_id, base_time + timedelta(days=days))
        except:
            pass
    
    # Get error history
    history = sgp4.get_error_history(norad_id)
    
    if history:
        print(f"\nError History ({len(history)} errors recorded):")
        print(f"{'Timestamp':^{TIMESTAMP_WIDTH}} | {'Error Code':^11} | {'Message'}")
        print("-" * 70)
        
        for error in history[:MAX_HISTORY_DISPLAY]:  # Show first N errors
            timestamp = error['timestamp'][:TIMESTAMP_WIDTH-1]  # Truncate timestamp
            code = error['error_code']
            message = error['error_message'][:MESSAGE_WIDTH]  # Truncate message
            
            print(f"{timestamp:^{TIMESTAMP_WIDTH}} | {code:^11d} | {message}")
        
        if len(history) > MAX_HISTORY_DISPLAY:
            print(f"... and {len(history) - MAX_HISTORY_DISPLAY} more errors")
    else:
        print(f"\nNo errors recorded (all propagations successful)")


def demo_two_body_fallback():
    """Demonstrate two-body fallback directly."""
    print_section("5. Two-Body Propagation Fallback")
    
    # Create initial state (ISS-like orbit)
    r0 = np.array([6778.0, 0.0, 0.0])  # km
    v0 = np.array([0.0, 7.669, 0.0])  # km/s
    
    print(f"\nInitial State:")
    print(f"  Position: [{r0[0]:.1f}, {r0[1]:.1f}, {r0[2]:.1f}] km")
    print(f"  Velocity: [{v0[0]:.3f}, {v0[1]:.3f}, {v0[2]:.3f}] km/s")
    
    fallback = TwoBodyFallback(r0, v0)
    elements = fallback.get_elements()
    
    print(f"\nOrbital Elements:")
    print(f"  Semi-major axis: {elements['a']:.1f} km")
    print(f"  Eccentricity: {elements['e']:.6f}")
    print(f"  Inclination: {np.degrees(elements['i']):.2f}°")
    print(f"  Period: {2*np.pi/elements['n']/60:.2f} minutes")
    
    # Propagate for various times
    times = [0, 1800, 3600, 5400, 7200]  # seconds
    
    print(f"\nPropagation Results:")
    print(f"{'Time (min)':>11} | {'Radius (km)':>12} | {'Speed (km/s)':>13}")
    print("-" * 40)
    
    for t in times:
        r, v = fallback.propagate(t)
        radius = np.linalg.norm(r)
        speed = np.linalg.norm(v)
        
        print(f"{t/60:11.1f} | {radius:12.1f} | {speed:13.3f}")
    
    print(f"\nNote: Two-body propagation conserves energy and angular momentum.")
    print(f"      It's less accurate than SGP4 but provides a reasonable fallback.")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  SGP4 ERROR RECOVERY AND FALLBACK DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how the SGP4 system gracefully handles errors")
    print("instead of crashing, providing useful diagnostics and fallback options.")
    
    try:
        demo_normal_propagation()
        demo_error_diagnostics()
        demo_fallback_mechanism()
        demo_error_history()
        demo_two_body_fallback()
        
        print_section("Summary")
        print("""
Key Features Demonstrated:

1. ✓ Normal propagation with error monitoring
   - Transparent error codes and status
   - No crashes on valid operations

2. ✓ Detailed error diagnostics
   - Physical interpretation of errors
   - Recommended actions for recovery
   - Orbital parameter context

3. ✓ Automatic fallback mechanism
   - Two-body propagation as backup
   - Graceful degradation when SGP4 fails
   - Clear indication when fallback is used

4. ✓ Error history tracking
   - Complete record of all errors
   - Useful for debugging and analysis
   - Helps identify problematic propagation times

5. ✓ Two-body propagation
   - Simple Keplerian dynamics
   - Energy and momentum conservation
   - Reliable fallback option

All demonstrations completed successfully!
        """)
    
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
