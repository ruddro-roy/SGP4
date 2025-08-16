#!/usr/bin/env python3
"""
Orbital Propagation Demonstration
Shows the orbital propagation functionality with TEME coordinate calculation
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tle_parser import TLEParser

def fix_epoch_year(epoch_year):
    """
    Y2K fix for TLE epoch years
    Years 57-99 = 1900s (1957-1999)
    Years 00-56 = 2000s (2000-2056)
    """
    if 57 <= epoch_year <= 99:
        return 1900 + epoch_year
    elif 0 <= epoch_year <= 56:
        return 2000 + epoch_year
    else:
        raise ValueError(f"Invalid epoch year: {epoch_year}")

def sensitivity_analyzer(parser, tle_data, bstar_variations=[-10, -5, 0, 5, 10]):
    """
    Analyze sensitivity to B* drag coefficient variations
    Counters drag overprediction in high solar activity
    """
    print(f"\n⚠️  PITFALLS & SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    # Extract epoch year for Y2K demonstration
    epoch_year = tle_data['epoch_year']
    fixed_year = fix_epoch_year(epoch_year)
    
    print(f"\n📅 EPOCH HANDLING (Y2K Fix):")
    print(f"   Raw epoch year: {epoch_year:02d}")
    print(f"   Fixed year: {fixed_year}")
    print(f"   Rule: Years 57-99 → 1900s, Years 00-56 → 2000s")
    
    # B* sensitivity analysis
    original_bstar = tle_data.get('bstar_drag', 0.0)
    print(f"\n🎯 DRAG OVERPREDICTION ANALYSIS:")
    print(f"   Original B*: {original_bstar:.8f}")
    print(f"   Testing B* variations: ±10% to counter high solar activity effects")
    
    # Time points for propagation (6 hours)
    time_points = np.linspace(0, 360, 37)  # 0 to 6 hours, 10-minute intervals
    
    # Store trajectories for each B* variation
    trajectories = {}
    position_divergence = []
    nominal_trajectory = None
    
    # First pass: get nominal trajectory (0% variation)
    for variation in bstar_variations:
        # Modify B* coefficient
        modified_tle = tle_data.copy()
        modified_bstar = original_bstar * (1 + variation/100.0)
        modified_tle['bstar_drag'] = modified_bstar
        
        # Propagate orbit with modified B*
        positions = []
        for t in time_points:
            result = parser.propagate_orbit(modified_tle, t)
            pos = result['position_teme_km']
            positions.append([pos['x'], pos['y'], pos['z']])
        
        trajectories[variation] = np.array(positions)
        
        # Store nominal trajectory for comparison
        if variation == 0:
            nominal_trajectory = np.array(positions)
    
    # Second pass: calculate divergences
    for variation in bstar_variations:
        if variation != 0:
            divergence = np.linalg.norm(trajectories[variation] - nominal_trajectory, axis=1)
            max_divergence = np.max(divergence)
            position_divergence.append((variation, max_divergence))
            print(f"   B* {variation:+3d}%: Max position divergence = {max_divergence:.2f} km")
    
    # Plot 3D trajectories
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    colors = ['red', 'orange', 'blue', 'green', 'purple']
    
    for i, variation in enumerate(bstar_variations):
        traj = trajectories[variation]
        label = f"B* {variation:+d}%" if variation != 0 else "Nominal B*"
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=colors[i], label=label, linewidth=2 if variation == 0 else 1)
    
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_title('3D Orbital Trajectories\nB* Sensitivity Analysis')
    ax1.legend()
    
    # Position divergence plot
    ax2 = fig.add_subplot(222)
    if position_divergence:
        variations, divergences = zip(*position_divergence)
        ax2.bar(variations, divergences, color='skyblue', alpha=0.7)
        ax2.set_xlabel('B* Variation (%)')
        ax2.set_ylabel('Max Position Divergence (km)')
        ax2.set_title('Position Divergence vs B* Variation')
        ax2.grid(True, alpha=0.3)
    
    # Altitude vs time for different B* values
    ax3 = fig.add_subplot(223)
    earth_radius = 6378.137
    
    for i, variation in enumerate(bstar_variations):
        traj = trajectories[variation]
        altitudes = np.linalg.norm(traj, axis=1) - earth_radius
        label = f"B* {variation:+d}%" if variation != 0 else "Nominal B*"
        ax3.plot(time_points, altitudes, color=colors[i], label=label, 
                linewidth=2 if variation == 0 else 1)
    
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Altitude (km)')
    ax3.set_title('Altitude Decay Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Orbital period variation
    ax4 = fig.add_subplot(224)
    period_variations = []
    
    for variation in bstar_variations:
        modified_tle = tle_data.copy()
        modified_bstar = original_bstar * (1 + variation/100.0)
        modified_tle['bstar_drag'] = modified_bstar
        
        # Calculate period from mean motion
        period = 1440 / modified_tle['mean_motion_rev_per_day']  # minutes
        period_variations.append(period)
    
    ax4.plot(bstar_variations, period_variations, 'o-', color='red', linewidth=2)
    ax4.set_xlabel('B* Variation (%)')
    ax4.set_ylabel('Orbital Period (minutes)')
    ax4.set_title('Period Sensitivity to Drag')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bstar_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return trajectories, position_divergence

def main():
    """Demonstrate orbital propagation functionality"""
    parser = TLEParser()
    
    # ISS TLE data
    iss_line1 = "1 25544U 98067A   23259.57580000  .00012022  00000-0  21844-3 0  9995"
    iss_line2 = "2 25544  51.6416 220.9944 0004263 122.0101 312.2755 15.49541986415598"
    iss_name = "ISS (ZARYA)"
    
    print("=" * 80)
    print("ORBITAL PROPAGATION DEMONSTRATION")
    print("=" * 80)
    
    # Parse TLE
    print(f"\n📡 Parsing TLE for: {iss_name}")
    tle_data = parser.parse_tle(iss_line1, iss_line2, iss_name)
    
    print(f"   NORAD ID: {tle_data['norad_id']}")
    print(f"   Inclination: {tle_data['inclination_deg']:.4f}°")
    print(f"   RAAN: {tle_data['raan_deg']:.4f}°")
    print(f"   Eccentricity: {tle_data['eccentricity']:.6f}")
    print(f"   Arg of Perigee: {tle_data['arg_perigee_deg']:.4f}°")
    print(f"   Mean Anomaly: {tle_data['mean_anomaly_deg']:.4f}°")
    print(f"   Mean Motion: {tle_data['mean_motion_rev_per_day']:.8f} rev/day")
    
    # Propagate at different time intervals
    time_intervals = [0, 30, 60, 90, 120]  # minutes
    
    print(f"\n🚀 ORBITAL PROPAGATION RESULTS (TEME Coordinates)")
    print(f"{'Time (min)':<12} {'X (km)':<12} {'Y (km)':<12} {'Z (km)':<12} {'Radius (km)':<12} {'True Anom (°)':<12}")
    print("-" * 80)
    
    for tsince in time_intervals:
        result = parser.propagate_orbit(tle_data, tsince)
        
        pos = result['position_teme_km']
        radius = result['orbital_radius_km']
        true_anom = result['anomalies']['true_anomaly_deg']
        
        print(f"{tsince:<12.0f} {pos['x']:<12.2f} {pos['y']:<12.2f} {pos['z']:<12.2f} {radius:<12.2f} {true_anom:<12.2f}")
    
    print(f"\n🌍 ECEF COORDINATES (Earth-Centered, Earth-Fixed)")
    print(f"{'Time (min)':<12} {'X (km)':<12} {'Y (km)':<12} {'Z (km)':<12} {'GMST (hrs)':<12}")
    print("-" * 80)
    
    for tsince in time_intervals:
        result = parser.propagate_orbit(tle_data, tsince)
        
        pos_ecef = result['position_ecef_km']
        gmst = result['gmst_hours']
        
        print(f"{tsince:<12.0f} {pos_ecef['x']:<12.2f} {pos_ecef['y']:<12.2f} {pos_ecef['z']:<12.2f} {gmst:<12.2f}")
    
    # Detailed analysis at epoch
    print(f"\n📊 DETAILED ANALYSIS AT EPOCH (t=0)")
    result_epoch = parser.propagate_orbit(tle_data, 0.0)
    
    print(f"   Semi-major axis: {result_epoch['semi_major_axis_km']:.2f} km")
    print(f"   Orbital period: {1440 / tle_data['mean_motion_rev_per_day']:.2f} minutes")
    
    # Anomalies
    anomalies = result_epoch['anomalies']
    print(f"\n   Anomalies:")
    print(f"     Mean Anomaly: {anomalies['mean_anomaly_deg']:.4f}°")
    print(f"     Eccentric Anomaly: {anomalies['eccentric_anomaly_deg']:.4f}°")
    print(f"     True Anomaly: {anomalies['true_anomaly_deg']:.4f}°")
    
    # Position vectors
    pos = result_epoch['position_teme_km']
    pos_ecef = result_epoch['position_ecef_km']
    orbital_pos = result_epoch['orbital_plane_coords_km']
    
    print(f"\n   Position Vectors:")
    print(f"     TEME: [{pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}] km")
    print(f"     ECEF: [{pos_ecef['x']:.2f}, {pos_ecef['y']:.2f}, {pos_ecef['z']:.2f}] km")
    print(f"     Orbital Plane: [{orbital_pos['x']:.2f}, {orbital_pos['y']:.2f}, 0.00] km")
    print(f"     Magnitude: {result_epoch['orbital_radius_km']:.2f} km")
    print(f"     GMST at Epoch: {result_epoch['gmst_hours']:.4f} hours")
    
    # Altitude calculation
    earth_radius = 6378.137  # km
    altitude = result_epoch['orbital_radius_km'] - earth_radius
    print(f"     Altitude: {altitude:.2f} km")
    
    # Show secular effects
    print(f"\n🌍 SECULAR EFFECTS:")
    print(f"   Nodal Precession: {tle_data['nodal_precession']:.8f}°/day")
    
    # Demonstrate coordinate transformation
    print(f"\n🔄 COORDINATE TRANSFORMATION VERIFICATION:")
    print(f"   The position vector is computed using:")
    print(f"   1. Solve Kepler's equation: M = E - e·sin(E)")
    print(f"   2. Calculate true anomaly: f = 2·atan(√((1+e)/(1-e))·tan(E/2))")
    print(f"   3. Compute orbital position: r = a(1-e²)/(1+e·cos(f))")
    print(f"   4. Transform to TEME: R = Rz(-Ω)·Rx(-i)·Rz(-ω)·[x,y,0]ᵀ")
    print(f"   5. Convert to ECEF: R = [[cosθ, sinθ, 0], [-sinθ, cosθ, 0], [0,0,1]]·TEME")
    print(f"      where θ = GMST = 18.697374558 + 0.06570982441908·d + 1.00273790935·h + 0.000026·t²")
    
    print(f"\n✅ Propagation complete! All calculations include:")
    print(f"   • Precise Kepler equation solving")
    print(f"   • TEME coordinate transformation")
    print(f"   • ECEF conversion using Greenwich Mean Sidereal Time")
    
    # Run sensitivity analysis for pitfalls demonstration
    trajectories, divergence = sensitivity_analyzer(parser, tle_data)
    
    print(f"\n🔍 PITFALLS SUMMARY:")
    print(f"   • Epoch Y2K handling: Critical for dates spanning 1957-2056")
    print(f"   • Drag overprediction: B* variations cause significant position errors")
    print(f"   • High solar activity: Increases atmospheric density, amplifying drag effects")
    print(f"   • Sensitivity analysis: ±10% B* variation shows trajectory divergence")
    print(f"   • Mitigation: Monte Carlo B* sampling for uncertainty quantification")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
