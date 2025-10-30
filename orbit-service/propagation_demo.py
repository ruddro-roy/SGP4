#!/usr/bin/env python3
"""
Orbital Propagation Demonstration

Demonstrates SGP4 orbital propagation with:
- Position and velocity calculation in TEME coordinates
- 3D orbit visualization
- Sensitivity analysis for drag coefficient variations
- Ground track plotting

This script shows how small changes in orbital parameters (especially the B* drag
coefficient) can affect long-term propagation accuracy. The analysis uses the
proven sgp4 library for reliable results.

Educational purpose: Understanding how atmospheric drag affects satellite orbits
over time, particularly for low Earth orbit satellites.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tle_parser import TLEParser
from sgp4.io import twoline2rv
from sgp4.api import WGS84
from datetime import timedelta

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

def sensitivity_analyzer(parser, tle_data, bstar_variations=[-50, -25, -10, 0, 10, 25, 50]):
    """
    Analyze sensitivity to B* drag coefficient variations using SGP4.
    Extended analysis over multiple days to show cumulative effects.
    """
    print(f"\n⚠️  PITFALLS & SENSITIVITY ANALYSIS (with SGP4)")
    print("=" * 80)

    original_bstar = tle_data.get('bstar_drag', 0.0)
    print(f"\n🎯 DRAG SENSITIVITY ANALYSIS:")
    print(f"   Original B*: {original_bstar:.8f}")
    print(f"   Applying variations: {bstar_variations}%")
    print(f"   Analysis period: 7 days (drag effects accumulate over time)")

    # Extended time period - 7 days with 3-hour intervals to show cumulative drag effects
    time_points = np.linspace(0, 7*24*60, 57)  # 0 to 7 days, 3-hour intervals
    trajectories = {}
    position_divergence = []
    altitude_data = {}
    nominal_trajectory = None

    print(f"\n   Processing B* variations...")
    for i, variation in enumerate(bstar_variations):
        modified_tle_data = tle_data.copy()
        modified_bstar = original_bstar * (1 + variation / 100.0)
        modified_tle_data['bstar_drag'] = modified_bstar
        
        print(f"   • B* {variation:+3d}%: {modified_bstar:.8f} (vs nominal {original_bstar:.8f})")

        line1, line2 = parser.tle_data_to_lines(modified_tle_data)
        satellite = twoline2rv(line1, line2, WGS84)

        positions = []
        altitudes = []
        epoch_datetime = tle_data['epoch_datetime']
        
        for t in time_points:
            current_time = epoch_datetime + timedelta(minutes=t)
            jd, fr = parser.datetime_to_jd_fr(current_time)
            error, r, v = satellite.sgp4(jd, fr)
            if error == 0:
                positions.append(r)
                altitude = np.linalg.norm(r) - 6378.137  # Earth radius
                altitudes.append(altitude)
            else:
                positions.append([0, 0, 0])
                altitudes.append(0)
        
        trajectories[variation] = np.array(positions)
        altitude_data[variation] = np.array(altitudes)

        if variation == 0:
            nominal_trajectory = np.array(positions)
            nominal_altitudes = np.array(altitudes)

    # Calculate divergences with more detailed analysis
    print(f"\n   Position Divergence Analysis:")
    for variation in bstar_variations:
        if variation != 0:
            divergence = np.linalg.norm(trajectories[variation] - nominal_trajectory, axis=1)
            max_divergence = np.max(divergence)
            final_divergence = divergence[-1]
            avg_divergence = np.mean(divergence)
            
            # Altitude difference analysis
            alt_diff = altitude_data[variation] - nominal_altitudes
            final_alt_diff = alt_diff[-1]
            
            position_divergence.append((variation, max_divergence))
            print(f"     B* {variation:+3d}%: Max={max_divergence:.1f}km, Final={final_divergence:.1f}km, Avg={avg_divergence:.1f}km, Alt_diff={final_alt_diff:.1f}km")
    
    # Plot comprehensive analysis
    fig = plt.figure(figsize=(20, 12))
    
    # 3D trajectory plot (first 24 hours only for clarity)
    ax1 = fig.add_subplot(221, projection='3d')
    colors = plt.cm.RdYlBu(np.linspace(0, 1, len(bstar_variations)))
    
    # Show only first 24 hours for 3D plot clarity
    hours_24_points = int(24 * 60 / (7*24*60/57))  # First 24 hours of data points
    
    for i, variation in enumerate(bstar_variations):
        traj = trajectories[variation][:hours_24_points]
        label = f"B* {variation:+d}%" if variation != 0 else "Nominal B*"
        linewidth = 3 if variation == 0 else 1.5
        alpha = 1.0 if variation == 0 else 0.7
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=colors[i], label=label, linewidth=linewidth, alpha=alpha)
    
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_title('3D Orbital Trajectories (First 24 Hours)\nB* Sensitivity Analysis')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Position divergence plot
    ax2 = fig.add_subplot(222)
    if position_divergence:
        variations, divergences = zip(*position_divergence)
        bars = ax2.bar(variations, divergences, color='skyblue', alpha=0.7, edgecolor='navy')
        ax2.set_xlabel('B* Variation (%)')
        ax2.set_ylabel('Max Position Divergence (km)')
        ax2.set_title('Maximum Position Divergence vs B* Variation')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, div in zip(bars, divergences):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{div:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Altitude vs time for different B* values
    ax3 = fig.add_subplot(223)
    time_days = time_points / (24 * 60)  # Convert to days
    
    for i, variation in enumerate(bstar_variations):
        altitudes = altitude_data[variation]
        label = f"B* {variation:+d}%" if variation != 0 else "Nominal B*"
        linewidth = 3 if variation == 0 else 1.5
        alpha = 1.0 if variation == 0 else 0.8
        ax3.plot(time_days, altitudes, color=colors[i], label=label, 
                linewidth=linewidth, alpha=alpha)
    
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Altitude (km)')
    ax3.set_title('Altitude Evolution Over 7 Days')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Position divergence evolution over time
    ax4 = fig.add_subplot(224)
    for i, variation in enumerate(bstar_variations):
        if variation != 0:
            divergence = np.linalg.norm(trajectories[variation] - nominal_trajectory, axis=1)
            label = f"B* {variation:+d}%"
            ax4.plot(time_days, divergence, color=colors[i], label=label, linewidth=1.5)
    
    ax4.set_xlabel('Time (days)')
    ax4.set_ylabel('Position Divergence (km)')
    ax4.set_title('Position Divergence Evolution')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
    print(f"   Original Line 1: {iss_line1}")
    print(f"   Original Line 2: {iss_line2}")
    tle_data = parser.parse_tle(iss_line1, iss_line2, iss_name)
    
    # Test TLE reconstruction
    reconstructed_line1, reconstructed_line2 = parser.tle_data_to_lines(tle_data)
    print(f"   Reconstructed L1: {reconstructed_line1}")
    print(f"   Reconstructed L2: {reconstructed_line2}")
    
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
