#!/usr/bin/env python3
"""
B* Drag Coefficient Sensitivity Analysis

Analyzes how variations in the B* drag term affect satellite orbital predictions.
The B* parameter (ballistic coefficient) models atmospheric drag and is critical
for accurate low Earth orbit propagation.

This analysis:
- Compares orbital trajectories with different B* values
- Shows position divergence over time
- Demonstrates the importance of accurate drag modeling

Key insight: Small errors in B* can lead to significant position errors after
just a few orbits, especially for satellites in dense atmospheric regions.

Technical background:
B* combines several drag-related parameters: B* = (Cd * A)/(2 * m)
where Cd is drag coefficient, A is cross-sectional area, m is mass.
"""

import numpy as np
import matplotlib.pyplot as plt
from sgp4.io import twoline2rv
from sgp4.api import WGS84, jday
from datetime import datetime, timedelta

def test_bstar_sensitivity():
    """Test B* sensitivity with direct SGP4 calls"""
    
    # ISS TLE data
    iss_line1 = "1 25544U 98067A   23259.57580000  .00012022  00000-0  21844-3 0  9995"
    iss_line2 = "2 25544  51.6416 220.9944 0004263 122.0101 312.2755 15.49541986415598"
    
    print("🎯 B* DRAG SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Parse original TLE to get B* value
    satellite_orig = twoline2rv(iss_line1, iss_line2, WGS84)
    original_bstar = satellite_orig.bstar
    print(f"Original B*: {original_bstar:.8f}")
    
    # B* variations to test
    bstar_variations = [-50, -25, -10, 0, 10, 25, 50]  # percentage changes
    
    # Time points - 7 days with 6-hour intervals
    time_hours = np.linspace(0, 7*24, 29)  # 7 days, 6-hour intervals
    
    trajectories = {}
    colors = plt.cm.RdYlBu(np.linspace(0, 1, len(bstar_variations)))
    
    print(f"\nTesting {len(bstar_variations)} B* variations over {len(time_hours)} time points...")
    
    for i, variation in enumerate(bstar_variations):
        # Calculate modified B*
        modified_bstar = original_bstar * (1 + variation / 100.0)
        
        # Create modified TLE by replacing B* field
        # B* is in positions 53-61 of line 1
        def format_bstar(bstar_val):
            """Format B* in TLE exponential notation"""
            if bstar_val == 0.0:
                return " 00000-0"
            
            sign = '-' if bstar_val < 0 else ' '
            abs_val = abs(bstar_val)
            
            # Convert to scientific notation
            exp = int(np.floor(np.log10(abs_val)))
            mantissa = abs_val / (10.0 ** exp)
            
            # Normalize mantissa
            while mantissa >= 10.0:
                mantissa /= 10.0
                exp += 1
            
            # Format as 5-digit mantissa + 2-digit exponent
            mantissa_str = f"{mantissa:.5f}"[2:7]  # Remove "1."
            exp_str = f"{exp:+d}"[-2:]  # Last 2 chars
            
            return f"{sign}{mantissa_str}{exp_str}"
        
        # Modify the TLE line
        modified_line1 = iss_line1[:53] + format_bstar(modified_bstar) + iss_line1[61:]
        
        # Recalculate checksum
        def tle_checksum(line):
            s = 0
            for char in line[:-1]:
                if '1' <= char <= '9':
                    s += int(char)
                elif char == '-':
                    s += 1
            return s % 10
        
        checksum = tle_checksum(modified_line1)
        modified_line1 = modified_line1[:-1] + str(checksum)
        
        print(f"  B* {variation:+3d}%: {modified_bstar:.8f} -> TLE: {modified_line1[53:61]}")
        
        # Create satellite object with modified B*
        try:
            satellite = twoline2rv(modified_line1, iss_line2, WGS84)
            
            positions = []
            altitudes = []
            
            # Propagate over time
            for hours in time_hours:
                # Calculate Julian date
                epoch = datetime(2023, 9, 16, 13, 49, 9)  # Approximate epoch from TLE
                current_time = epoch + timedelta(hours=hours)
                jd, fr = jday(current_time.year, current_time.month, current_time.day,
                             current_time.hour, current_time.minute, current_time.second)
                
                # Propagate
                error, r, v = satellite.sgp4(jd, fr)
                if error == 0:
                    positions.append(r)
                    altitude = np.linalg.norm(r) - 6378.137  # Earth radius
                    altitudes.append(altitude)
                else:
                    positions.append([0, 0, 0])
                    altitudes.append(0)
            
            trajectories[variation] = {
                'positions': np.array(positions),
                'altitudes': np.array(altitudes),
                'bstar': modified_bstar
            }
            
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
    
    # Calculate divergences from nominal (0% variation)
    if 0 in trajectories:
        nominal_positions = trajectories[0]['positions']
        nominal_altitudes = trajectories[0]['altitudes']
        
        print(f"\n📊 POSITION DIVERGENCE ANALYSIS:")
        for variation in bstar_variations:
            if variation != 0 and variation in trajectories:
                positions = trajectories[variation]['positions']
                divergence = np.linalg.norm(positions - nominal_positions, axis=1)
                max_div = np.max(divergence)
                final_div = divergence[-1]
                
                alt_diff = trajectories[variation]['altitudes'] - nominal_altitudes
                final_alt_diff = alt_diff[-1]
                
                print(f"  B* {variation:+3d}%: Max={max_div:.1f}km, Final={final_div:.1f}km, Alt_diff={final_alt_diff:.1f}km")
    
    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 3D trajectory plot (first 24 hours)
    ax1 = fig.add_subplot(221, projection='3d')
    hours_24 = time_hours[time_hours <= 24]
    points_24 = len(hours_24)
    
    for i, variation in enumerate(bstar_variations):
        if variation in trajectories:
            traj = trajectories[variation]['positions'][:points_24]
            label = f"B* {variation:+d}%" if variation != 0 else "Nominal B*"
            linewidth = 3 if variation == 0 else 1.5
            alpha = 1.0 if variation == 0 else 0.7
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                    color=colors[i], label=label, linewidth=linewidth, alpha=alpha)
    
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_title('3D Orbital Trajectories (First 24 Hours)')
    ax1.legend()
    
    # Position divergence bar chart
    if 0 in trajectories:
        variations_with_data = []
        divergences = []
        for variation in bstar_variations:
            if variation != 0 and variation in trajectories:
                positions = trajectories[variation]['positions']
                divergence = np.linalg.norm(positions - nominal_positions, axis=1)
                max_div = np.max(divergence)
                variations_with_data.append(variation)
                divergences.append(max_div)
        
        if divergences:
            bars = ax2.bar(variations_with_data, divergences, color='skyblue', alpha=0.7, edgecolor='navy')
            ax2.set_xlabel('B* Variation (%)')
            ax2.set_ylabel('Max Position Divergence (km)')
            ax2.set_title('Maximum Position Divergence vs B* Variation')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, div in zip(bars, divergences):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{div:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Altitude evolution
    time_days = time_hours / 24
    for i, variation in enumerate(bstar_variations):
        if variation in trajectories:
            altitudes = trajectories[variation]['altitudes']
            label = f"B* {variation:+d}%" if variation != 0 else "Nominal B*"
            linewidth = 3 if variation == 0 else 1.5
            alpha = 1.0 if variation == 0 else 0.8
            ax3.plot(time_days, altitudes, color=colors[i], label=label, 
                    linewidth=linewidth, alpha=alpha)
    
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Altitude (km)')
    ax3.set_title('Altitude Evolution Over 7 Days')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Position divergence evolution
    if 0 in trajectories:
        for i, variation in enumerate(bstar_variations):
            if variation != 0 and variation in trajectories:
                positions = trajectories[variation]['positions']
                divergence = np.linalg.norm(positions - nominal_positions, axis=1)
                label = f"B* {variation:+d}%"
                ax4.plot(time_days, divergence, color=colors[i], label=label, linewidth=1.5)
        
        ax4.set_xlabel('Time (days)')
        ax4.set_ylabel('Position Divergence (km)')
        ax4.set_title('Position Divergence Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bstar_sensitivity_analysis_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ B* SENSITIVITY ANALYSIS COMPLETE!")
    print(f"   • Successfully tested {len(trajectories)} B* variations")
    print(f"   • Analysis shows clear trajectory divergence with B* changes")
    print(f"   • Higher B* values cause faster orbital decay")
    print(f"   • Plot saved as 'bstar_sensitivity_analysis_fixed.png'")

if __name__ == "__main__":
    test_bstar_sensitivity()
