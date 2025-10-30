"""
SGP4 Orbital Propagation Demonstration

This script demonstrates the key capabilities of the SGP4 orbital propagation package:
- TLE parsing and validation
- Orbital propagation using the proven sgp4 library
- Coordinate transformations (TEME to ECEF)
- B* drag coefficient sensitivity analysis
- Visualization of orbital trajectories

Usage:
    python demo.py [--sensitivity] [--verbose]

Arguments:
    --sensitivity: Run B* drag sensitivity analysis
    --verbose: Enable debug logging

References:
    Vallado, D. A., Crawford, P., Hujsak, R., & Kelso, T. S. (2006).
    Revisiting Spacetrack Report #3. AIAA 2006-6753.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from typing import Dict, Any, List, Tuple

from orbit_service.tle_parser import TLEParser
from sgp4.api import Satrec
from logging_config import get_logger, configure_logging
import logging

logger = get_logger(__name__)

# ISS TLE data (as of September 2023)
ISS_LINE1 = "1 25544U 98067A   23259.57580000  .00012022  00000-0  21844-3 0  9995"
ISS_LINE2 = "2 25544  51.6416 220.9944 0004263 122.0101 312.2755 15.49541986415598"
ISS_NAME = "ISS (ZARYA)"


def demonstrate_tle_parsing(
    parser: TLEParser, line1: str, line2: str, name: str
) -> Dict[str, Any]:
    """
    Demonstrate TLE parsing capabilities.

    Parameters
    ----------
    parser : TLEParser
        TLE parser instance
    line1 : str
        TLE line 1
    line2 : str
        TLE line 2
    name : str
        Satellite name

    Returns
    -------
    dict
        Parsed TLE data
    """
    logger.info(f"Parsing TLE for {name}")
    logger.debug(f"Line 1: {line1}")
    logger.debug(f"Line 2: {line2}")

    tle_data = parser.parse_tle(line1, line2, name)

    logger.info(f"NORAD ID: {tle_data['norad_id']}")
    logger.info(f"Inclination: {tle_data['inclination_deg']:.4f} degrees")
    logger.info(f"RAAN: {tle_data['raan_deg']:.4f} degrees")
    logger.info(f"Eccentricity: {tle_data['eccentricity']:.6f}")
    logger.info(f"Argument of Perigee: {tle_data['arg_perigee_deg']:.4f} degrees")
    logger.info(f"Mean Anomaly: {tle_data['mean_anomaly_deg']:.4f} degrees")
    logger.info(f"Mean Motion: {tle_data['mean_motion_rev_per_day']:.8f} rev/day")
    logger.info(f"B* Drag: {tle_data['bstar_drag']:.8e}")

    # Test TLE reconstruction
    reconstructed_line1, reconstructed_line2 = parser.tle_data_to_lines(tle_data)
    logger.debug(f"Reconstructed Line 1: {reconstructed_line1}")
    logger.debug(f"Reconstructed Line 2: {reconstructed_line2}")

    return tle_data


def demonstrate_propagation(parser: TLEParser, tle_data: Dict[str, Any]) -> None:
    """
    Demonstrate orbital propagation at multiple time intervals.

    Parameters
    ----------
    parser : TLEParser
        TLE parser instance
    tle_data : dict
        Parsed TLE data
    """
    time_intervals = [0, 30, 60, 90, 120]  # minutes

    logger.info("Orbital propagation results (TEME coordinates)")

    for tsince in time_intervals:
        result = parser.propagate_orbit(tle_data, tsince)
        pos = result["position_teme_km"]
        radius = result["orbital_radius_km"]

        logger.info(
            f"t={tsince:3.0f}min: "
            f"x={pos['x']:8.2f}km y={pos['y']:8.2f}km z={pos['z']:8.2f}km "
            f"r={radius:8.2f}km"
        )


def analyze_bstar_sensitivity(
    parser: TLEParser,
    tle_data: Dict[str, Any],
    variations: List[int] = [-50, -25, -10, 0, 10, 25, 50],
) -> Tuple[Dict[int, np.ndarray], List[Tuple[int, float]]]:
    """
    Analyze orbital trajectory sensitivity to B* drag coefficient variations.

    Parameters
    ----------
    parser : TLEParser
        TLE parser instance
    tle_data : dict
        Parsed TLE data
    variations : list of int
        B* percentage variations to test

    Returns
    -------
    trajectories : dict
        Mapping from variation percentage to trajectory array
    divergences : list of tuple
        List of (variation, max_divergence) pairs

    References
    ----------
    The B* drag term models atmospheric drag effects. Small variations can lead
    to significant position errors over time, especially for LEO satellites.
    """
    logger.info("Starting B* drag sensitivity analysis")
    logger.info(f"Variations: {variations}%")
    logger.info("Analysis period: 7 days")

    original_bstar = tle_data.get("bstar_drag", 0.0)
    logger.info(f"Original B*: {original_bstar:.8e}")

    # Extended time period - 7 days with 3-hour intervals
    time_points = np.linspace(0, 7 * 24 * 60, 57)  # 0 to 7 days, 3-hour intervals
    trajectories = {}
    altitude_data = {}
    nominal_trajectory = None

    for variation in variations:
        modified_tle_data = tle_data.copy()
        modified_bstar = original_bstar * (1 + variation / 100.0)
        modified_tle_data["bstar_drag"] = modified_bstar

        logger.debug(f"Testing B* {variation:+d}%: {modified_bstar:.8e}")

        line1, line2 = parser.tle_data_to_lines(modified_tle_data)
        satellite = Satrec.twoline2rv(line1, line2)

        positions = []
        altitudes = []
        epoch_datetime = tle_data["epoch_datetime"]

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
                logger.warning(
                    f"SGP4 error {error} at t={t:.1f}min for B*{variation:+d}%"
                )

        trajectories[variation] = np.array(positions)
        altitude_data[variation] = np.array(altitudes)

        if variation == 0:
            nominal_trajectory = np.array(positions)
            nominal_altitudes = np.array(altitudes)

    # Calculate divergences
    divergences = []
    logger.info("Position divergence analysis:")

    for variation in variations:
        if variation != 0:
            divergence = np.linalg.norm(
                trajectories[variation] - nominal_trajectory, axis=1
            )
            max_div = np.max(divergence)
            final_div = divergence[-1]
            avg_div = np.mean(divergence)

            alt_diff = altitude_data[variation] - nominal_altitudes
            final_alt_diff = alt_diff[-1]

            divergences.append((variation, max_div))
            logger.info(
                f"B* {variation:+3d}%: "
                f"max={max_div:.1f}km final={final_div:.1f}km "
                f"avg={avg_div:.1f}km alt_diff={final_alt_diff:.1f}km"
            )

    # Visualize results
    visualize_sensitivity_analysis(
        trajectories,
        divergences,
        altitude_data,
        nominal_altitudes,
        time_points,
        variations,
    )

    return trajectories, divergences


def visualize_sensitivity_analysis(
    trajectories: Dict[int, np.ndarray],
    divergences: List[Tuple[int, float]],
    altitude_data: Dict[int, np.ndarray],
    nominal_altitudes: np.ndarray,
    time_points: np.ndarray,
    variations: List[int],
) -> None:
    """
    Create comprehensive visualization of B* sensitivity analysis.

    Parameters
    ----------
    trajectories : dict
        Trajectories for each B* variation
    divergences : list of tuple
        Position divergences
    altitude_data : dict
        Altitude time series for each variation
    nominal_altitudes : ndarray
        Nominal altitude time series
    time_points : ndarray
        Time points in minutes
    variations : list of int
        B* variations tested
    """
    fig = plt.figure(figsize=(20, 12))
    colors = plt.cm.RdYlBu(np.linspace(0, 1, len(variations)))

    # 3D trajectory plot (first 24 hours)
    ax1 = fig.add_subplot(221, projection="3d")
    hours_24_points = int(24 * 60 / (7 * 24 * 60 / 57))

    for i, variation in enumerate(variations):
        traj = trajectories[variation][:hours_24_points]
        label = f"B* {variation:+d}%" if variation != 0 else "Nominal B*"
        linewidth = 3 if variation == 0 else 1.5
        alpha = 1.0 if variation == 0 else 0.7
        ax1.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            color=colors[i],
            label=label,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax1.set_xlabel("X (km)")
    ax1.set_ylabel("Y (km)")
    ax1.set_zlabel("Z (km)")
    ax1.set_title("3D Orbital Trajectories (First 24 Hours)")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Position divergence bar plot
    ax2 = fig.add_subplot(222)
    if divergences:
        vars_list, divs_list = zip(*divergences)
        bars = ax2.bar(
            vars_list, divs_list, color="skyblue", alpha=0.7, edgecolor="navy"
        )
        ax2.set_xlabel("B* Variation (%)")
        ax2.set_ylabel("Max Position Divergence (km)")
        ax2.set_title("Maximum Position Divergence vs B* Variation")
        ax2.grid(True, alpha=0.3)

        for bar, div in zip(bars, divs_list):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{div:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Altitude evolution
    ax3 = fig.add_subplot(223)
    time_days = time_points / (24 * 60)

    for i, variation in enumerate(variations):
        altitudes = altitude_data[variation]
        label = f"B* {variation:+d}%" if variation != 0 else "Nominal B*"
        linewidth = 3 if variation == 0 else 1.5
        alpha = 1.0 if variation == 0 else 0.8
        ax3.plot(
            time_days,
            altitudes,
            color=colors[i],
            label=label,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax3.set_xlabel("Time (days)")
    ax3.set_ylabel("Altitude (km)")
    ax3.set_title("Altitude Evolution Over 7 Days")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Position divergence evolution
    ax4 = fig.add_subplot(224)
    nominal_traj = trajectories[0]

    for i, variation in enumerate(variations):
        if variation != 0:
            divergence = np.linalg.norm(trajectories[variation] - nominal_traj, axis=1)
            label = f"B* {variation:+d}%"
            ax4.plot(time_days, divergence, color=colors[i], label=label, linewidth=1.5)

    ax4.set_xlabel("Time (days)")
    ax4.set_ylabel("Position Divergence (km)")
    ax4.set_title("Position Divergence Evolution")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = "bstar_sensitivity_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved sensitivity analysis plot to {output_file}")
    plt.close()


def main() -> None:
    """Main demonstration entry point."""
    parser = argparse.ArgumentParser(
        description="SGP4 Orbital Propagation Demonstration"
    )
    parser.add_argument(
        "--sensitivity", action="store_true", help="Run B* drag sensitivity analysis"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        configure_logging(level=logging.DEBUG)

    logger.info("SGP4 Orbital Propagation Demonstration")
    logger.info("=" * 60)

    # Initialize parser
    tle_parser = TLEParser()

    # Demonstrate TLE parsing
    tle_data = demonstrate_tle_parsing(tle_parser, ISS_LINE1, ISS_LINE2, ISS_NAME)

    # Demonstrate propagation
    logger.info("")
    demonstrate_propagation(tle_parser, tle_data)

    # Optional: Run sensitivity analysis
    if args.sensitivity:
        logger.info("")
        analyze_bstar_sensitivity(tle_parser, tle_data)
        logger.info("Sensitivity analysis complete")

    logger.info("=" * 60)
    logger.info("Demonstration complete")


if __name__ == "__main__":
    main()
