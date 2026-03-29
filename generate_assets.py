"""
Asset generation script for SGP4 repository documentation.

Produces publication-quality plots demonstrating the prototype's capabilities:
- ISS orbit ground track
- B* drag sensitivity analysis (4-panel)
- Propagation accuracy comparison
- Architecture overview diagram

Usage:
    python generate_assets.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import timedelta
from orbit_service.tle_parser import TLEParser
from sgp4.api import Satrec

# Consistent style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "monospace",
    "font.size": 10,
})

ACCENT = "#58a6ff"
ACCENT2 = "#f78166"
ACCENT3 = "#3fb950"
ACCENT4 = "#d2a8ff"
WARN = "#d29922"

ISS_LINE1 = "1 25544U 98067A   23259.57580000  .00012022  00000-0  21844-3 0  9995"
ISS_LINE2 = "2 25544  51.6416 220.9944 0004263 122.0101 312.2755 15.49541986415598"


def generate_ground_track():
    """Generate ISS ground track over ~3 orbits."""
    parser = TLEParser()
    tle_data = parser.parse_tle(ISS_LINE1, ISS_LINE2, "ISS (ZARYA)")

    # Propagate for ~4.5 hours (roughly 3 ISS orbits)
    minutes = np.linspace(0, 270, 800)
    lats, lons = [], []

    for t in minutes:
        result = parser.propagate_orbit(tle_data, t)
        lats.append(result["latitude_deg"])
        lons.append(result["longitude_deg"])

    lats = np.array(lats)
    lons = np.array(lons)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Simple coastline rectangle hints (no external data needed)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.fill_between([-180, 180], -90, 90, color="#0d1117", alpha=1)

    # Grid for lat/lon
    for lat_line in range(-60, 90, 30):
        ax.axhline(lat_line, color="#21262d", linewidth=0.5)
    for lon_line in range(-150, 180, 30):
        ax.axvline(lon_line, color="#21262d", linewidth=0.5)

    # Plot ground track with segments (break at longitude wraps)
    seg_lats, seg_lons = [lats[0]], [lons[0]]
    for i in range(1, len(lons)):
        if abs(lons[i] - lons[i - 1]) > 180:
            ax.plot(seg_lons, seg_lats, color=ACCENT, linewidth=1.8, alpha=0.9)
            seg_lats, seg_lons = [], []
        seg_lats.append(lats[i])
        seg_lons.append(lons[i])
    ax.plot(seg_lons, seg_lats, color=ACCENT, linewidth=1.8, alpha=0.9)

    # Mark start and end
    ax.scatter(lons[0], lats[0], color=ACCENT3, s=80, zorder=5, marker="o", edgecolors="white", linewidths=0.8)
    ax.scatter(lons[-1], lats[-1], color=ACCENT2, s=80, zorder=5, marker="s", edgecolors="white", linewidths=0.8)

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("ISS Ground Track — 3 Orbits (~4.5 hours)  |  SGP4 Propagation", fontsize=13, fontweight="bold", pad=12)

    # Inclination band
    ax.axhline(51.64, color=ACCENT4, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(-51.64, color=ACCENT4, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(175, 53, "i = 51.6°", ha="right", fontsize=8, color=ACCENT4, alpha=0.7)

    legend_elements = [
        mpatches.Patch(color=ACCENT3, label="Start position"),
        mpatches.Patch(color=ACCENT2, label="End position"),
        mpatches.Patch(color=ACCENT, label="Ground track"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9, facecolor="#161b22", edgecolor="#30363d")

    plt.tight_layout()
    fig.savefig("assets/iss_ground_track.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  -> assets/iss_ground_track.png")


def generate_bstar_sensitivity():
    """Generate 4-panel B* sensitivity analysis."""
    parser = TLEParser()
    tle_data = parser.parse_tle(ISS_LINE1, ISS_LINE2, "ISS (ZARYA)")
    original_bstar = tle_data["bstar_drag"]

    variations = [-50, -25, -10, 0, 10, 25, 50]
    time_points = np.linspace(0, 7 * 24 * 60, 57)
    time_days = time_points / (24 * 60)

    trajectories = {}
    altitude_data = {}

    for variation in variations:
        modified = tle_data.copy()
        modified["bstar_drag"] = original_bstar * (1 + variation / 100.0)
        # Remove original lines so tle_data_to_lines reconstructs with modified bstar
        modified.pop("line1", None)
        modified.pop("line2", None)
        line1, line2 = parser.tle_data_to_lines(modified)
        satellite = Satrec.twoline2rv(line1, line2)

        positions, altitudes = [], []
        epoch_dt = tle_data["epoch_datetime"]

        for t in time_points:
            current = epoch_dt + timedelta(minutes=t)
            jd, fr = parser.datetime_to_jd_fr(current)
            error, r, v = satellite.sgp4(jd, fr)
            if error == 0:
                positions.append(r)
                altitudes.append(np.linalg.norm(r) - 6378.137)
            else:
                positions.append([0, 0, 0])
                altitudes.append(0)

        trajectories[variation] = np.array(positions)
        altitude_data[variation] = np.array(altitudes)

    nominal = trajectories[0]
    nominal_alt = altitude_data[0]

    # Color map
    cmap = plt.cm.coolwarm
    norm_vals = np.linspace(0, 1, len(variations))

    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 2, hspace=0.32, wspace=0.3)

    # --- Panel 1: 3D trajectory (first 24h) ---
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.set_facecolor("#0d1117")
    hrs24 = max(1, int(24 * 60 / (7 * 24 * 60 / 57)))

    for i, var in enumerate(variations):
        traj = trajectories[var][:hrs24]
        lw = 2.5 if var == 0 else 1.2
        alpha = 1.0 if var == 0 else 0.65
        label = "Nominal" if var == 0 else f"{var:+d}%"
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=cmap(norm_vals[i]), linewidth=lw, alpha=alpha, label=label)

    ax1.set_xlabel("X (km)", fontsize=8)
    ax1.set_ylabel("Y (km)", fontsize=8)
    ax1.set_zlabel("Z (km)", fontsize=8)
    ax1.set_title("3D Trajectories — First 24 Hours", fontsize=11, fontweight="bold", pad=10)
    ax1.tick_params(labelsize=7)

    # --- Panel 2: Max divergence bar ---
    ax2 = fig.add_subplot(gs[0, 1])
    divergences = []
    for var in variations:
        if var != 0:
            div = np.max(np.linalg.norm(trajectories[var] - nominal, axis=1))
            divergences.append((var, div))

    vars_list, divs_list = zip(*divergences)
    colors_bar = [ACCENT2 if v > 0 else ACCENT for v in vars_list]
    bars = ax2.bar(vars_list, divs_list, color=colors_bar, alpha=0.85, edgecolor="#30363d", width=8)
    for bar, div in zip(bars, divs_list):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{div:.0f}", ha="center", va="bottom", fontsize=8, color="#c9d1d9")
    ax2.set_xlabel("B* Variation (%)")
    ax2.set_ylabel("Max Position Divergence (km)")
    ax2.set_title("Peak Divergence vs. B* Variation", fontsize=11, fontweight="bold", pad=10)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Altitude evolution ---
    ax3 = fig.add_subplot(gs[1, 0])
    for i, var in enumerate(variations):
        lw = 2.5 if var == 0 else 1.0
        alpha = 1.0 if var == 0 else 0.7
        label = "Nominal" if var == 0 else f"{var:+d}%"
        ax3.plot(time_days, altitude_data[var], color=cmap(norm_vals[i]), linewidth=lw, alpha=alpha, label=label)

    ax3.set_xlabel("Time (days)")
    ax3.set_ylabel("Altitude (km)")
    ax3.set_title("Altitude Evolution — 7 Days", fontsize=11, fontweight="bold", pad=10)
    ax3.legend(fontsize=7, loc="upper right", ncol=2, facecolor="#161b22", edgecolor="#30363d")
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Divergence evolution ---
    ax4 = fig.add_subplot(gs[1, 1])
    for i, var in enumerate(variations):
        if var != 0:
            div_series = np.linalg.norm(trajectories[var] - nominal, axis=1)
            ax4.plot(time_days, div_series, color=cmap(norm_vals[i]), linewidth=1.3, label=f"{var:+d}%")

    ax4.set_xlabel("Time (days)")
    ax4.set_ylabel("Position Divergence (km)")
    ax4.set_title("Divergence Growth Over Time", fontsize=11, fontweight="bold", pad=10)
    ax4.legend(fontsize=7, loc="upper left", facecolor="#161b22", edgecolor="#30363d")
    ax4.grid(True, alpha=0.3)

    fig.suptitle("B* Drag Coefficient Sensitivity Analysis  —  ISS (NORAD 25544)", fontsize=14, fontweight="bold", y=0.98, color="#c9d1d9")
    fig.savefig("assets/bstar_sensitivity.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  -> assets/bstar_sensitivity.png")


def generate_propagation_demo():
    """Generate a compact propagation result plot showing altitude + radius over one day."""
    parser = TLEParser()
    tle_data = parser.parse_tle(ISS_LINE1, ISS_LINE2, "ISS (ZARYA)")

    minutes = np.linspace(0, 24 * 60, 300)
    altitudes, radii, lats = [], [], []

    for t in minutes:
        result = parser.propagate_orbit(tle_data, t)
        altitudes.append(result["altitude_km"])
        radii.append(result["orbital_radius_km"])
        lats.append(result["latitude_deg"])

    altitudes = np.array(altitudes)
    radii = np.array(radii)
    lats = np.array(lats)
    hours = minutes / 60

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(hours, altitudes, color=ACCENT, linewidth=1.2)
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("ISS Orbital Parameters — 24-Hour Propagation  |  SGP4", fontsize=13, fontweight="bold", pad=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(np.mean(altitudes), color=WARN, linewidth=0.8, linestyle="--", alpha=0.6)
    axes[0].text(23.5, np.mean(altitudes) + 0.3, f"mean={np.mean(altitudes):.1f} km", ha="right", fontsize=8, color=WARN)

    axes[1].plot(hours, radii, color=ACCENT3, linewidth=1.2)
    axes[1].set_ylabel("Orbital Radius (km)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(hours, lats, color=ACCENT4, linewidth=1.2)
    axes[2].set_ylabel("Sub-satellite Latitude (deg)")
    axes[2].set_xlabel("Time (hours since epoch)")
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(51.64, color="#484f58", linewidth=0.6, linestyle=":")
    axes[2].axhline(-51.64, color="#484f58", linewidth=0.6, linestyle=":")

    plt.tight_layout()
    fig.savefig("assets/propagation_24h.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  -> assets/propagation_24h.png")


def generate_architecture_diagram():
    """Generate a system architecture overview as a diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor=ACCENT, linewidth=1.5)
    box_style2 = dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor=ACCENT3, linewidth=1.5)
    box_style3 = dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor=ACCENT2, linewidth=1.5)
    box_style4 = dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor=ACCENT4, linewidth=1.5)
    box_dim = dict(boxstyle="round,pad=0.4", facecolor="#21262d", edgecolor="#30363d", linewidth=1)

    title_props = dict(fontsize=11, fontweight="bold", ha="center", va="center", color="#c9d1d9")
    sub_props = dict(fontsize=8, ha="center", va="center", color="#8b949e")

    # Title
    ax.text(7, 7.5, "SGP4 Orbital Propagation — System Architecture", fontsize=15, fontweight="bold", ha="center", color="#c9d1d9")

    # --- Input layer ---
    ax.text(2, 6.3, "TLE Input", **title_props, bbox=box_dim)
    ax.text(2, 5.7, "Two-Line Element sets\n(NORAD / CelesTrak)", **sub_props)

    # --- Core modules ---
    ax.text(7, 6.3, "TLE Parser", **title_props, bbox=box_style)
    ax.text(7, 5.6, "Parse, validate, reconstruct\nCoordinate transforms", **sub_props)

    ax.text(7, 4.2, "SGP4 Propagator", **title_props, bbox=box_style)
    ax.text(7, 3.5, "Proven sgp4 library\nWGS-72 constants", **sub_props)

    # --- Branches ---
    ax.text(3, 2.3, "Error Recovery", **title_props, bbox=box_style3)
    ax.text(3, 1.6, "Fallback to two-body\nDiagnostics + history", **sub_props)

    ax.text(7, 2.3, "Live Tracking", **title_props, bbox=box_style2)
    ax.text(7, 1.6, "Multi-satellite mgmt\nBatch propagation", **sub_props)

    ax.text(11, 2.3, "Differentiable SGP4", **title_props, bbox=box_style4)
    ax.text(11, 1.6, "PyTorch autograd\nML corrections (exp.)", **sub_props)

    # --- Output ---
    ax.text(12, 6.3, "Output", **title_props, bbox=box_dim)
    ax.text(12, 5.7, "Position / Velocity\nLat, Lon, Alt", **sub_props)

    # --- Arrows ---
    arrow_props = dict(arrowstyle="->,head_width=0.3,head_length=0.15", color="#484f58", linewidth=1.5)
    ax.annotate("", xy=(5.5, 6.3), xytext=(3.3, 6.3), arrowprops=arrow_props)
    ax.annotate("", xy=(7, 5.1), xytext=(7, 5.4), arrowprops=arrow_props)
    ax.annotate("", xy=(10.5, 6.3), xytext=(8.5, 6.3), arrowprops=arrow_props)

    # Down from propagator
    ax.annotate("", xy=(3, 2.9), xytext=(5.5, 3.8), arrowprops=arrow_props)
    ax.annotate("", xy=(7, 2.9), xytext=(7, 3.2), arrowprops=arrow_props)
    ax.annotate("", xy=(11, 2.9), xytext=(8.5, 3.8), arrowprops=arrow_props)

    # Dashed fallback arrow
    ax.annotate("", xy=(5.5, 4.0), xytext=(3.5, 2.9),
                arrowprops=dict(arrowstyle="->,head_width=0.2", color=ACCENT2, linewidth=1, linestyle="dashed"))
    ax.text(3.8, 3.5, "fallback", fontsize=7, color=ACCENT2, rotation=55)

    # Reference implementation note
    ax.text(12, 4.2, "Reference SGP4", **title_props, bbox=box_dim)
    ax.text(12, 3.6, "Educational impl.\nValidation baseline", **sub_props)
    ax.annotate("", xy=(12, 3.1), xytext=(12, 3.4), arrowprops=dict(arrowstyle="<->", color="#484f58", linewidth=0.8, linestyle="dotted"))
    ax.text(12.7, 3.25, "validate", fontsize=7, color="#484f58")

    # Bottom note
    ax.text(7, 0.5, "Prototype built for learning and demonstration  •  Not certified for operational use",
            fontsize=9, ha="center", color="#484f58", style="italic")

    fig.savefig("assets/architecture.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  -> assets/architecture.png")


def generate_error_recovery_diagram():
    """Generate error recovery flow diagram."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    box = dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor=ACCENT, linewidth=1.5)
    box_err = dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor=ACCENT2, linewidth=1.5)
    box_ok = dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor=ACCENT3, linewidth=1.5)
    box_q = dict(boxstyle="round,pad=0.5", facecolor="#21262d", edgecolor=WARN, linewidth=1.5)

    tp = dict(fontsize=10, fontweight="bold", ha="center", va="center", color="#c9d1d9")
    sp = dict(fontsize=8, ha="center", va="center", color="#8b949e")
    arrow = dict(arrowstyle="->,head_width=0.25", color="#484f58", linewidth=1.3)

    ax.text(6, 6.5, "Error Recovery and Fallback Flow", fontsize=14, fontweight="bold", ha="center", color="#c9d1d9")

    # Flow
    ax.text(2, 5.5, "Load TLE", **tp, bbox=box)
    ax.annotate("", xy=(4.5, 5.5), xytext=(3.2, 5.5), arrowprops=arrow)

    ax.text(6, 5.5, "SGP4\nPropagate", **tp, bbox=box)
    ax.annotate("", xy=(6, 4.6), xytext=(6, 5.0), arrowprops=arrow)

    ax.text(6, 4.2, "Error?", **tp, bbox=box_q)

    # No error path
    ax.annotate("", xy=(9.5, 4.2), xytext=(7.5, 4.2), arrowprops=arrow)
    ax.text(8.5, 4.5, "No", fontsize=9, color=ACCENT3, ha="center")
    ax.text(10.5, 4.2, "Return\nResult", **tp, bbox=box_ok)

    # Error path
    ax.annotate("", xy=(6, 3.0), xytext=(6, 3.7), arrowprops=arrow)
    ax.text(6.4, 3.4, "Yes", fontsize=9, color=ACCENT2)

    ax.text(6, 2.6, "Log Error\n+ Diagnostics", **tp, bbox=box_err)
    ax.annotate("", xy=(6, 1.6), xytext=(6, 2.1), arrowprops=arrow)

    ax.text(6, 1.2, "Fallback\nenabled?", **tp, bbox=box_q)

    # Fallback enabled
    ax.annotate("", xy=(9.5, 1.2), xytext=(7.5, 1.2), arrowprops=arrow)
    ax.text(8.5, 1.5, "Yes", fontsize=9, color=ACCENT3, ha="center")
    ax.text(10.5, 1.2, "Two-Body\nFallback", **tp, bbox=box_ok)

    # Fallback disabled
    ax.annotate("", xy=(3.0, 1.2), xytext=(4.5, 1.2), arrowprops=arrow)
    ax.text(3.7, 1.5, "No", fontsize=9, color=ACCENT2, ha="center")
    ax.text(1.8, 1.2, "Raise\nError", **tp, bbox=box_err)

    ax.text(6, 0.3, "Two-body fallback conserves energy and angular momentum but ignores perturbations",
            fontsize=8, ha="center", color="#484f58", style="italic")

    fig.savefig("assets/error_recovery_flow.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  -> assets/error_recovery_flow.png")


if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)

    print("Generating visual assets...")
    generate_ground_track()
    generate_bstar_sensitivity()
    generate_propagation_demo()
    generate_architecture_diagram()
    generate_error_recovery_diagram()
    print("\nDone. All assets saved to assets/")
