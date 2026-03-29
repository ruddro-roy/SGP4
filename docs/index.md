---
title: SGP4 Orbital Propagation Prototype
---

# SGP4 Orbital Propagation Prototype

A Python engineering prototype for satellite orbit prediction using the SGP4 model, built to understand orbital mechanics from first principles.

## What It Does

- Parses Two-Line Element (TLE) sets and extracts orbital parameters
- Propagates satellite positions using the SGP4 algorithm
- Transforms coordinates between TEME, ECEF, and geodetic frames
- Handles propagation failures with automatic two-body fallback
- Wraps propagation in PyTorch for gradient-based analysis (experimental)

## ISS Ground Track

<img src="https://raw.githubusercontent.com/ruddro-roy/SGP4/main/assets/iss_ground_track.png" alt="ISS Ground Track" width="100%">

*ISS ground track over 3 orbits (~4.5 h).*

## System Architecture

<img src="https://raw.githubusercontent.com/ruddro-roy/SGP4/main/assets/architecture.png" alt="Architecture" width="100%">

## B* Drag Sensitivity

<img src="https://raw.githubusercontent.com/ruddro-roy/SGP4/main/assets/bstar_sensitivity.png" alt="B* Sensitivity" width="100%">

*B* varied ±50%; km-scale position divergence within days.*

## 24-Hour Propagation

<img src="https://raw.githubusercontent.com/ruddro-roy/SGP4/main/assets/propagation_24h.png" alt="24h Propagation" width="100%">

## Error Recovery

<img src="https://raw.githubusercontent.com/ruddro-roy/SGP4/main/assets/error_recovery_flow.png" alt="Error Recovery" width="100%">

When SGP4 fails, the system logs diagnostics and falls back to Keplerian propagation.

## Quick Start

```bash
git clone https://github.com/ruddro-roy/SGP4.git
cd SGP4
pip install -r requirements.txt
python demo.py --sensitivity
```

## Links

- [Full documentation (README)](https://github.com/ruddro-roy/SGP4#readme)
- [Error Recovery Guide](https://github.com/ruddro-roy/SGP4/blob/main/ERROR_RECOVERY.md)
- [Source code](https://github.com/ruddro-roy/SGP4)

---

*Engineering prototype, not certified for operational use.*
