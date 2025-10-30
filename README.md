# SGP4 Orbital Propagation

Research-grade Python implementation of SGP4 orbital propagation with support for automatic differentiation.

## Overview

This package provides SGP4 (Simplified General Perturbations Satellite Orbit Model 4) orbital propagation capabilities for satellite orbit prediction. The implementation follows Vallado et al. (2006) specifications and uses the proven `sgp4` library for accurate results.

**Key Features:**
- TLE parsing and validation
- SGP4 orbital propagation using established algorithms
- PyTorch wrapper for automatic differentiation
- Coordinate transformations (TEME to ECEF)
- B* drag coefficient sensitivity analysis
- Educational reference implementation

## Installation

```bash
git clone https://github.com/ruddro-roy/SGP4-experiment.git
cd SGP4-experiment
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- sgp4 >= 2.23
- numpy >= 1.26.2
- matplotlib >= 3.8.2
- torch >= 2.2.0 (optional, for differentiable features)

## Usage

### Basic Propagation

```python
from sgp4.api import Satrec

line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

satellite = Satrec.twoline2rv(line1, line2)
jd, fr = 2458826.5, 0.0
error, position, velocity = satellite.sgp4(jd, fr)
```

### TLE Parsing

```python
from orbit_service.tle_parser import TLEParser

parser = TLEParser()
tle_data = parser.parse_tle(line1, line2, name="ISS")
print(f"B* drag: {tle_data['bstar_drag']:.8e}")
print(f"Inclination: {tle_data['inclination_deg']:.4f} deg")
```

### Differentiable Propagation

```python
from orbit_service.differentiable_sgp4_torch import DifferentiableSGP4
import torch

dsgp4 = DifferentiableSGP4(line1, line2)
tsince = torch.tensor(360.0, requires_grad=True)
position, velocity = dsgp4(tsince)

# Compute gradients
loss = torch.norm(position)
loss.backward()
gradient = tsince.grad
```

### Running Demonstrations

```bash
# Basic demonstration
python demo.py

# With B* sensitivity analysis
python demo.py --sensitivity

# With verbose logging
python demo.py --verbose
```

## Testing

```bash
python -m pytest tests/ -v
```

## Project Structure

```
SGP4-experiment/
├── orbit_service/          # Main package
│   ├── tle_parser.py       # TLE parsing and propagation
│   ├── differentiable_sgp4_torch.py  # PyTorch wrapper
│   ├── sgp4_reference.py   # Educational reference
│   ├── live_sgp4.py        # Production tracking
│   └── perturbation_scanner.py  # Deviation analysis
├── tests/                  # Unit tests
├── demo.py                 # Demonstration script
├── config.py               # Configuration and constants
├── logging_config.py       # Logging configuration
└── requirements.txt        # Dependencies
```

## Physical Constants

Uses WGS-72 gravitational constants as specified for SGP4:
- Earth radius: 6378.135 km
- Gravitational parameter μ: 398600.8 km³/s²
- J2: 0.00108262998905892

## References

**Primary Reference:**
Vallado, D. A., Crawford, P., Hujsak, R., & Kelso, T. S. (2006). Revisiting Spacetrack Report #3. AIAA 2006-6753, AIAA/AAS Astrodynamics Specialist Conference.

**Additional References:**
- Hoots, F. R., & Roehrich, R. L. (1980). Models for Propagation of NORAD Element Sets. Spacetrack Report No. 3.
- Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications (4th ed.). Microcosm Press.

## Citation

If you use this code in your research, please cite:

```
@misc{sgp4_experiment,
  author = {Roy, Ruddro},
  title = {SGP4 Orbital Propagation with Differentiable Computing},
  year = {2025},
  url = {https://github.com/ruddro-roy/SGP4-experiment}
}
```

## Disclaimer

**This software is for research and educational purposes only.**

This implementation is NOT intended for:
- Operational satellite tracking
- Safety-critical applications
- Real-time collision avoidance
- Regulatory compliance
- Navigation or guidance systems

For operational use cases requiring high precision and reliability:
- Use certified tracking systems
- Obtain TLE data from authoritative sources (Space-Track.org, CelesTrak)
- Consult with orbital mechanics specialists
- Follow established aerospace safety protocols

**No warranty is provided. Use at your own risk.**

## TLE Data Updates

The fallback TLE data in `config.py` should be updated periodically:
- **LEO satellites:** Weekly
- **MEO satellites:** Monthly  
- **GEO satellites:** Quarterly

Sources for current TLE data:
- [Space-Track.org](https://www.space-track.org/) (requires registration)
- [CelesTrak.org](https://celestrak.org/) (public access)

## License

This project is provided for educational and research purposes.

## Contributing

Contributions are welcome. Please ensure:
- Code follows PEP 8 style guidelines
- All functions have type hints and docstrings
- Tests pass before submitting pull requests
- No emojis or informal comments in code
