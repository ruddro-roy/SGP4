# SGP4 Orbital Propagation with Differentiable Computing

This project implements SGP4 (Simplified General Perturbations Satellite Orbit Model 4) orbital propagation with support for automatic differentiation using PyTorch. The implementation follows the SGP4 model as described in Vallado et al. (2006) "Revisiting Spacetrack Report #3" (AAS 06-675) and uses WGS-72 gravitational constants.

## Overview

SGP4 is a widely used analytical propagation model for satellites in Earth orbit. This implementation provides:

- Standard SGP4 propagation using the proven sgp4 library (Vallado et al.)
- TLE (Two-Line Element) parsing and validation
- Differentiable wrapper using PyTorch for gradient-based analysis
- Coordinate transformations (TEME to ECEF)
- Sensitivity analysis tools for drag coefficients and other parameters

## Technical Approach

The project uses the established SGP4 library as the core propagator and wraps it with PyTorch to enable automatic differentiation. This allows gradient computation through the propagation chain while maintaining the accuracy of the proven implementation.

### Key Components

**TLE Parser** (`tle_parser.py`): Parses Two-Line Element sets according to standard specifications, extracts orbital elements, and reconstructs TLE strings with modified parameters.

**Differentiable SGP4** (`differentiable_sgp4_torch.py`): PyTorch wrapper around the sgp4 library that enables gradient computation and includes an optional neural network for learned corrections.

**Reference Implementations** (`sgp4_reference.py`, `sgp4_final.py`, `sgp4_corrected.py`): Pure Python implementations of SGP4 for educational purposes and validation, implementing the algorithm from AAS 06-675.

**Demonstration Scripts**: Scripts showing orbital propagation, B* drag coefficient sensitivity, and other analyses.

## Installation

```bash
git clone https://github.com/ruddro-roy/SGP4-experiment.git
cd SGP4-experiment
pip install -r orbit-service/requirements.txt
```

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

### Differentiable Propagation

```python
from orbit-service.differentiable_sgp4_torch import DifferentiableSGP4
import torch

dsgp4 = DifferentiableSGP4(line1, line2)
tsince = torch.tensor(360.0, requires_grad=True)  # 360 minutes
position, velocity = dsgp4(tsince)

# Compute gradients
loss = torch.norm(position)
loss.backward()
gradient = tsince.grad
```

### TLE Parsing

```python
from orbit-service.tle_parser import TLEParser

parser = TLEParser()
tle_data = parser.parse_tle(line1, line2, name="ISS")
print(f"B* drag coefficient: {tle_data['bstar_drag']:.8e}")
print(f"Eccentricity: {tle_data['eccentricity']:.6f}")
print(f"Inclination: {tle_data['inclination_deg']:.4f} degrees")
```

### Running Demonstrations

```bash
cd orbit-service

# Basic propagation demonstration
python propagation_demo.py

# B* drag sensitivity analysis
python bstar_sensitivity_test.py

# Simple B* demonstration
python bstar_demo_simple.py
```

## Implementation Details

### Constants

All implementations use WGS-72 gravitational constants as specified for SGP4:

- Gravitational parameter (μ): 398600.8 km³/s²
- Earth radius (R_E): 6378.135 km
- J2: 0.00108262998905892
- J3: -0.00000253215306
- J4: -0.00000165597

### Coordinate Systems

The implementation handles the following coordinate frames:

- **TEME** (True Equator Mean Equinox): The native output frame of SGP4
- **ECEF** (Earth-Centered Earth-Fixed): Ground station coordinates, converted via GMST rotation

### TLE Format

Two-Line Element sets follow the standard format:

```
Line 1: 1 NNNNNC NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN
Line 2: 2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNN
```

Key fields include:
- Epoch (year and day of year)
- Mean motion and derivatives
- B* drag term (ballistic coefficient)
- Classical orbital elements (inclination, RAAN, eccentricity, argument of perigee, mean anomaly)

## Testing

Run the test suite:

```bash
cd orbit-service
python test_sgp4.py
python test_both_implementations.py
```

## Dependencies

- `sgp4>=2.23`: Official SGP4 implementation by Brandon Rhodes (based on Vallado et al.)
- `numpy>=1.26.2`: Numerical computations
- `matplotlib>=3.8.2`: Plotting and visualization
- `torch>=2.2.0`: Automatic differentiation (optional, for differentiable features)

## References

1. Vallado, D. A., Crawford, P., Hujsak, R., & Kelso, T. S. (2006). "Revisiting Spacetrack Report #3." AIAA 2006-6753, AIAA/AAS Astrodynamics Specialist Conference.

2. Hoots, F. R., & Roehrich, R. L. (1980). "Models for Propagation of NORAD Element Sets." Spacetrack Report No. 3.

3. Vallado, D. A. (2013). "Fundamentals of Astrodynamics and Applications" (4th ed.). Microcosm Press.

## License

This project is provided for educational and research purposes. For operational satellite tracking, use certified tracking systems.

## Limitations

This implementation is suitable for:
- Learning orbital mechanics concepts
- Sensitivity analysis and gradient-based optimization
- TLE parsing and manipulation
- Educational demonstrations

It is not intended for:
- Operational satellite tracking requiring high precision
- Safety-critical applications
- Real-time collision avoidance
- Regulatory compliance scenarios

For production use cases, consult with orbital mechanics specialists and use certified tracking systems.
