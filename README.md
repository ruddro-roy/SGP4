# SGP4 Orbital Propagation

Python utilities for the SGP4 (Simplified General Perturbations 4) orbit propagation model, including TLE parsing helpers and optional differentiable PyTorch wrappers.

## Features

- Parse and validate two-line element (TLE) sets
- Propagate orbits with the canonical SGP4 equations
- Optional PyTorch module for gradient-based analysis
- Coordinate transforms (TEME to ECEF) and drag sensitivity utilities
- Demo scripts and unit tests for getting started quickly

## Installation

```bash
git clone https://github.com/ruddro-roy/SGP4.git
cd SGP4
pip install -r requirements.txt
```

**Runtime requirements:**
- Python 3.8+
- sgp4 >= 2.23
- numpy >= 1.26.2
- matplotlib >= 3.8.2
- torch >= 2.2.0 (only needed for differentiable workflows)

## Usage

### Basic propagation

```python
from sgp4.api import Satrec

line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

satellite = Satrec.twoline2rv(line1, line2)
jd, fr = 2458826.5, 0.0
error, position, velocity = satellite.sgp4(jd, fr)
```

### TLE parsing helpers

```python
from orbit_service.tle_parser import TLEParser

parser = TLEParser()
tle_data = parser.parse_tle(line1, line2, name="ISS")
print(f"B* drag: {tle_data['bstar_drag']:.8e}")
print(f"Inclination: {tle_data['inclination_deg']:.4f} deg")
```

### Differentiable propagation (optional)

```python
from orbit_service.differentiable_sgp4_torch import DifferentiableSGP4
import torch

dsgp4 = DifferentiableSGP4(line1, line2)
tsince = torch.tensor(360.0, requires_grad=True)
position, velocity = dsgp4(tsince)

loss = torch.norm(position)
loss.backward()
print(tsince.grad)
```

### Demo scripts

```bash
python demo.py              # run the default demo
python demo.py --sensitivity  # explore B* drag effects
python demo.py --verbose      # enable detailed logging
```

## Testing

```bash
python -m pytest tests -v
```

## Project layout

```
orbit_service/
├── tle_parser.py
├── differentiable_sgp4_torch.py
├── sgp4_reference.py
├── live_sgp4.py
└── perturbation_scanner.py
tests/
demo.py
config.py
logging_config.py
requirements.txt
```

## References

- Vallado, D. A., Crawford, P., Hujsak, R., & Kelso, T. S. (2006). Revisiting Spacetrack Report #3. AIAA 2006-6753.
- Hoots, F. R., & Roehrich, R. L. (1980). Models for Propagation of NORAD Element Sets. Spacetrack Report No. 3.
- Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications (4th ed.).

## Disclaimer

Use these scripts for experimentation and education only. They are not certified for operational orbit determination, collision avoidance, or safety-critical tasks.

## License

License selection is pending.
