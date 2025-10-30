# SGP4 Experiment

## Research Preview / Prototype

This is a **research-grade prototype** exploring differentiable orbital mechanics and machine learning for satellite tracking. It's not production-ready - it's experimental code for learning and experimentation.

## What's Here

I'm experimenting with combining the proven SGP4 orbital propagation model with modern differentiable computing (PyTorch) to enable gradient-based optimization and ML corrections.

**Core Experiments:**
- Differentiable SGP4 implementation in PyTorch
- B* drag coefficient sensitivity analysis
- TLE parsing and orbital element extraction
- Various SGP4 implementation iterations (accurate, corrected, reference, etc.)
- Orbital propagation demonstrations

## Demo: B* Drag Coefficient Analysis

<img width="4767" height="1779" alt="bstar_parsing_accuracy" src="https://github.com/user-attachments/assets/93dcceac-e813-4e7c-b721-b46ce8ccdf05" />

Comparing parsed TLE B* values against the reference SGP4 library for:
- ISS (low B* ≈ −1.16×10⁻⁵)
- High-drag satellite (B* ≈ 5.43×10⁻⁴)
- Deep-space object (B* = 0)

Both implementations yield identical B* values at machine precision.

## How to Use

This is experimental code. To play with it:

```bash
# Clone the repo
git clone https://github.com/ruddro-roy/SGP4-experiment.git
cd SGP4-experiment

# Install dependencies
pip install -r orbit-service/requirements.txt

# Run demos
cd orbit-service
python propagation_demo.py
python bstar_demo_simple.py
python bstar_sensitivity_test.py
```

## What's Interesting

**Differentiable SGP4**: Traditional orbital propagators are black boxes. This PyTorch implementation enables:
- Gradient computation through the propagation chain
- Automatic differentiation for sensitivity analysis
- Potential for ML-based corrections
- Uncertainty quantification

**Multiple Iterations**: You'll see several SGP4 implementations (`sgp4_accurate.py`, `sgp4_corrected.py`, `sgp4_reference.py`, etc.). These represent different experimental approaches and iterations.

## Status

This is **research code**. It's:
- ✅ Good for learning orbital mechanics
- ✅ Good for experimenting with differentiable physics
- ✅ Good for understanding SGP4 internals
- ❌ NOT production-ready
- ❌ NOT for operational satellite tracking
- ❌ NOT thoroughly tested or validated

## Contributing

If you're interested in orbital mechanics or differentiable physics:
- Try the code and share what you learn
- Experiment with the differentiable SGP4
- Test gradient computations
- Share insights or improvements

## Disclaimer

This is experimental research code for educational purposes. For real satellite operations, use certified tracking systems and consult with experts.
