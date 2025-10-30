# Error Recovery and Fallback Mechanisms

## Overview

The SGP4 orbital propagation system now includes comprehensive error recovery and fallback mechanisms to handle edge cases gracefully instead of failing hard. This document describes these features and how to use them.

## Key Features

### 1. Automatic Fallback to Two-Body Propagation

When SGP4 propagation fails (due to numerical instabilities, satellite decay, or invalid orbital elements), the system can automatically fall back to simple two-body (Keplerian) propagation.

**Advantages:**
- Prevents complete failure when SGP4 encounters problems
- Provides approximate results when exact propagation is not possible
- Uses energy-conserving physics for reliable fallback behavior

**Limitations:**
- Two-body propagation ignores atmospheric drag
- Less accurate than SGP4 (no perturbations)
- Best used for short-term fallback, not long-term prediction

### 2. Detailed Error Diagnostics

All propagation errors now include:
- **Error code** with clear description
- **Physical interpretation** explaining what went wrong
- **Orbital parameters** at the time of error
- **Recommended actions** for resolution

### 3. Error State Tracking

The system maintains:
- Error history for each satellite
- Last valid state for potential recovery
- Diagnostic information for debugging

## Usage Examples

### Basic Usage with Fallback Enabled

```python
from orbit_service.live_sgp4 import LiveSGP4
from datetime import datetime, timezone

# Create SGP4 instance with fallback enabled (default)
sgp4 = LiveSGP4(enable_fallback=True)

# Load satellite
line1 = "1 25544U 98067A   23259.57580000  .00012022  00000-0  21844-3 0  9995"
line2 = "2 25544  51.6416 220.9944 0004263 122.0101 312.2755 15.49541986415598"
norad_id = sgp4.load_satellite(line1, line2, "ISS")

# Propagate
result = sgp4.propagate(norad_id, datetime.now(timezone.utc))

# Check result
if result['fallback_used']:
    print("Warning: Using two-body fallback")
    print(result['fallback_warning'])
else:
    print("Normal SGP4 propagation succeeded")

print(f"Position: {result['position_km']}")
print(f"Velocity: {result['velocity_kms']}")
```

### Accessing Error Diagnostics

```python
result = sgp4.propagate(norad_id, some_future_time)

if result['sgp4_error_code'] != 0:
    # Error occurred
    print(f"Error: {result['sgp4_error_message']}")
    
    # Get detailed diagnostics
    diag = result['error_diagnostics']
    print(f"\nPhysical Meaning:")
    print(diag['physical_meaning'])
    
    print(f"\nRecommended Action:")
    print(diag['recommended_action'])
    
    print(f"\nOrbital Parameters at Error:")
    print(diag['orbital_parameters'])
```

### Error History Tracking

```python
# Get error history for a satellite
history = sgp4.get_error_history(norad_id)

for error in history:
    print(f"Time: {error['timestamp']}")
    print(f"Error: {error['error_message']}")
    print()
```

### Using Two-Body Fallback Directly

```python
from orbit_service.two_body_fallback import TwoBodyFallback
import numpy as np

# Initial state from SGP4 or other source
r0 = np.array([6778.0, 0.0, 0.0])  # km
v0 = np.array([0.0, 7.669, 0.0])   # km/s

# Create fallback propagator
fallback = TwoBodyFallback(r0, v0)

# Get orbital elements
elements = fallback.get_elements()
print(f"Period: {2*np.pi/elements['n']/60:.2f} minutes")

# Propagate
dt_seconds = 3600.0  # 1 hour
r, v = fallback.propagate(dt_seconds)
print(f"Position: {r} km")
print(f"Velocity: {v} km/s")
```

### Differentiable SGP4 Error Handling

```python
from orbit_service.differentiable_sgp4_torch import DifferentiableSGP4
import torch

dsgp4 = DifferentiableSGP4(line1, line2)

# Propagate
tsince = torch.tensor(1440.0)  # 1 day
r, v = dsgp4(tsince)

# Check for errors
if dsgp4.last_error != 0:
    print(f"SGP4 error {dsgp4.last_error} at t={dsgp4.last_error_time} min")
    print("Using fallback state or zeros")
else:
    print("Propagation successful")
```

## Error Codes and Meanings

| Code | Description | Physical Meaning |
|------|-------------|------------------|
| 0 | No error | Propagation successful |
| 1 | Mean eccentricity out of range | Orbit may be unbound or TLE corrupted |
| 2 | Mean motion negative | Invalid orbital parameters |
| 3 | Perturbed eccentricity out of range | Propagation instability, often from extreme drag |
| 4 | Semi-latus rectum negative | Unphysical orbit state |
| 5 | Satellite decayed | Below minimum altitude, re-entered |
| 6 | Satellite decayed (low altitude) | Same as 5, different detection method |

## Best Practices

### 1. When to Enable Fallback

**Enable fallback (default) when:**
- You need robustness over accuracy
- Handling diverse satellite populations
- Propagating to uncertain future times
- Building user-facing applications

**Disable fallback when:**
- You need to know when SGP4 fails
- Accuracy is critical
- You're validating TLE data
- You want to detect problematic satellites

### 2. Interpreting Results

Always check the propagation method:

```python
if result['propagation_method'] == 'two_body_fallback':
    # Less accurate, use with caution
    # Consider using fresh TLE data
    pass
elif result['propagation_method'] == 'sgp4':
    # Normal accuracy
    pass
```

### 3. Handling Errors

```python
try:
    result = sgp4.propagate(norad_id, timestamp)
    
    if result['sgp4_error_code'] != 0:
        # Error occurred but fallback succeeded
        log.warning(f"SGP4 error, using fallback: {result['sgp4_error_message']}")
        
        # Use result with reduced confidence
        process_result(result, confidence='low')
    else:
        # Normal propagation
        process_result(result, confidence='high')
        
except RuntimeError as e:
    # Both SGP4 and fallback failed
    log.error(f"Propagation completely failed: {e}")
    # Use alternative data source or skip
```

### 4. Long-term Propagation

For long-term propagation (>7 days):
- Expect more errors, especially for LEO satellites
- Use fresh TLE data when possible
- Consider the age of the TLE epoch
- Fallback becomes less accurate over time

## Testing

Run the error recovery tests:

```bash
python -m pytest tests/test_error_recovery.py -v
```

Run the demonstration:

```bash
python demo_error_recovery.py
```

## Implementation Details

### Two-Body Propagation

The fallback uses classical Keplerian orbital mechanics:

1. Convert initial state to orbital elements
2. Propagate mean anomaly forward in time
3. Solve Kepler's equation for eccentric anomaly
4. Convert back to position and velocity

**Conservation properties:**
- Orbital energy conserved
- Angular momentum conserved  
- Orbital shape unchanged (no drag)

### Error Recovery Strategy

When SGP4 fails:

1. Log error with full diagnostics
2. Attempt to get initial state at epoch
3. If epoch state unavailable, estimate from TLE elements
4. Create two-body propagator with initial state
5. Propagate using Keplerian dynamics
6. Return result with fallback flag

### State Caching

To improve fallback performance:
- Last valid SGP4 state is cached
- Fallback propagators are reused
- Error history is maintained but limited to 100 entries

## Limitations

### Two-Body Fallback Limitations

1. **No atmospheric drag** - altitude decay not modeled
2. **No perturbations** - ignores J2, J3, J4 effects
3. **No precession** - RAAN and argument of perigee don't change
4. **Coordinate frame** - assumes inertial frame approximation

### When Fallback Won't Help

Fallback cannot fix:
- Completely invalid TLE data
- Corrupted orbital elements
- Satellites that have already decayed
- Physically impossible orbits (e.g., underground)

## Performance

- **Normal SGP4**: ~10-50 μs per propagation
- **Two-body fallback**: ~5-20 μs per propagation
- **Fallback overhead**: ~1-2 μs (state conversion)

The fallback is actually faster than SGP4 but less accurate.

## Future Enhancements

Potential future improvements:

1. Adaptive time-stepping for fallback
2. Drag approximation in fallback mode
3. Machine learning error prediction
4. Automatic TLE refresh on error
5. Multi-level fallback hierarchy

## References

- Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications (4th ed.)
- Hoots, F. R., & Roehrich, R. L. (1980). Spacetrack Report No. 3
- Vallado, D. A., et al. (2006). Revisiting Spacetrack Report #3. AIAA 2006-6753

## Support

For issues or questions:
- Check the test suite for usage examples
- Run the demo script for interactive examples
- Review error diagnostics in propagation results
- Consult the error code table above
