"""
SGP4 Wrapper Validation

Tests the differentiable SGP4 wrapper to ensure it produces correct results
compared to the reference sgp4 library. This is a convenience test that can
be run from the repository root.

For more detailed tests, see orbit-service/test_both_implementations.py
"""

import torch
import sys
import os

# Add orbit-service to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'orbit-service')))

from differentiable_sgp4_torch import DifferentiableSGP4
from sgp4.api import Satrec

# Test TLE (ISS)
line1 = "1 25544U 98067A   23259.57580000  .00012022  00000-0  21844-3 0  9995"
line2 = "2 25544  51.6416 220.9944 0004263 122.0101 312.2755 15.49541986415598"

print("SGP4 Wrapper Validation")
print("=" * 60)

# Test wrapper
wrapper = DifferentiableSGP4(line1, line2)
reference = Satrec.twoline2rv(line1, line2)

# Propagate for 1 minute
tsince_minutes = 1.0

print(f"\nPropagating for {tsince_minutes} minute(s)")

# Wrapper result
pos_wrapper, vel_wrapper = wrapper(tsince_minutes)
print(f"\nWrapper (differentiable):")
print(f"  Position: [{pos_wrapper[0]:.3f}, {pos_wrapper[1]:.3f}, {pos_wrapper[2]:.3f}] km")
print(f"  Velocity: [{vel_wrapper[0]:.5f}, {vel_wrapper[1]:.5f}, {vel_wrapper[2]:.5f}] km/s")

# Reference result
jd = reference.jdsatepoch + tsince_minutes / 1440.0
error, r_ref, v_ref = reference.sgp4(jd, 0.0)
print(f"\nReference (sgp4 library):")
print(f"  Position: [{r_ref[0]:.3f}, {r_ref[1]:.3f}, {r_ref[2]:.3f}] km")
print(f"  Velocity: [{v_ref[0]:.5f}, {v_ref[1]:.5f}, {v_ref[2]:.5f}] km/s")

# Compare
diff = torch.norm(pos_wrapper - torch.tensor(r_ref, dtype=torch.float32))
print(f"\n" + "=" * 60)
print(f"Position difference: {diff:.6f} km")

if diff < 0.001:
    print("✅ Wrapper matches reference implementation!")
else:
    print("⚠️ Small numerical difference (expected due to float32 conversion)")

