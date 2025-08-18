# test_both_implementations.py
import torch
import sys
import os

# Add orbit-service to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'orbit-service')))

from differentiable_sgp4_torch import DifferentiableSGP4 as DifferentiableSGP4Wrapper
from differentiable_sgp4 import DifferentiableSGP4 as DifferentiableSGP4Pure

# Test TLE (ISS)
line1 = "1 25544U 98067A   23259.57580000  .00012022  00000-0  21844-3 0  9995"
line2 = "2 25544  51.6416 220.9944 0004263 122.0101 312.2755 15.49541986415598"

# Test both implementations
wrapper = DifferentiableSGP4Wrapper(line1, line2)
pure = DifferentiableSGP4Pure()

# Propagate for a short time to diagnose
tsince_minutes = 1.0

print("--- Comparing SGP4 Implementations ---")
print(f"Propagating for {tsince_minutes} minute(s).\n")

# --- Wrapper Implementation --- 
print("--- Wrapper (sgp4 library) ---")
pos_wrapper, vel_wrapper = wrapper(tsince_minutes)
print(f"Position (ECI): {pos_wrapper.detach().numpy()}")
print(f"Velocity (ECI): {vel_wrapper.detach().numpy()}\n")

# --- Pure PyTorch Implementation --- 
print("--- Pure PyTorch ---")
pos_pure, vel_pure = pure.propagate(line1, line2, tsince_minutes)
print(f"Position (ECI): {pos_pure.detach().numpy()}")
print(f"Velocity (ECI): {vel_pure.detach().numpy()}\n")

# --- Final Comparison ---
diff = torch.norm(pos_wrapper - pos_pure)
print("--- Comparison ---")
print(f"Position difference: {diff:.3f} km")

if diff > 1.0:
    print("⚠️ Implementations diverge")
else:
    print("✅ Implementations match!")
