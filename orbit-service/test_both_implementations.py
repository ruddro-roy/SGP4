import torch
from differentiable_sgp4_torch import DifferentiableSGP4 as WrapperSGP4
from differentiable_sgp4 import DifferentiableSGP4 as PureSGP4

# Test TLE (ISS)
line1 = "1 25544U 98067A   23259.57580000  .00012022  00000-0  21844-3 0  9995"
line2 = "2 25544  51.6416 220.9944 0004263 122.0101 312.2755 15.49541986415598"

# Test both implementations
wrapper = WrapperSGP4(line1, line2)
pure = PureSGP4()

# Propagate 1 minute for detailed debugging
tsince_1min = torch.tensor(1.0)

# Wrapper version
r_wrapper_1min, v_wrapper_1min = wrapper(tsince_1min)

# Pure version
r_pure_1min, v_pure_1min = pure.propagate(line1, line2, 1.0)

# Detailed debug output for tsince=1.0 minute propagation
print("\nDetailed debug output for tsince=1.0 minute propagation:")
print(f"Wrapper position: {r_wrapper_1min.tolist()}")
print(f"Wrapper velocity: {v_wrapper_1min.tolist()}")
print(f"Pure position: {r_pure_1min.tolist()}")
print(f"Pure velocity: {v_pure_1min.tolist()}")

# Compare at 1 minute
difference_1min = torch.norm(r_wrapper_1min - r_pure_1min)
print(f"\nPosition difference at 1 minute: {difference_1min:.3f} km")

# Propagate 6 hours
tsince_6hr = torch.tensor(360.0)
r_wrapper, v_wrapper = wrapper(tsince_6hr)
r_pure, v_pure = pure.propagate(line1, line2, 360.0)

# Compare at 6 hours
difference = torch.norm(r_wrapper - r_pure)
print(f"\nPosition difference at 6 hours: {difference:.3f} km")

if difference < 10:
    print("✅ Implementations match!")
else:
    print("⚠️ Implementations diverge - check coordinate transforms")
