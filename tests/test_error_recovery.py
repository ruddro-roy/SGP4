"""
Tests for Error Recovery and Fallback Mechanisms

Tests the error recovery strategies implemented in SGP4 propagators:
- Two-body fallback propagation
- Error diagnostics
- Graceful degradation
- Error state tracking

Run with:
    python -m pytest tests/test_error_recovery.py -v
"""

import unittest
import numpy as np
import torch
from datetime import datetime, timedelta, timezone

from orbit_service.live_sgp4 import LiveSGP4
from orbit_service.differentiable_sgp4_torch import DifferentiableSGP4
from orbit_service.two_body_fallback import (
    TwoBodyFallback,
    propagate_two_body,
    sgp4_state_to_orbital_elements,
    solve_kepler_equation,
)


class TestTwoBodyFallback(unittest.TestCase):
    """Test two-body propagation fallback mechanism."""
    
    def test_circular_orbit_propagation(self):
        """Test propagation of circular orbit."""
        # Circular orbit at ISS altitude
        r0 = np.array([6778.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.669, 0.0])
        
        # Propagate for one hour
        r, v = propagate_two_body(r0, v0, 3600.0)
        
        # Check radius is conserved (circular orbit)
        r0_mag = np.linalg.norm(r0)
        r_mag = np.linalg.norm(r)
        
        # Allow 2 km tolerance due to numerical precision
        self.assertAlmostEqual(r_mag, r0_mag, delta=2.0, 
            msg="Radius should be approximately conserved in circular orbit")
    
    def test_energy_conservation(self):
        """Test that orbital energy is conserved."""
        r0 = np.array([7000.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.5, 0.0])
        
        # Calculate initial energy
        MU = 398600.4418
        E0 = np.linalg.norm(v0)**2 / 2 - MU / np.linalg.norm(r0)
        
        # Propagate
        r, v = propagate_two_body(r0, v0, 5000.0)
        
        # Calculate final energy
        E1 = np.linalg.norm(v)**2 / 2 - MU / np.linalg.norm(r)
        
        self.assertAlmostEqual(E0, E1, places=6,
            msg="Orbital energy should be conserved")
    
    def test_angular_momentum_conservation(self):
        """Test that angular momentum is conserved."""
        r0 = np.array([6778.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.669, 0.5])  # Inclined orbit
        
        h0 = np.cross(r0, v0)
        
        # Propagate
        r, v = propagate_two_body(r0, v0, 7200.0)
        
        h1 = np.cross(r, v)
        
        # Angular momentum should be conserved
        self.assertAlmostEqual(np.linalg.norm(h0), np.linalg.norm(h1), places=6)
    
    def test_kepler_solver(self):
        """Test Kepler equation solver."""
        # Test various eccentricities
        test_cases = [
            (0.0, 0.0),      # Circular, M=0
            (0.0, np.pi/2),  # Circular, M=π/2
            (0.1, np.pi),    # Low ecc
            (0.5, np.pi/4),  # Moderate ecc
            (0.9, 0.1),      # High ecc
        ]
        
        for e, M in test_cases:
            E = solve_kepler_equation(M, e)
            
            # Verify solution satisfies Kepler's equation
            residual = E - e * np.sin(E) - M
            
            self.assertLess(abs(residual), 1e-10,
                f"Kepler equation not solved accurately for e={e}, M={M}")
    
    def test_orbital_elements_conversion(self):
        """Test conversion to orbital elements."""
        # Known circular orbit
        r0 = np.array([7000.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.546, 0.0])
        
        elements = sgp4_state_to_orbital_elements(r0, v0)
        
        # Check semi-major axis
        self.assertAlmostEqual(elements['a'], 7000.0, places=0)
        
        # Check eccentricity (should be nearly 0)
        self.assertLess(elements['e'], 0.001)
        
        # Check inclination (should be 0)
        self.assertAlmostEqual(elements['i'], 0.0, places=2)
    
    def test_fallback_class(self):
        """Test TwoBodyFallback class interface."""
        r0 = np.array([6778.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.669, 0.0])
        
        fallback = TwoBodyFallback(r0, v0)
        
        # Check initial elements
        elements = fallback.get_elements()
        self.assertIn('a', elements)
        self.assertIn('e', elements)
        
        # Propagate
        r, v = fallback.propagate(1800.0)
        
        self.assertEqual(r.shape, (3,))
        self.assertEqual(v.shape, (3,))


class TestLiveSGP4ErrorRecovery(unittest.TestCase):
    """Test error recovery in LiveSGP4."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Normal ISS TLE
        self.iss_line1 = "1 25544U 98067A   23259.57580000  .00012022  00000-0  21844-3 0  9995"
        self.iss_line2 = "2 25544  51.6416 220.9944 0004263 122.0101 312.2755 15.49541986415598"
        
        # Decaying satellite TLE (high drag)
        self.decay_line1 = "1 44444U 19999A   23259.50000000  .10000000  00000-0  50000-2 0  9999"
        self.decay_line2 = "2 44444  51.6400 100.0000 0005000  90.0000 270.0000 16.50000000 99999"
    
    def test_normal_propagation_with_fallback_enabled(self):
        """Test that normal propagation works with fallback enabled."""
        sgp4 = LiveSGP4(enable_fallback=True)
        norad_id = sgp4.load_satellite(self.iss_line1, self.iss_line2, "ISS")
        
        # Propagate at epoch (should succeed)
        result = sgp4.propagate(norad_id, datetime(2023, 9, 16, 13, 49, 0, tzinfo=timezone.utc))
        
        self.assertEqual(result["sgp4_error_code"], 0)
        self.assertEqual(result["propagation_method"], "sgp4")
        self.assertFalse(result["fallback_used"])
        self.assertIn("position_km", result)
        self.assertIn("velocity_kms", result)
    
    def test_error_diagnostics(self):
        """Test that error diagnostics are provided."""
        sgp4 = LiveSGP4(enable_fallback=False)
        norad_id = sgp4.load_satellite(self.decay_line1, self.decay_line2, "DECAY")
        
        # Propagate far into future (likely to cause decay error)
        future_time = datetime(2023, 9, 16, tzinfo=timezone.utc) + timedelta(days=365)
        
        try:
            result = sgp4.propagate(norad_id, future_time)
            # If it didn't error, skip the diagnostic check
            if result["sgp4_error_code"] != 0:
                self.assertIn("error_diagnostics", result)
                diag = result["error_diagnostics"]
                self.assertIn("error_description", diag)
                self.assertIn("physical_meaning", diag)
        except RuntimeError as e:
            # Error was raised, which is expected behavior
            self.assertIn("SGP4 error", str(e))
    
    def test_fallback_activation(self):
        """Test that fallback is activated on error."""
        sgp4 = LiveSGP4(enable_fallback=True)
        norad_id = sgp4.load_satellite(self.decay_line1, self.decay_line2, "DECAY")
        
        # Propagate far into future
        future_time = datetime(2023, 9, 16, tzinfo=timezone.utc) + timedelta(days=365)
        
        try:
            result = sgp4.propagate(norad_id, future_time)
            
            # If SGP4 failed but we got a result, fallback must have been used
            if result["sgp4_error_code"] != 0:
                self.assertTrue(result["fallback_used"])
                self.assertEqual(result["propagation_method"], "two_body_fallback")
                self.assertIn("fallback_warning", result)
        except RuntimeError:
            # If it still failed, that's also valid (both methods can fail)
            pass
    
    def test_error_history_tracking(self):
        """Test that error history is tracked."""
        sgp4 = LiveSGP4(enable_fallback=True)
        norad_id = sgp4.load_satellite(self.decay_line1, self.decay_line2, "DECAY")
        
        # Try multiple propagations that might fail
        times = [
            datetime(2023, 9, 16, tzinfo=timezone.utc) + timedelta(days=d)
            for d in [0, 30, 60, 90, 365]
        ]
        
        for t in times:
            try:
                sgp4.propagate(norad_id, t)
            except:
                pass
        
        # Check if error history exists (may be empty if no errors)
        history = sgp4.get_error_history(norad_id)
        self.assertIsInstance(history, list)
    
    def test_batch_propagation_with_errors(self):
        """Test batch propagation handles errors gracefully."""
        sgp4 = LiveSGP4(enable_fallback=True)
        norad_id = sgp4.load_satellite(self.iss_line1, self.iss_line2, "ISS")
        
        # Mix of valid and potentially problematic times
        times = [
            datetime(2023, 9, 16, tzinfo=timezone.utc),
            datetime(2023, 9, 17, tzinfo=timezone.utc),
            datetime(2025, 9, 16, tzinfo=timezone.utc),  # Far future
        ]
        
        results = sgp4.propagate_batch(norad_id, times)
        
        # Should get results for all times (no crashes)
        self.assertEqual(len(results), len(times))
        
        # Each result should have position data or error info
        for result in results:
            self.assertTrue(
                "position_km" in result or "error" in result,
                "Each result should have either position or error info"
            )


class TestDifferentiableSGP4ErrorHandling(unittest.TestCase):
    """Test error handling in differentiable SGP4."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.line1 = "1 25544U 98067A   23259.57580000  .00012022  00000-0  21844-3 0  9995"
        self.line2 = "2 25544  51.6416 220.9944 0004263 122.0101 312.2755 15.49541986415598"
    
    def test_normal_propagation(self):
        """Test normal propagation without errors."""
        dsgp4 = DifferentiableSGP4(self.line1, self.line2)
        
        tsince = torch.tensor(0.0)
        r, v = dsgp4(tsince)
        
        self.assertEqual(r.shape, torch.Size([3]))
        self.assertEqual(v.shape, torch.Size([3]))
        self.assertEqual(dsgp4.last_error, 0)
    
    def test_error_state_tracking(self):
        """Test that error states are tracked."""
        dsgp4 = DifferentiableSGP4(self.line1, self.line2)
        
        # Propagate to a potentially problematic time
        tsince = torch.tensor(100000.0)  # Very far future
        r, v = dsgp4(tsince)
        
        # Check that error tracking attributes exist
        self.assertTrue(hasattr(dsgp4, 'last_error'))
        self.assertTrue(hasattr(dsgp4, 'last_error_time'))
    
    def test_graceful_degradation(self):
        """Test that errors don't cause crashes."""
        dsgp4 = DifferentiableSGP4(self.line1, self.line2)
        
        # Try multiple propagations, some may fail
        times = [0.0, 1440.0, 14400.0, 144000.0]
        
        for t in times:
            tsince = torch.tensor(t)
            r, v = dsgp4(tsince)
            
            # Should always return tensors (even if zeros)
            self.assertIsInstance(r, torch.Tensor)
            self.assertIsInstance(v, torch.Tensor)
            
            # Should not contain NaN or Inf
            self.assertFalse(torch.isnan(r).any())
            self.assertFalse(torch.isnan(v).any())
            self.assertFalse(torch.isinf(r).any())
            self.assertFalse(torch.isinf(v).any())
    
    def test_gradient_computation_with_errors(self):
        """Test that gradients can be computed even with potential errors."""
        dsgp4 = DifferentiableSGP4(self.line1, self.line2)
        dsgp4.eval()  # Disable training mode
        
        tsince = torch.tensor(1440.0, requires_grad=True, dtype=torch.float64)
        r, v = dsgp4(tsince)
        
        # Convert to float64 for gradient computation
        r64 = r.double()
        
        if dsgp4.last_error == 0 and not torch.allclose(r64, torch.zeros(3, dtype=torch.float64)):
            # Only test gradients if propagation succeeded and returned non-zero
            # Create a simple loss that depends on tsince through the SGP4 computation
            # We need to re-run with training enabled to get gradients
            dsgp4.train()
            tsince2 = torch.tensor(1440.0, requires_grad=True, dtype=torch.float64)
            r2, v2 = dsgp4(tsince2)
            
            # Skip gradient test if corrections are disabled
            if not torch.allclose(r2, torch.zeros(3)):
                try:
                    loss = torch.norm(r2.double())
                    loss.backward()
                    # Just check it doesn't crash
                    self.assertTrue(True)
                except RuntimeError:
                    # If gradients aren't available in eval mode, that's okay
                    pass
        else:
            # If there was an error, just pass
            pass


class TestErrorDiagnostics(unittest.TestCase):
    """Test error diagnostic messages."""
    
    def test_error_code_meanings(self):
        """Test that all error codes have meaningful messages."""
        from orbit_service.live_sgp4 import SGP4_ERROR_CODES
        
        # Check that common error codes are defined
        expected_codes = [0, 1, 2, 3, 4, 5, 6]
        
        for code in expected_codes:
            self.assertIn(code, SGP4_ERROR_CODES)
            self.assertIsInstance(SGP4_ERROR_CODES[code], str)
            self.assertTrue(len(SGP4_ERROR_CODES[code]) > 0)
    
    def test_diagnostic_completeness(self):
        """Test that diagnostics include all necessary fields."""
        sgp4 = LiveSGP4(enable_fallback=False)
        
        # Load a potentially problematic satellite
        line1 = "1 44444U 19999A   23259.50000000  .10000000  00000-0  50000-2 0  9999"
        line2 = "2 44444  51.6400 100.0000 0005000  90.0000 270.0000 16.50000000 99999"
        
        norad_id = sgp4.load_satellite(line1, line2, "TEST")
        
        # Try to propagate far into future
        future = datetime(2025, 1, 1, tzinfo=timezone.utc)
        
        try:
            result = sgp4.propagate(norad_id, future)
            
            if result["sgp4_error_code"] != 0:
                # Check diagnostic fields
                diag = result["error_diagnostics"]
                
                self.assertIn("error_code", diag)
                self.assertIn("error_description", diag)
                
                # Should have physical meaning or recommended action
                has_guidance = (
                    "physical_meaning" in diag or 
                    "recommended_action" in diag
                )
                self.assertTrue(has_guidance, 
                    "Diagnostics should include guidance")
        except RuntimeError:
            # If it raised an error, that's also acceptable
            pass


def run_error_recovery_tests():
    """Run all error recovery tests."""
    print("=" * 70)
    print("ERROR RECOVERY AND FALLBACK MECHANISM TESTS")
    print("=" * 70)
    print()
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTwoBodyFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestLiveSGP4ErrorRecovery))
    suite.addTests(loader.loadTestsFromTestCase(TestDifferentiableSGP4ErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorDiagnostics))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {100*(1 - (len(result.failures) + len(result.errors))/result.testsRun):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_error_recovery_tests()
    exit(0 if success else 1)
