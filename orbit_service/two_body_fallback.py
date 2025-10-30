"""
Two-Body Propagation Fallback Module

Provides a simple Keplerian (two-body) orbital propagation as a fallback
when SGP4 propagation fails. This is less accurate than SGP4 but ensures
graceful degradation rather than complete failure.

The two-body problem assumes only gravitational force from Earth (no drag,
no perturbations). This is acceptable for short-term fallback propagation
when SGP4 encounters numerical issues.

References:
    Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications (4th ed.).
"""

import math
import numpy as np
from typing import Tuple, Optional


# Earth gravitational parameter (km^3/s^2)
MU_EARTH = 398600.4418


def sgp4_state_to_orbital_elements(r: np.ndarray, v: np.ndarray) -> dict:
    """
    Convert position and velocity to classical orbital elements.
    
    Args:
        r: Position vector [x, y, z] in km
        v: Velocity vector [vx, vy, vz] in km/s
        
    Returns:
        Dictionary of orbital elements:
        - a: semi-major axis (km)
        - e: eccentricity
        - i: inclination (rad)
        - raan: right ascension of ascending node (rad)
        - argp: argument of perigee (rad)
        - nu: true anomaly (rad)
        - M: mean anomaly (rad)
        - n: mean motion (rad/s)
    """
    r_vec = np.array(r)
    v_vec = np.array(v)
    
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)
    
    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h_mag = np.linalg.norm(h_vec)
    
    # Node vector
    k_vec = np.array([0, 0, 1])
    n_vec = np.cross(k_vec, h_vec)
    n_mag = np.linalg.norm(n_vec)
    
    # Eccentricity vector
    e_vec = ((v_mag**2 - MU_EARTH/r_mag) * r_vec - np.dot(r_vec, v_vec) * v_vec) / MU_EARTH
    e = np.linalg.norm(e_vec)
    
    # Specific orbital energy
    energy = v_mag**2 / 2 - MU_EARTH / r_mag
    
    # Semi-major axis
    if abs(e - 1.0) > 1e-10:  # Not parabolic
        a = -MU_EARTH / (2 * energy)
    else:
        a = float('inf')
    
    # Inclination
    i = math.acos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))
    
    # RAAN
    if n_mag > 1e-10:
        raan = math.acos(np.clip(n_vec[0] / n_mag, -1.0, 1.0))
        if n_vec[1] < 0:
            raan = 2 * math.pi - raan
    else:
        raan = 0.0
    
    # Argument of perigee
    if n_mag > 1e-10 and e > 1e-10:
        argp = math.acos(np.clip(np.dot(n_vec, e_vec) / (n_mag * e), -1.0, 1.0))
        if e_vec[2] < 0:
            argp = 2 * math.pi - argp
    else:
        argp = 0.0
    
    # True anomaly
    if e > 1e-10:
        nu = math.acos(np.clip(np.dot(e_vec, r_vec) / (e * r_mag), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0:
            nu = 2 * math.pi - nu
    else:
        nu = 0.0
    
    # Mean anomaly and mean motion
    if e < 1.0:  # Elliptical orbit
        E = 2 * math.atan(math.sqrt((1-e)/(1+e)) * math.tan(nu/2))
        M = E - e * math.sin(E)
        n = math.sqrt(MU_EARTH / abs(a)**3)
    else:
        M = 0.0
        n = 0.0
    
    return {
        'a': a,
        'e': e,
        'i': i,
        'raan': raan,
        'argp': argp,
        'nu': nu,
        'M': M,
        'n': n,
        'h_mag': h_mag,
    }


def solve_kepler_equation(M: float, e: float, tolerance: float = 1e-12, max_iter: int = 50) -> float:
    """
    Solve Kepler's equation for eccentric anomaly.
    
    Args:
        M: Mean anomaly (rad)
        e: Eccentricity
        tolerance: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Eccentric anomaly E (rad)
    """
    # Initial guess
    if e < 0.8:
        E = M
    else:
        E = math.pi if M > math.pi else -math.pi
    
    # Newton-Raphson iteration
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        
        if abs(f) < tolerance:
            break
        
        if abs(fp) < 1e-12:
            break
            
        E = E - f / fp
    
    return E


def propagate_two_body(r0: np.ndarray, v0: np.ndarray, dt_seconds: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate orbit using two-body dynamics.
    
    Args:
        r0: Initial position vector [x, y, z] in km
        v0: Initial velocity vector [vx, vy, vz] in km/s
        dt_seconds: Time to propagate (seconds)
        
    Returns:
        Tuple of (position, velocity) at time dt
    """
    # Convert to orbital elements
    elements = sgp4_state_to_orbital_elements(r0, v0)
    
    a = elements['a']
    e = elements['e']
    i = elements['i']
    raan = elements['raan']
    argp = elements['argp']
    M0 = elements['M']
    n = elements['n']
    
    # Check for valid orbit
    if a <= 0 or e >= 1.0 or not np.isfinite(a):
        # Invalid orbit, return original state
        return r0, v0
    
    # Propagate mean anomaly
    M = M0 + n * dt_seconds
    M = M % (2 * math.pi)
    
    # Solve Kepler's equation
    E = solve_kepler_equation(M, e)
    
    # True anomaly
    nu = 2 * math.atan2(
        math.sqrt(1 + e) * math.sin(E/2),
        math.sqrt(1 - e) * math.cos(E/2)
    )
    
    # Distance
    r_mag = a * (1 - e * math.cos(E))
    
    # Position in orbital plane
    r_op = np.array([
        r_mag * math.cos(nu),
        r_mag * math.sin(nu),
        0.0
    ])
    
    # Velocity in orbital plane
    h = math.sqrt(MU_EARTH * a * (1 - e**2))
    v_op = np.array([
        -MU_EARTH / h * math.sin(nu),
        MU_EARTH / h * (e + math.cos(nu)),
        0.0
    ])
    
    # Rotation matrices
    cos_raan = math.cos(raan)
    sin_raan = math.sin(raan)
    cos_i = math.cos(i)
    sin_i = math.sin(i)
    cos_argp = math.cos(argp)
    sin_argp = math.sin(argp)
    
    # Perifocal to TEME (simplified, assumes TEME ~ ECI for this fallback)
    R_raan = np.array([
        [cos_raan, -sin_raan, 0],
        [sin_raan, cos_raan, 0],
        [0, 0, 1]
    ])
    
    R_i = np.array([
        [1, 0, 0],
        [0, cos_i, -sin_i],
        [0, sin_i, cos_i]
    ])
    
    R_argp = np.array([
        [cos_argp, -sin_argp, 0],
        [sin_argp, cos_argp, 0],
        [0, 0, 1]
    ])
    
    # Combined rotation
    R = R_raan @ R_i @ R_argp
    
    # Transform to inertial frame
    r = R @ r_op
    v = R @ v_op
    
    return r, v


class TwoBodyFallback:
    """
    Fallback propagator using two-body dynamics.
    
    This class provides a simple interface for propagating orbits when
    SGP4 fails. It's less accurate but ensures graceful degradation.
    """
    
    def __init__(self, r0: np.ndarray, v0: np.ndarray):
        """
        Initialize fallback propagator.
        
        Args:
            r0: Initial position vector [x, y, z] in km
            v0: Initial velocity vector [vx, vy, vz] in km/s
        """
        self.r0 = np.array(r0)
        self.v0 = np.array(v0)
        self.elements = sgp4_state_to_orbital_elements(r0, v0)
        
    def propagate(self, dt_seconds: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate orbit to time dt.
        
        Args:
            dt_seconds: Time to propagate (seconds)
            
        Returns:
            Tuple of (position, velocity) in km and km/s
        """
        return propagate_two_body(self.r0, self.v0, dt_seconds)
    
    def get_elements(self) -> dict:
        """Get initial orbital elements."""
        return self.elements.copy()


def test_two_body_propagation():
    """Test two-body propagation."""
    print("Testing Two-Body Propagation")
    print("=" * 50)
    
    # ISS-like orbit at periapsis (circular approximation)
    # Use realistic ISS orbital parameters
    a = 6778.0  # Semi-major axis in km
    v_circular = math.sqrt(MU_EARTH / a)  # Circular orbital velocity
    
    r0 = np.array([a, 0.0, 0.0])  # km
    v0 = np.array([0.0, v_circular, 0.0])  # km/s
    
    print(f"Initial state:")
    print(f"  Position: [{r0[0]:.1f}, {r0[1]:.1f}, {r0[2]:.1f}] km")
    print(f"  Velocity: [{v0[0]:.3f}, {v0[1]:.3f}, {v0[2]:.3f}] km/s")
    print(f"  Radius: {np.linalg.norm(r0):.1f} km")
    print(f"  Speed: {np.linalg.norm(v0):.3f} km/s")
    
    fallback = TwoBodyFallback(r0, v0)
    elements = fallback.get_elements()
    
    print(f"\nOrbital elements:")
    print(f"  Semi-major axis: {elements['a']:.1f} km")
    print(f"  Eccentricity: {elements['e']:.6f}")
    print(f"  Inclination: {math.degrees(elements['i']):.2f} deg")
    print(f"  Period: {2*math.pi/elements['n']/60:.2f} min")
    
    # Propagate for a shorter time to test
    dt = 3600.0  # 1 hour in seconds
    
    print(f"\nPropagating for {dt/60:.1f} minutes...")
    r, v = fallback.propagate(dt)
    
    print(f"Final position: [{r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}] km")
    print(f"Final velocity: [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}] km/s")
    print(f"Final radius: {np.linalg.norm(r):.1f} km")
    print(f"Final speed: {np.linalg.norm(v):.3f} km/s")
    
    # Check conservation of energy and angular momentum
    E0 = np.linalg.norm(v0)**2 / 2 - MU_EARTH / np.linalg.norm(r0)
    E1 = np.linalg.norm(v)**2 / 2 - MU_EARTH / np.linalg.norm(r)
    
    h0 = np.linalg.norm(np.cross(r0, v0))
    h1 = np.linalg.norm(np.cross(r, v))
    
    print(f"\nConservation checks:")
    print(f"  Energy change: {abs(E1 - E0):.6e} km^2/s^2")
    print(f"  Angular momentum change: {abs(h1 - h0):.6e} km^2/s")
    
    if abs(E1 - E0) < 1e-6 and abs(h1 - h0) < 1e-6:
        print("✓ Test PASSED - Energy and momentum conserved")
    else:
        print("✗ Test FAILED - Conservation violated")


if __name__ == "__main__":
    test_two_body_propagation()
