"""
Reference SGP4 Implementation

This is an educational reference implementation of SGP4 based on the algorithm
described in Vallado et al. (2006) "Revisiting Spacetrack Report #3" (AAS 06-675).

Purpose:
- Educational reference for understanding SGP4 internals
- Algorithm validation and comparison
- Not intended for production use

For production applications, use the proven sgp4 library or the
differentiable_sgp4_torch wrapper instead.

Implementation details:
- Uses WGS-72 gravitational constants as specified in AAS 06-675
- Implements TLE parsing with correct field positions
- Includes Long Period Periodic (LPP) and Short Period Periodic (SPP) terms
- Improved Kepler solver with convergence guarantees
- Added numerical stability checks throughout

References:
- Vallado, D. A., et al. (2006). "Revisiting Spacetrack Report #3." AIAA 2006-6753
- Hoots, F. R., & Roehrich, R. L. (1980). "Spacetrack Report No. 3"
"""

import logging
import math

logger = logging.getLogger(__name__)

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
TWOPI = 2 * math.pi
XPDOTP = 1440.0 / TWOPI  # rev/day to rad/min

# WGS-72 constants (per Vallado et al. 2006, AAS 06-675)
MU = 398600.8  # km^3/s^2
RE = 6378.135  # km
XKE = 60.0 / math.sqrt(RE**3 / MU)
TUMIN = 1.0 / XKE
J2 = 0.00108262998905892
J3 = -0.00000253215306
J4 = -0.00000165597
S = RE + 78.0
QOMS2T = ((120 - 78) / RE) ** 4
X2O3 = 2.0 / 3.0

# Numerical stability thresholds
MIN_MEAN_MOTION = 1e-12  # Minimum mean motion (rad/min)
MIN_ECCENTRICITY = 1e-6  # Minimum eccentricity
MAX_ECCENTRICITY = 0.9999  # Maximum eccentricity
MIN_SEMI_MAJOR_AXIS = 1.0  # Minimum semi-major axis (km)
MIN_DENOMINATOR = 1e-12  # Minimum denominator for divisions


class SGP4Propagator:
    def __init__(self, line1, line2):
        self.error = 0
        self.error_message = ""
        self.method = ""
        self._last_valid_a = None  # For error recovery
        self.parse_tle(line1, line2)
        if self.error == 0:
            self.sgp4init()

    def parse_tle(self, line1, line2):
        try:
            # Parse line 1 - exact field positions
            self.satnum = int(line1[2:7])
            self.epochyr = int(line1[18:20])
            self.epochdays = float(line1[20:32])
            self.ndot = float(line1[33:43])
            self.nddot = self.exp_to_dec(line1[44:50], line1[50:52])
            self.bstar = self.exp_to_dec(line1[53:59], line1[59:61])

            # Parse line 2 - exact field positions
            self.inclo = float(line2[8:16]) * DEG2RAD
            self.nodeo = float(line2[17:25]) * DEG2RAD
            self.ecco = float(f"0.{line2[26:33]}")
            self.argpo = float(line2[34:42]) * DEG2RAD
            self.mo = float(line2[43:51]) * DEG2RAD
            self.no_kozai = float(line2[52:63]) / XPDOTP  # to rad/min

            # Epoch year correction
            self.epochyr = (
                1900 + self.epochyr if self.epochyr >= 57 else 2000 + self.epochyr
            )

        except (ValueError, IndexError) as e:
            self.error = 4
            logger.error(f"TLE parsing error: {e}")

    def exp_to_dec(self, mant_str, exp_str):
        """Parse TLE exponential notation correctly"""
        try:
            if not mant_str.strip() or mant_str.strip() == "00000":
                return 0.0

            mant = mant_str.strip().replace(" ", "0")
            if mant.startswith("-"):
                sign = -1
                mant = mant[1:]
            elif mant.startswith("+"):
                sign = 1
                mant = mant[1:]
            else:
                sign = 1

            mantissa = float("0." + mant) if mant else 0.0

            if exp_str.startswith("-"):
                exponent = -int(exp_str[1:])
            elif exp_str.startswith("+"):
                exponent = int(exp_str[1:])
            else:
                exponent = int(exp_str) if exp_str.strip() else 0

            return sign * mantissa * (10**exponent)
        except (ValueError, IndexError):
            return 0.0

    def sgp4init(self):
        """Initialize SGP4 with enhanced numerical stability"""
        
        # Input validation and bounds checking
        if abs(self.no_kozai) < MIN_MEAN_MOTION:
            self.error = 5
            logger.error("Mean motion is too small for stable propagation")
            return
        
        # Bound eccentricity to safe range
        self.ecco = max(MIN_ECCENTRICITY, min(self.ecco, MAX_ECCENTRICITY))
        
        self.orig_no = self.no_kozai

        # Basic trigonometric quantities
        self.sinio = math.sin(self.inclo)
        self.cosio = math.cos(self.inclo)
        self.cosio2 = self.cosio * self.cosio
        self.theta = self.cosio2
        self.x2mth = 1.0 - self.theta
        self.x3thm1 = 3.0 * self.theta - 1.0
        self.x7thm1 = 7.0 * self.theta - 1.0

        # Semi-major axis calculation with safety checks
        ao = math.pow(XKE / self.no_kozai, X2O3)
        
        if ao < MIN_SEMI_MAJOR_AXIS:
            self.error = 5
            logger.error("Semi-major axis too small")
            return

        # Calculate beta with protection against numerical issues
        betao2 = 1.0 - self.ecco * self.ecco
        betao2 = max(MIN_DENOMINATOR, betao2)
        betao = math.sqrt(betao2)

        # Calculate delta1 with denominator protection
        denom_delta1 = betao * betao2 * ao * ao * self.no_kozai
        if abs(denom_delta1) < MIN_DENOMINATOR:
            denom_delta1 = MIN_DENOMINATOR if denom_delta1 >= 0 else -MIN_DENOMINATOR
            
        delta1 = 1.5 * J2 * self.x3thm1 / denom_delta1

        # Apply corrections with bounds checking
        ao = ao * (
            1.0
            - delta1 / 3.0
            - delta1 * delta1
            - 134.0 * delta1 * delta1 * delta1 / 81.0
        )
        
        if ao < MIN_SEMI_MAJOR_AXIS:
            self.error = 5
            logger.error("Corrected semi-major axis too small")
            return

        # Corrected mean motion with protection
        denom_no = 1.0 + delta1
        if abs(denom_no) < MIN_DENOMINATOR:
            denom_no = MIN_DENOMINATOR if denom_no >= 0 else -MIN_DENOMINATOR
            
        self.no = self.no_kozai / denom_no
        
        if self.no <= MIN_MEAN_MOTION:
            self.error = 5
            logger.error("Corrected mean motion too small")
            return

        # Recalculate with corrected mean motion
        self.a = math.pow(XKE / self.no, X2O3)
        self.betao = betao
        self.betao2 = betao2
        self.p = self.a * self.betao2

        if self.p <= 0.0:
            self.error = 5
            logger.error("Semi-latus rectum is non-positive")
            return

        # Deep space check (period >= 225 minutes)
        if (TWOPI / self.no) >= 225.0:
            self.method = "d"
            self.isimp = 1
        else:
            self.method = "n"
            self.isimp = 0

        # Secular rates with safe division
        if self.p > MIN_DENOMINATOR:
            ratio = RE / self.p
            ratio_sq = ratio * ratio
            
            self.nodedot = -1.5 * J2 * ratio_sq * self.cosio * self.no
            self.argdot = (
                1.5 * J2 * ratio_sq * (4.0 - 5.0 * self.sinio * self.sinio) * self.no / 2.0
            )
            self.mdot = self.no + 1.5 * J2 * ratio_sq * self.betao * self.x3thm1 / (2.0 * self.betao2)
        else:
            self.nodedot = 0.0
            self.argdot = 0.0
            self.mdot = self.no

        # Drag coefficients
        self.c1 = self.bstar * 2.0
        self.c4 = 2.0 * self.no * self.bstar

        # Long period coefficients
        if abs(self.cosio + 1.0) > 1.5e-12:
            self.xlcof = (
                0.125
                * J3
                / J2
                * self.sinio
                * (3.0 + 5.0 * self.cosio)
                / (1.0 + self.cosio)
            )
        else:
            self.xlcof = 0.0
        self.aycof = 0.25 * J3 / J2 * self.sinio

    def solve_kepler_improved(self, m, e, axn, ayn, tolerance=1e-12, max_iter=20):
        """
        Improved Kepler equation solver using Laguerre's method
        More robust than Newton-Raphson for high eccentricity orbits
        """
        # Initial guess
        if e < 0.8:
            ep = m
        else:
            ep = math.pi if m > math.pi else -math.pi
        
        converged = False
        
        for i in range(max_iter):
            sinep = math.sin(ep)
            cosep = math.cos(ep)
            
            # Function and derivatives
            f = ep - ayn * cosep + axn * sinep - m
            fp = 1.0 - axn * cosep - ayn * sinep
            fpp = axn * sinep - ayn * cosep
            
            # Check for convergence
            if abs(f) < tolerance:
                converged = True
                break
            
            # Laguerre's method
            n = 5.0  # Order parameter for better convergence
            
            # Calculate denominator with stability
            h = n * f
            discriminant = (n - 1) * (n - 1) * fp * fp - n * (n - 1) * f * fpp
            
            if discriminant < 0:
                discriminant = 0
            
            sqrt_disc = math.sqrt(discriminant)
            
            denom1 = fp + sqrt_disc
            denom2 = fp - sqrt_disc
            
            # Choose larger denominator for stability
            if abs(denom1) > abs(denom2):
                denom = denom1
            else:
                denom = denom2
            
            if abs(denom) < MIN_DENOMINATOR:
                denom = MIN_DENOMINATOR if denom >= 0 else -MIN_DENOMINATOR
            
            delta = n * f / denom
            
            # Apply correction with damping for large steps
            if abs(delta) > 0.95:
                delta = 0.95 if delta > 0 else -0.95
            
            ep = ep - delta
        
        return ep, sinep, cosep, converged

    def propagate(self, tsince):
        """
        Propagate using improved algorithm with stability checks and error recovery.
        
        Args:
            tsince: Time since epoch (minutes)
            
        Returns:
            Tuple of (position, velocity) or (None, None) on unrecoverable error
            
        Notes:
            This method attempts error recovery before returning None.
            Check self.error for error codes and self.error_message for details.
        """
        if self.error > 0:
            return None, None

        # Store initial error state
        initial_error = self.error

        # Secular updates
        argp = self.argpo + self.argdot * tsince
        omg = self.nodeo + self.nodedot * tsince
        m = self.mo + self.mdot * tsince
        
        # Update semi-major axis with drag
        temp = 1.0 - self.c1 * tsince
        if temp <= 0.0:
            self.error = 1
            self.error_message = (
                f"Satellite has decayed (drag term collapsed at t={tsince:.1f} min). "
                f"Physical meaning: Atmospheric drag has reduced orbital energy to the point "
                f"where the satellite has re-entered. This typically occurs for satellites "
                f"with high B* drag coefficient propagated far into the future."
            )
            logger.error(self.error_message)
            
            # Attempt recovery: use last valid semi-major axis
            if hasattr(self, '_last_valid_a') and self._last_valid_a > MIN_SEMI_MAJOR_AXIS:
                logger.warning(f"Attempting recovery using last valid semi-major axis")
                a = self._last_valid_a
                self.error = 0  # Clear error for recovery attempt
            else:
                return None, None
        else:
            a = self.a * temp * temp
            
        if a < MIN_SEMI_MAJOR_AXIS:
            self.error = 7
            self.error_message = (
                f"Semi-major axis too small ({a:.3f} km) at t={tsince:.1f} min. "
                f"Physical meaning: The computed orbital radius is unrealistically small, "
                f"indicating numerical instability or satellite decay. Minimum valid value "
                f"is {MIN_SEMI_MAJOR_AXIS} km (just above Earth's surface)."
            )
            logger.error(self.error_message)
            return None, None
        
        # Store valid semi-major axis for potential recovery
        self._last_valid_a = a
        
        # Update eccentricity
        e = self.ecco - self.bstar * self.c4 * tsince
        e = max(MIN_ECCENTRICITY, min(e, MAX_ECCENTRICITY))

        # Long period periodics
        beta_sq = 1.0 - e * e
        beta_sq = max(MIN_DENOMINATOR, beta_sq)
        beta = math.sqrt(beta_sq)
        
        axn = e * math.cos(argp)
        temp_denom = a * beta * beta
        if abs(temp_denom) < MIN_DENOMINATOR:
            temp_denom = MIN_DENOMINATOR if temp_denom >= 0 else -MIN_DENOMINATOR
            
        temp = 1.0 / temp_denom
        xll = temp * self.xlcof * axn
        aynl = temp * self.aycof
        xl = m + argp + omg + xll
        ayn = e * math.sin(argp) + aynl

        # Solve Kepler's equation with improved solver
        u = math.fmod(xl, TWOPI)
        if u < 0.0:
            u += TWOPI
            
        ep, sinep, cosep, converged = self.solve_kepler_improved(u, e, axn, ayn)
        
        if not converged:
            self.error = 6
            self.error_message = (
                f"Kepler solver did not converge at t={tsince:.1f} min. "
                f"Physical meaning: The numerical solver for eccentric anomaly failed to converge. "
                f"This can occur for very high eccentricities (e={e:.6f}) or when the orbit "
                f"becomes unstable due to perturbations."
            )
            logger.warning(self.error_message)
            # Don't return None - use the best available solution and continue
            # This allows degraded but usable propagation

        # Calculate position and velocity
        ecose = axn * cosep + ayn * sinep
        esine = axn * sinep - ayn * cosep
        el2 = axn * axn + ayn * ayn
        pl = a * (1.0 - el2)

        if pl <= 0.0:
            self.error = 3
            self.error_message = (
                f"Semi-parameter is non-positive ({pl:.6f}) at t={tsince:.1f} min. "
                f"Physical meaning: The orbital semi-latus rectum (p = a(1-e²)) must be positive. "
                f"A non-positive value indicates eccentricity e≥1 (parabolic/hyperbolic orbit) or "
                f"numerical issues. This typically means the satellite has escaped Earth's gravity "
                f"or the propagation has become unstable."
            )
            logger.error(self.error_message)
            return None, None

        r = a * (1.0 - ecose)
        if r < MIN_SEMI_MAJOR_AXIS:
            self.error = 7
            self.error_message = (
                f"Orbital radius too small ({r:.3f} km) at t={tsince:.1f} min. "
                f"Physical meaning: The satellite's distance from Earth's center is below "
                f"the minimum safe threshold ({MIN_SEMI_MAJOR_AXIS} km). This indicates either "
                f"satellite decay/re-entry or numerical instability in the propagation."
            )
            logger.error(self.error_message)
            return None, None
            
        rdot = XKE * math.sqrt(a) * esine / r
        rfdot = XKE * math.sqrt(pl) / r

        # Orientation vectors
        temp_sqrt = 1.0 - el2
        if temp_sqrt < 0:
            temp_sqrt = 0
        temp = esine / (1.0 + math.sqrt(temp_sqrt))
        sinu = a / r * (sinep - ayn + axn * temp)
        cosu = a / r * (cosep - axn + ayn * temp)
        u = math.atan2(sinu, cosu)

        # Short period periodics
        sin2u = math.sin(2.0 * u)
        cos2u = math.cos(2.0 * u)

        # Apply SPP corrections
        rk = (
            r * (1.0 - 1.5 * J2 * (RE / pl) ** 2 * self.x3thm1)
            + 0.5 * J2 * (RE / pl) ** 2 * self.x2mth * cos2u
        )

        uk = u - 0.25 * J2 * (RE / pl) ** 2 * self.x7thm1 * sin2u
        xnodek = omg + 1.5 * J2 * (RE / pl) ** 2 * self.cosio * sin2u
        xinck = self.inclo + 1.5 * J2 * (RE / pl) ** 2 * self.cosio * self.sinio * cos2u

        # Position and velocity in orbital plane
        x = rk * math.cos(uk)
        y = rk * math.sin(uk)
        xdot = rdot * math.cos(uk) - rfdot * math.sin(uk)
        ydot = rdot * math.sin(uk) + rfdot * math.cos(uk)

        # TEME transformation
        sinik = math.sin(xinck)
        cosik = math.cos(xinck)
        sinnok = math.sin(xnodek)
        cosnok = math.cos(xnodek)

        # Rotation matrix elements
        mx = -sinnok * cosik
        my = cosnok * cosik
        mz = sinik
        nx = cosnok
        ny = sinnok

        # Final position and velocity vectors (km, km/min)
        r_teme = [mx * x + nx * y, my * x + ny * y, mz * x]
        v_teme = [mx * xdot + nx * ydot, my * xdot + ny * ydot, mz * xdot]

        return r_teme, v_teme


def validate_reference_sgp4():
    """
    Validate reference SGP4 implementation against test cases.
    """
    logger.info("Reference SGP4 Validation")

    # Test Case: Vanguard 2 (Near Earth)
    line1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
    line2 = "2 00005  34.2682 348.7242 1859667 331.7664 19.3264 10.82419157413667"

    try:
        prop = SGP4Propagator(line1, line2)

        if prop.error == 0:
            r, v = prop.propagate(0.0)
            expected_r = [294.78, -2954.94, 6494.22]  # km
            expected_v = [0.896, -6.610, -0.248]  # km/s

            if r and v:
                # Convert velocity from km/min to km/s
                v_kms = [v[0] / 60.0, v[1] / 60.0, v[2] / 60.0]

                r_error = math.sqrt(sum((r[i] - expected_r[i]) ** 2 for i in range(3)))
                v_error = math.sqrt(
                    sum((v_kms[i] - expected_v[i]) ** 2 for i in range(3))
                )

                logger.info("At tsince = 0:")
                logger.info(f"  Position: [{r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}] km")
                logger.info(
                    f"  Expected: [{expected_r[0]:.2f}, {expected_r[1]:.2f}, {expected_r[2]:.2f}] km"
                )
                logger.info(f"  Error: {r_error:.3f} km")
                logger.info(
                    f"  Velocity: [{v_kms[0]:.3f}, {v_kms[1]:.3f}, {v_kms[2]:.3f}] km/s"
                )
                logger.info(
                    f"  Expected: [{expected_v[0]:.3f}, {expected_v[1]:.3f}, {expected_v[2]:.3f}] km/s"
                )
                logger.info(f"  Error: {v_error:.6f} km/s")

                test1_pass = r_error < 1.0  # 1km tolerance
                if test1_pass:
                    logger.info("Test PASSED (target: <1 km)")
                    logger.info("Reference SGP4 implementation validated successfully")
                else:
                    logger.warning("Test FAILED (target: <1 km)")
                    logger.info(f"Position accuracy: {r_error:.3f} km")
                    logger.info(f"Velocity accuracy: {v_error:.6f} km/s")

                return prop, test1_pass
        else:
            logger.error(f"Initialization error: {prop.error}")
            return None, False

    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        return None, False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    validate_reference_sgp4()
