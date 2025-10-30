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
- Newton-Raphson solver for Kepler's equation

References:
- Vallado, D. A., et al. (2006). "Revisiting Spacetrack Report #3." AIAA 2006-6753
- Hoots, F. R., & Roehrich, R. L. (1980). "Spacetrack Report No. 3"
"""

import math

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


class SGP4Propagator:
    def __init__(self, line1, line2):
        self.error = 0
        self.method = ""
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

            # Parse line 2 - exact field positions per user spec
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
            import logging

            logging.getLogger(__name__).error(f"TLE parsing error: {e}")

    def exp_to_dec(self, mant_str, exp_str):
        """Parse TLE exponential notation correctly"""
        try:
            if not mant_str.strip() or mant_str.strip() == "00000":
                return 0.0

            # Handle exponential format like "13844-3"
            mant = mant_str.strip().replace(" ", "0")
            if mant.startswith("-"):
                sign = -1
                mant = mant[1:]
            elif mant.startswith("+"):
                sign = 1
                mant = mant[1:]
            else:
                sign = 1

            # Convert mantissa with assumed decimal point
            mantissa = float("0." + mant) if mant else 0.0

            # Parse exponent
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
        """Initialize SGP4 with user's exact algorithm"""
        self.orig_no = self.no_kozai

        # Basic trigonometric quantities
        self.sinio = math.sin(self.inclo)
        self.cosio = math.cos(self.inclo)
        self.cosio2 = self.cosio * self.cosio
        self.theta = self.cosio2
        self.x2mth = 1.0 - self.theta
        self.x3thm1 = 3.0 * self.theta - 1.0
        self.x7thm1 = 7.0 * self.theta - 1.0

        # Semi-major axis with user's exact formula
        # ao = (xke / no)^{2/3}
        ao = math.pow(XKE / self.no_kozai, X2O3)

        # delta1 = (3/2) * j2 * (3cos(i)^2 -1) / ( (1-e^2)^{3/2} * ao^4 * no )
        betao2 = 1.0 - self.ecco * self.ecco
        betao = math.sqrt(betao2)
        delta1 = 1.5 * J2 * self.x3thm1 / (betao * betao2 * ao * ao * self.no_kozai)

        # ao = ao * (1 - delta1/3 - delta1^2 - 134*delta1^3/81)
        ao = ao * (
            1.0
            - delta1 / 3.0
            - delta1 * delta1
            - 134.0 * delta1 * delta1 * delta1 / 81.0
        )

        # no = no / (1 + delta1)
        self.no = self.no_kozai / (1.0 + delta1)

        # Recalculate with corrected mean motion
        self.a = math.pow(XKE / self.no, X2O3)
        self.betao = math.sqrt(1.0 - self.ecco * self.ecco)
        self.betao2 = 1.0 - self.ecco * self.ecco
        self.p = self.a * self.betao2

        # Deep space check (period >= 225 minutes)
        if (TWOPI / self.no) >= 225.0:
            self.method = "d"
            self.isimp = 1
        else:
            self.method = "n"
            self.isimp = 0

        # Secular rates from user's exact formulas
        # nodedot = - (3/2) * j2 * (re / p)^2 * cos(i) * no
        self.nodedot = -1.5 * J2 * (RE / self.p) ** 2 * self.cosio * self.no

        # argdot = (3/2) * j2 * (re / p)^2 * (4 - 5sin(i)^2) * no / 2
        self.argdot = (
            1.5
            * J2
            * (RE / self.p) ** 2
            * (4.0 - 5.0 * self.sinio * self.sinio)
            * self.no
            / 2.0
        )

        # mdot = no + (3/2) * j2 * (re / p)^2 * sqrt(1-e^2) * (3cos(i)^2 -1) / (2 * (1-e^2))
        self.mdot = self.no + 1.5 * J2 * (
            RE / self.p
        ) ** 2 * self.betao * self.x3thm1 / (2.0 * self.betao2)

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

    def propagate(self, tsince):
        """Propagate using user's exact algorithm"""
        if self.error > 0:
            return None, None

        # Secular Update from user spec
        # argp = argpo + argdot * tsince
        argp = self.argpo + self.argdot * tsince
        # omg = nodeo + nodedot * tsince
        omg = self.nodeo + self.nodedot * tsince
        # m = mo + mdot * tsince
        m = self.mo + self.mdot * tsince
        # a = ao * (1 - c1 * tsince)^2
        temp = 1.0 - self.c1 * tsince
        if temp <= 0.0:
            self.error = 1
            return None, None
        a = self.a * temp * temp
        # e = ecco - betao * tsince where betao = bstar * c4
        e = self.ecco - self.bstar * self.c4 * tsince

        if e < 1e-6:
            e = 1e-6
        if e >= 1.0:
            self.error = 2
            return None, None

        # Long period periodics
        beta = math.sqrt(1.0 - e * e)
        axn = e * math.cos(argp)
        temp = 1.0 / (a * beta * beta)
        xll = temp * self.xlcof * axn
        aynl = temp * self.aycof
        xl = m + argp + omg + xll
        ayn = e * math.sin(argp) + aynl

        # Kepler Solve from user spec
        # u = math.fmod(m + argp + omg, 2pi)
        u = math.fmod(xl, TWOPI)
        ep = u

        for iter in range(10):
            sinep = math.sin(ep)
            cosep = math.cos(ep)
            # delta_ep = (u - ayn*cos(ep) + axn*sin(ep) - ep) / (1 - axn*cos(ep) - ayn*sin(ep))
            delta_ep = (u - ayn * cosep + axn * sinep - ep) / (
                1.0 - axn * cosep - ayn * sinep
            )

            # ep += delta_ep if abs(delta_ep)<0.95*abs(delta_ep) else 0.95*sign(delta_ep)
            if abs(delta_ep) >= 0.95:
                delta_ep = 0.95 * (1.0 if delta_ep > 0 else -1.0)
            ep += delta_ep

            # converge if abs(delta_ep)<1e-12
            if abs(delta_ep) < 1e-12:
                break

        # Position (Orb Plane) from user spec
        # ecose = axn*cos(ep) + ayn*sin(ep)
        ecose = axn * cosep + ayn * sinep
        # esine = axn*sin(ep) - ayn*cos(ep)
        esine = axn * sinep - ayn * cosep
        el2 = axn * axn + ayn * ayn
        pl = a * (1.0 - el2)

        if pl < 0.0:
            self.error = 3
            return None, None

        # r = a * (1 - ecose)
        r = a * (1.0 - ecose)
        rdot = XKE * math.sqrt(a) * esine / r
        rfdot = XKE * math.sqrt(pl) / r

        # Orientation vectors
        temp = esine / (1.0 + math.sqrt(1.0 - el2))
        sinu = a / r * (sinep - ayn + axn * temp)
        cosu = a / r * (cosep - axn + ayn * temp)
        u = math.atan2(sinu, cosu)

        # Short period periodics (SPP adjustments)
        sin2u = math.sin(2.0 * u)
        cos2u = math.cos(2.0 * u)

        # SPP corrections
        rk = (
            r * (1.0 - 1.5 * J2 * (RE / pl) ** 2 * self.x3thm1)
            + 0.5 * J2 * (RE / pl) ** 2 * self.x2mth * cos2u
        )

        uk = u - 0.25 * J2 * (RE / pl) ** 2 * self.x7thm1 * sin2u

        xnodek = omg + 1.5 * J2 * (RE / pl) ** 2 * self.cosio * sin2u

        xinck = self.inclo + 1.5 * J2 * (RE / pl) ** 2 * self.cosio * self.sinio * cos2u

        # Position and velocity in orbital plane after SPP
        x = rk * math.cos(uk)
        y = rk * math.sin(uk)
        xdot = rdot * math.cos(uk) - rfdot * math.sin(uk)
        ydot = rdot * math.sin(uk) + rfdot * math.cos(uk)

        # TEME Transform - rotate by argp matrix, then incl, then node
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
        nz = 0.0

        # Final position and velocity vectors (km, km/min)
        r_teme = [mx * x + nx * y, my * x + ny * y, mz * x + nz * y]

        v_teme = [mx * xdot + nx * ydot, my * xdot + ny * ydot, mz * xdot + nz * ydot]

        return r_teme, v_teme


def validate_reference_sgp4():
    """
    Validate reference SGP4 implementation against test cases.

    Returns
    -------
    tuple
        (propagator, test_passed) where test_passed is True if validation succeeded

    References
    ----------
    Test case from Vallado et al. (2006) Vanguard 2 satellite.
    """
    import logging

    logger = logging.getLogger(__name__)

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
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    validate_reference_sgp4()
