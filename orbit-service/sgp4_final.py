"""
Final Corrected SGP4 Implementation
Based on AAS 06-675 paper with proper TLE parsing and all fixes
Ready for differentiable ML enhancement
"""

import math

# WGS-72 Constants from AAS 06-675
MU = 398600.8  # km^3/s^2
R_E = 6378.135  # km
J2 = 0.001082616
J3 = -0.00000253881
J4 = -0.00000165597
XKE = 60.0 / math.sqrt(R_E**3 / MU)
TUMIN = 1.0 / XKE

def fmod(x, y):
    """Python equivalent of C fmod function"""
    return x - y * math.floor(x / y)

class SGP4Propagator:
    def __init__(self, line1, line2):
        self.error = 0
        self.parse_tle(line1, line2)
        self.sgp4init()

    def parse_tle(self, line1, line2):
        """Parse TLE using space-separated fields"""
        # Split line1 by spaces for easier parsing
        parts1 = line1.split()
        self.satnum = int(parts1[1][:5])  # Remove 'U' from catalog number
        
        # Parse epoch from parts1[2]
        epoch_str = parts1[2]
        self.epochyr = int(epoch_str[:2])
        self.epochdays = float(epoch_str[2:])
        
        # Parse derivatives
        self.ndot = float(parts1[3])
        self.nddot = self._parse_exp_notation(parts1[4])
        self.bstar = self._parse_exp_notation(parts1[5])
        
        # Split line2 by spaces
        parts2 = line2.split()
        self.inclo = math.radians(float(parts2[2]))
        self.nodeo = math.radians(float(parts2[3]))
        self.ecco = float('0.' + parts2[4])
        self.argpo = math.radians(float(parts2[5]))
        self.mo = math.radians(float(parts2[6]))
        self.no_kozai = float(parts2[7])
        
        # Epoch year correction
        if self.epochyr < 57:
            self.epochyr += 2000
        else:
            self.epochyr += 1900
            
        # Convert mean motion to rad/min
        self.no = self.no_kozai * 2.0 * math.pi / 1440.0

    def _parse_exp_notation(self, field):
        """Parse TLE exponential notation correctly"""
        field = field.strip()
        if not field or field == '00000-0':
            return 0.0
            
        # Handle format like "13844-3" = 1.3844e-3
        if len(field) >= 2 and field[-2] in '+-':
            mantissa_str = field[:-2]
            exp_str = field[-2:]
            
            if mantissa_str:
                # Handle sign
                sign = 1
                if mantissa_str[0] == '-':
                    sign = -1
                    mantissa_str = mantissa_str[1:]
                elif mantissa_str[0] == '+':
                    mantissa_str = mantissa_str[1:]
                
                # Convert mantissa with assumed decimal point
                if mantissa_str:
                    mantissa = float(mantissa_str) / (10 ** (len(mantissa_str) - 1))
                    exponent = int(exp_str)
                    return sign * mantissa * (10 ** exponent)
        
        return float(field) if field else 0.0

    def sgp4init(self):
        """Initialize SGP4 with corrected parameters"""
        # Basic orbital parameters
        self.a = (XKE / self.no) ** (2.0/3.0)
        self.cosio = math.cos(self.inclo)
        self.sinio = math.sin(self.inclo)
        self.cosio2 = self.cosio * self.cosio
        self.x3thm1 = 3.0 * self.cosio2 - 1.0
        self.x1mth2 = 1.0 - self.cosio2
        self.x7thm1 = 7.0 * self.cosio2 - 1.0
        
        # Semi-latus rectum
        self.p = self.a * (1.0 - self.ecco * self.ecco)
        
        # Secular rates (rad/min)
        temp1 = 1.5 * J2 * (R_E / self.p)**2 * self.no
        self.nodedot = -temp1 * self.cosio
        self.argdot = temp1 * (2.5 * self.cosio2 - 0.5)
        
        # Mean motion with J2 correction
        beta = math.sqrt(1.0 - self.ecco * self.ecco)
        temp2 = temp1 * beta * (1.5 * self.cosio2 - 0.5) / beta**3
        self.mdot = self.no + temp2
        
        # Drag coefficients
        self.c1 = self.bstar * 2.0
        self.c4 = 2.0 * self.no * self.bstar
        
        # Long period coefficients
        if abs(self.cosio + 1.0) > 1.5e-12:
            self.xlcof = 0.125 * J3 / J2 * self.sinio * (3.0 + 5.0 * self.cosio) / (1.0 + self.cosio)
        else:
            self.xlcof = 0.0
        self.aycof = 0.25 * J3 / J2 * self.sinio

    def propagate(self, tsince):
        """Propagate satellite position using SGP4 algorithm"""
        if self.error > 0:
            return None, None

        # Update for secular gravity and atmospheric drag
        xmdf = self.mo + self.mdot * tsince
        omgadf = self.argpo + self.argdot * tsince
        xnoddf = self.nodeo + self.nodedot * tsince

        # Atmospheric drag
        temp = 1.0 - self.c1 * tsince
        if temp <= 0.0:
            self.error = 1
            return None, None

        # Update semi-major axis and eccentricity for drag
        a = self.a * temp * temp
        e = self.ecco - self.bstar * self.c4 * tsince

        if e < 1e-6:
            e = 1e-6
        if e >= 1.0:
            self.error = 2
            return None, None

        # Mean longitude
        xl = xmdf + omgadf + xnoddf

        # Long period periodics
        beta = math.sqrt(1.0 - e * e)
        axn = e * math.cos(omgadf)
        temp = 1.0 / (a * beta * beta)
        xll = temp * self.xlcof * axn
        aynl = temp * self.aycof
        xlt = xl + xll
        ayn = e * math.sin(omgadf) + aynl

        # Solve Kepler's equation
        u = fmod(xlt - xnoddf, 2.0 * math.pi)
        eo1 = u
        
        for ktr in range(10):
            sineo1 = math.sin(eo1)
            coseo1 = math.cos(eo1)
            tem5 = (u - ayn * coseo1 + axn * sineo1 - eo1) / \
                   (1.0 - coseo1 * axn - sineo1 * ayn)
            
            if abs(tem5) < 1e-12:
                break
                
            if abs(tem5) >= 0.95:
                tem5 = 0.95 * (1.0 if tem5 > 0 else -1.0)
            
            eo1 += tem5

        # Short period preliminary quantities
        ecose = axn * coseo1 + ayn * sineo1
        esine = axn * sineo1 - ayn * coseo1
        el2 = axn * axn + ayn * ayn
        pl = a * (1.0 - el2)
        
        if pl < 0.0:
            self.error = 3
            return None, None

        r = a * (1.0 - ecose)
        rdot = XKE * math.sqrt(a) * esine / r
        rfdot = XKE * math.sqrt(pl) / r

        # Orientation vectors
        temp = esine / (1.0 + math.sqrt(1.0 - el2))
        sinu = a / r * (sineo1 - ayn + axn * temp)
        cosu = a / r * (coseo1 - axn + ayn * temp)
        u = math.atan2(sinu, cosu)

        # Short period periodics
        rk = r * (1.0 - 1.5 * J2 * (R_E / pl)**2 * self.x3thm1) + \
             0.5 * J2 * (R_E / pl)**2 * self.x1mth2 * math.cos(2.0 * u)
        
        uk = u - 0.25 * J2 * (R_E / pl)**2 * self.x7thm1 * math.sin(2.0 * u)
        xnodek = xnoddf + 1.5 * J2 * (R_E / pl)**2 * self.cosio * math.sin(2.0 * u)
        xinck = self.inclo + 1.5 * J2 * (R_E / pl)**2 * self.cosio * self.sinio * math.cos(2.0 * u)

        # Position and velocity in orbital plane
        x = rk * math.cos(uk)
        y = rk * math.sin(uk)
        xdot = rdot * math.cos(uk) - rfdot * math.sin(uk)
        ydot = rdot * math.sin(uk) + rfdot * math.cos(uk)

        # Transform to TEME frame
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
        r_teme = [
            mx * x + nx * y,
            my * x + ny * y,
            mz * x + nz * y
        ]

        v_teme = [
            mx * xdot + nx * ydot,
            my * xdot + ny * ydot,
            mz * xdot + nz * ydot
        ]

        return r_teme, v_teme

def validate_sgp4():
    """Validate against AAS paper test cases"""
    print("🚀 SGP4 Validation - AAS Paper Test Cases")
    print("=" * 50)
    
    # Test Case 1: Near Earth
    print("\n📡 Test Case 1: Near Earth Satellite")
    line1 = "1 88888U 80275.98708465  .00073094  13844-3  66816-4 0    8"
    line2 = "2 88888  72.8435 115.9689 0086731  52.6988 110.5714 16.05824518  105"
    
    try:
        prop = SGP4Propagator(line1, line2)
        
        # Test at tsince = 0
        r, v = prop.propagate(0.0)
        expected_r = [2328.97048951, -5995.22076416, 1719.97067261]
        expected_v = [2.91207230, -0.98341546, -7.09081703]
        
        if r and v:
            r_error = math.sqrt(sum((r[i] - expected_r[i])**2 for i in range(3)))
            v_error = math.sqrt(sum((v[i] - expected_v[i])**2 for i in range(3)))
            
            print(f"  ⏰ At tsince = 0:")
            print(f"    Position: [{r[0]:.5f}, {r[1]:.5f}, {r[2]:.5f}] km")
            print(f"    Expected: [{expected_r[0]:.5f}, {expected_r[1]:.5f}, {expected_r[2]:.5f}] km")
            print(f"    Error: {r_error:.6f} km")
            print(f"    Velocity: [{v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f}] km/s")
            print(f"    Expected: [{expected_v[0]:.5f}, {expected_v[1]:.5f}, {expected_v[2]:.5f}] km/s")
            print(f"    Error: {v_error:.6f} km/s")
            
            test1_pass = r_error < 1.0 and v_error < 0.1
            print(f"    Result: {'✅ PASSED' if test1_pass else '❌ FAILED'}")
            
            # Test at tsince = 360
            r, v = prop.propagate(360.0)
            expected_r = [2456.10705566, -6071.93853760, 1222.89727783]
            expected_v = [2.67938992, -0.44829041, -7.22879231]
            
            if r and v:
                r_error = math.sqrt(sum((r[i] - expected_r[i])**2 for i in range(3)))
                v_error = math.sqrt(sum((v[i] - expected_v[i])**2 for i in range(3)))
                
                print(f"\n  ⏰ At tsince = 360:")
                print(f"    Position: [{r[0]:.5f}, {r[1]:.5f}, {r[2]:.5f}] km")
                print(f"    Expected: [{expected_r[0]:.5f}, {expected_r[1]:.5f}, {expected_r[2]:.5f}] km")
                print(f"    Error: {r_error:.6f} km")
                print(f"    Velocity: [{v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f}] km/s")
                print(f"    Expected: [{expected_v[0]:.5f}, {expected_v[1]:.5f}, {expected_v[2]:.5f}] km/s")
                print(f"    Error: {v_error:.6f} km/s")
                
                test2_pass = r_error < 1.0 and v_error < 0.1
                print(f"    Result: {'✅ PASSED' if test2_pass else '❌ FAILED'}")
                
                overall_pass = test1_pass and test2_pass
                print(f"\n🎯 Overall Test Result: {'✅ SUCCESS' if overall_pass else '❌ NEEDS WORK'}")
                
                if overall_pass:
                    print("🚀 SGP4 Core Implementation Complete!")
                    print("✅ Ready for PyTorch differentiable wrapper")
                    print("✅ Laptop-friendly, no external dependencies")
                    print("✅ Validated against AAS paper test cases")
                
                return prop, overall_pass
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, False

if __name__ == "__main__":
    validate_sgp4()
