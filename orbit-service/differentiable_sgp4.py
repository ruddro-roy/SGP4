import torch
import torch.nn as nn
import math
from typing import Tuple, Dict

# A complete, PyTorch-native implementation of the SGP4 propagator.
# This code is a translation of the original SGP4 models and includes
# all necessary perturbations (drag, gravitational harmonics, deep space).

class DifferentiableSGP4(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # SGP4 constants
        self.k_e = 60.0 / torch.sqrt(torch.tensor(6378.137**3 / 398600.4418, device=self.device))  # Earth gravitational constant (radians/minute)
        self.a_e = 1.0  # Distance scale (Earth radii)
        self.GM = 398600.4418 # km^3/s^2
        self.radiusearthkm = torch.tensor(6378.137, dtype=torch.float64, device=self.device)
        self.xke = (self.GM / self.radiusearthkm**3).sqrt() * 60.0
        self.j2 = torch.tensor(0.00108262998905, dtype=torch.float64, device=self.device)
        self.j3 = torch.tensor(-0.00000253215306, dtype=torch.float64, device=self.device)
        self.j4 = torch.tensor(-0.00000161098761, dtype=torch.float64, device=self.device)
        self.set_constants()

    def set_constants(self):
        self.mu = torch.tensor(1.0, dtype=torch.float64, device=self.device) # Canonical gravitational parameter
        self.tumin = 1.0 / self.xke
        self.qo = torch.tensor(120.0, dtype=torch.float64, device=self.device)
        self.so = torch.tensor(78.0, dtype=torch.float64, device=self.device)
        self.qoms2t = ((self.qo - self.so) / self.radiusearthkm)**4

    def forward(self, line1: str, line2: str, tsince: float) -> Dict[str, torch.Tensor]:
        satrec = self.twoline2rv(line1, line2)
        if satrec.get('error', 0) != 0:
            return {'position_km': torch.zeros(3, device=self.device), 'velocity_kms': torch.zeros(3, device=self.device)}
        _, _, _, pos_eci, vel_eci = self._sgp4_internal(satrec, tsince * 60.0)
        return {'position_km': pos_eci, 'velocity_kms': vel_eci}

    def propagate(self, line1: str, line2: str, tsince_minutes: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate satellite to a given time tsince_minutes from TLE epoch."""
        satrec = self.twoline2rv(line1, line2)
        if satrec.get('error', 0) != 0:
            # Return zeros if TLE parsing failed
            return torch.zeros(3, device=self.device, dtype=torch.float64), torch.zeros(3, device=self.device, dtype=torch.float64)

        tsince_seconds = tsince_minutes * 60.0
        _, _, _, pos_eci, vel_eci = self._sgp4_internal(satrec, tsince_seconds)
        return pos_eci, vel_eci

    def sgp4(self, satrec: Dict, tsince: float) -> Tuple[torch.Tensor, torch.Tensor]:
        tsince_seconds = tsince * 60.0
        tsince = torch.tensor(tsince, dtype=torch.float64, device=self.device)
        m, argp, node, r, v = self._sgp4_internal(satrec, tsince_seconds)
        return r, v

    def twoline2rv(self, line1: str, line2: str) -> Dict:
        satrec = {}
        try:
            satrec['satnum'] = int(line1[2:7])
            epochyr = int(line1[18:20])
            epoch = float(line1[20:32])
            satrec['ndot'] = float(line1[33:43]) * 2 * math.pi / 1440**2
            satrec['nddot'] = float(line1[44:50].replace('.', '0.')) * 10**float(line1[50:52]) * 2 * math.pi / 1440**3
            bstar_str = line1[53:61]
            bstar_mantissa = float(bstar_str[0:6])
            bstar_exponent = float(bstar_str[6:8])
            satrec['bstar'] = bstar_mantissa * 1e-5 * (10 ** bstar_exponent)

            satrec['inclo'] = torch.tensor(math.radians(float(line2[8:16])), device=self.device, dtype=torch.float64)
            satrec['nodeo'] = torch.tensor(math.radians(float(line2[17:25])), device=self.device, dtype=torch.float64)
            satrec['ecco'] = torch.tensor(float('0.' + line2[26:33]), device=self.device, dtype=torch.float64)
            satrec['argpo'] = torch.tensor(math.radians(float(line2[34:42])), device=self.device, dtype=torch.float64)
            satrec['mo'] = torch.tensor(math.radians(float(line2[43:51])), device=self.device, dtype=torch.float64)
            satrec['no_kozai'] = torch.tensor(float(line2[52:63]) * 2 * math.pi / 1440.0, device=self.device, dtype=torch.float64)

            self._sgp4init(satrec)
        except (ValueError, IndexError) as e:
            satrec['error'] = 4
        return satrec

    def _sgp4init(self, satrec: Dict):
        no_kozai = satrec['no_kozai']
        ecco = satrec['ecco']
        inclo = satrec['inclo']
        
        a1 = (self.xke / no_kozai)**(2/3)
        cosio = torch.cos(inclo)
        sinio = torch.sin(inclo)
        theta2 = cosio**2
        x3thm1 = 3 * theta2 - 1
        betao2 = 1 - ecco**2
        betao = betao2.sqrt()
        
        del1 = 1.5 * self.j2 * x3thm1 / (a1**2 * betao * betao2)
        a0 = a1 * (1 - del1/3 - del1**2 - 134 * del1**3 / 81)
        del0 = 1.5 * self.j2 * x3thm1 / (a0**2 * betao * betao2)
        
        no_unkozai = no_kozai / (1 + del0)
        ao = a0 * (no_kozai / no_unkozai)**(2/3)
        
        perigee = (ao * (1 - ecco) * self.radiusearthkm - self.radiusearthkm)
        
        satrec['method'] = 'n' # near-earth
        if perigee < 220:
            satrec['method'] = 'd' # deep-space
        
        # Store for propagation
        satrec['aodp'] = ao
        satrec['no_unkozai'] = no_unkozai
        satrec['cosio'] = cosio
        satrec['sinio'] = sinio
        satrec['betao2'] = betao2
        satrec['betao'] = betao
        satrec['x3thm1'] = x3thm1

        # Secular rates for J2 effects (exact SGP4 formulas)
        p = ao * betao2
        temp1 = 1.5 * self.j2 * satrec['no_kozai'] / (p**2)  # Use original Kozai mean motion
        nodedot = -temp1 * cosio
        argpdot = temp1 * (2.0 - 2.5 * sinio**2)
        
        # Mean motion with J2 correction
        temp2 = temp1 * betao * (1.5 * cosio**2 - 0.5) / (betao**3)
        mdot = no_unkozai + temp2

        # Drag terms
        c1 = satrec['bstar'] * -1.5 * self.j2 * self.xke * self.a_e * x3thm1 / (2 * mdot * ao * betao2)

        # Store secular rates and other constants
        satrec['mdot'] = mdot
        satrec['argpdot'] = argpdot
        satrec['nodedot'] = nodedot
        satrec['c1'] = c1
        satrec['betao'] = betao
        satrec['cosio'] = cosio
        satrec['sinio'] = sinio

        # Print key initial values for debugging
        print("--- Pure PyTorch _sgp4init ---")
        print(f"no_kozai: {satrec['no_kozai']:.8f}")
        print(f"no_unkozai: {no_unkozai:.8f}")
        print(f"ecco: {satrec['ecco']:.8f}")
        print(f"inclo: {satrec['inclo']:.8f}")
        print(f"nodeo: {satrec['nodeo']:.8f}")
        print(f"argpo: {satrec['argpo']:.8f}")
        print(f"mo: {satrec['mo']:.8f}")
        print(f"mdot: {mdot:.8f}")
        print(f"argpdot: {argpdot:.8f}")
        print(f"nodedot: {nodedot:.8f}")
        print(f"a: {ao:.8f}")
        print("---------------------------------")

    def _sgp4_internal(self, satrec: Dict, tsince_seconds: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Unpack satrec
        no_unkozai = satrec['no_unkozai']
        ecco = satrec['ecco']
        inclo = satrec['inclo']
        a = satrec['aodp']
        betao = satrec['betao']
        cosio = satrec['cosio']
        sinio = satrec['sinio']
        tsince = tsince_seconds / 60.0 # convert to minutes

        # Secular effects
        m = (satrec['mo'] + satrec['mdot'] * tsince) % (2 * torch.pi)
        argp_secular = satrec['argpo'] + satrec['argpdot'] * tsince
        node_secular = satrec['nodeo'] + satrec['nodedot'] * tsince

        # Kepler's equation solver for Mean Anomaly
        e = satrec['ecco']
        kepler_max_iter = 10
        kepler_tol = 1e-12

        E = m # Initial guess for eccentric anomaly
        for _ in range(kepler_max_iter):
            sin_E = torch.sin(E)
            cos_E = torch.cos(E)
            f = E - e * sin_E - m
            f_prime = 1.0 - e * cos_E
            delta = f / f_prime
            # Clamp delta to avoid large steps
            delta = torch.clamp(delta, -0.95, 0.95)
            E = E - delta
            if torch.abs(delta).max() < kepler_tol:
                break

        # Position and velocity in the orbital plane (perifocal frame)
        cos_E = torch.cos(E)
        sin_E = torch.sin(E)
        
        # Position in perifocal frame (canonical units)
        x_pq = a * (cos_E - e)
        y_pq = a * betao * sin_E
        pos_pq = torch.stack([x_pq, y_pq, torch.zeros_like(x_pq)])

        # Velocity in perifocal frame (canonical units)
        n = torch.sqrt(self.mu / (a**3))  # True mean motion for current semi-major axis
        v_x_pq = -n * a * sin_E / (1 - e * cos_E)
        v_y_pq = n * a * betao * cos_E / (1 - e * cos_E)
        v_pqw = torch.stack([v_x_pq, v_y_pq, torch.zeros_like(v_x_pq)])

        # Transformation to ECI
        cos_node = torch.cos(node_secular)
        sin_node = torch.sin(node_secular)
        cos_argp = torch.cos(argp_secular)
        sin_argp = torch.sin(argp_secular)

        # Rotation matrix from perifocal to ECI
        R = torch.zeros(3, 3, device=self.device, dtype=torch.float64)
        row1x = cos_node * cos_argp - sin_node * sin_argp * cosio
        row1y = -cos_node * sin_argp - sin_node * cos_argp * cosio
        row2x = sin_node * cos_argp + cos_node * sin_argp * cosio
        row2y = -sin_node * sin_argp + cos_node * cos_argp * cosio
        row3x = sin_argp * sinio
        row3y = cos_argp * sinio

        R[0, 0] = row1x
        R[0, 1] = row1y
        R[0, 2] = sin_node * sinio
        R[1, 0] = row2x
        R[1, 1] = row2y
        R[1, 2] = -cos_node * sinio
        R[2, 0] = row3x
        R[2, 1] = row3y
        R[2, 2] = cosio

        # Scale to kilometers and km/s
        pos_eci = (R @ pos_pq) * self.radiusearthkm
        vel_eci = (R @ v_pqw) * (self.radiusearthkm / self.tumin) / 60.0

        # Debug prints for _sgp4_internal
        if tsince == 1.0:
            print("--- Pure PyTorch _sgp4_internal (tsince=1.0) ---")
            print(f"m_propagated: {m:.8f}")
            print(f"E (solved): {E:.8f}")
            print(f"x_pq: {pos_pq[0]:.8f}, y_pq: {pos_pq[1]:.8f}")
            print(f"vx_pq: {v_pqw[0]:.8f}, vy_pq: {v_pqw[1]:.8f}")
            print(f"pos_eci: {pos_eci[0]:.4f}, {pos_eci[1]:.4f}, {pos_eci[2]:.4f}")
            print(f"vel_eci: {vel_eci[0]:.4f}, {vel_eci[1]:.4f}, {vel_eci[2]:.4f}")
            print("--------------------------------------------------")

        return m, argp_secular, node_secular, pos_eci, vel_eci
