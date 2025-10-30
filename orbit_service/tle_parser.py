    def teme_to_ecef_precise(self, r_teme: np.ndarray, v_teme: np.ndarray, epoch_datetime: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precise TEME to ECEF transformation with proper Earth rotation
        
        Args:
            r_teme: Position vector in TEME coordinates [x, y, z] (km)
            v_teme: Velocity vector in TEME coordinates [vx, vy, vz] (km/s)
            epoch_datetime: Epoch datetime for transformation
            
        Returns:
            Tuple of (r_ecef, v_ecef) in ECEF coordinates
        """
        # Calculate Julian centuries from J2000
        jd, fr = self.datetime_to_jd_fr(epoch_datetime)
        T = (jd - 2451545.0 + fr) / 36525.0
        
        # GMST with higher order terms (IAU 2000B)
        gmst_sec = (
            67310.54841 +
            (876600.0 * 3600.0 + 8640184.812866) * T +
            0.093104 * T * T -
            6.2e-6 * T * T * T
        )
        
        # Convert to radians and normalize
        gmst_rad = (gmst_sec % 86400.0) * (2.0 * math.pi / 86400.0)
        
        # Include equation of equinoxes for better accuracy
        omega = 125.04452 - 1934.136261 * T
        delta_psi = -0.000319 * math.sin(math.radians(omega))
        eqeq = delta_psi * math.cos(math.radians(23.4393))
        
        # Greenwich Apparent Sidereal Time
        gast = gmst_rad + eqeq
        
        # Rotation matrix from TEME to ECEF
        cos_gast = math.cos(gast)
        sin_gast = math.sin(gast)
        
        # Transform position
        r_ecef = np.array([
            cos_gast * r_teme[0] + sin_gast * r_teme[1],
            -sin_gast * r_teme[0] + cos_gast * r_teme[1],
            r_teme[2]
        ])
        
        # Transform velocity (includes Earth rotation rate)
        omega_earth = 7.2921159e-5  # rad/s
        
        v_ecef = np.array([
            cos_gast * v_teme[0] + sin_gast * v_teme[1] + omega_earth * r_ecef[1],
            -sin_gast * v_teme[0] + cos_gast * v_teme[1] - omega_earth * r_ecef[0],
            v_teme[2]
        ])
        
        return r_ecef, v_ecef

    def ecef_to_geodetic_precise(self, r_ecef: np.ndarray) -> Tuple[float, float, float]:
        """
        Accurate ECEF to geodetic conversion using Bowring's method
        
        Args:
            r_ecef: Position vector in ECEF coordinates [x, y, z] (km)
            
        Returns:
            Tuple of (latitude_deg, longitude_deg, altitude_km)
        """
        # WGS84 parameters
        a = 6378.137  # km
        f = 1.0 / 298.257223563
        b = a * (1.0 - f)
        e2 = 2.0 * f - f * f
        ep2 = e2 / (1.0 - e2)
        
        x, y, z = r_ecef
        
        # Longitude
        lon = math.atan2(y, x)
        
        # Distance from z-axis
        p = math.sqrt(x * x + y * y)
        
        # Handle pole cases
        if p < 1e-10:
            lat = math.pi / 2.0 if z > 0 else -math.pi / 2.0
            alt = abs(z) - b
            return math.degrees(lat), math.degrees(lon), alt
        
        # Initial estimate using Bowring's formula
        theta = math.atan2(z * a, p * b)
        
        # Iterate for latitude (usually converges in 2-3 iterations)
        for _ in range(5):
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            
            lat = math.atan2(
                z + ep2 * b * sin_theta * sin_theta * sin_theta,
                p - e2 * a * cos_theta * cos_theta * cos_theta
            )
            
            # Update theta for next iteration
            sin_lat = math.sin(lat)
            N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
            
            # Check convergence
            new_theta = math.atan2(z + e2 * N * sin_lat, p)
            if abs(new_theta - theta) < 1e-12:
                break
            theta = new_theta
        
        # Calculate altitude
        cos_lat = math.cos(lat)
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        
        if cos_lat > 1e-10:
            alt = p / cos_lat - N
        else:
            alt = z / sin_lat - N * (1.0 - e2)
        
        return math.degrees(lat), math.degrees(lon), alt
