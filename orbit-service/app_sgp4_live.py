"""
Live SGP4 Orbit Microservice - Direct SGP4 Implementation
Making SGP4 live with proven sgp4 library
"""

import os
import asyncio
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import traceback

import redis
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sgp4.api import Satrec
import requests
from pydantic import BaseModel, Field
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global configuration
class OrbitServiceConfig:
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    CELESTRAK_BASE = os.getenv('CELESTRAK_API_BASE', 'https://celestrak.org')
    CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))  # 1 hour
    MAX_SATELLITES = int(os.getenv('MAX_SATELLITES_RENDER', '20000'))
    UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL_MS', '30000')) // 1000
    THREAT_THRESHOLD_KM = float(os.getenv('CONJUNCTION_THRESHOLD_KM', '10.0'))
    AUTONOMOUS_MODE = os.getenv('AUTO_TOKEN_ROTATION', 'true').lower() == 'true'

config = OrbitServiceConfig()

# Initialize Redis
try:
    redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connection established")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

# Initialize thread pool
executor = ThreadPoolExecutor(max_workers=8)

class SatelliteData(BaseModel):
    """Satellite data model with validation"""
    norad_id: int
    name: str
    line1: str
    line2: str
    epoch: datetime
    mean_motion: float
    eccentricity: float
    inclination: float
    arg_perigee: float
    raan: float
    mean_anomaly: float

class OrbitPosition(BaseModel):
    """Orbital position model"""
    timestamp: datetime
    latitude: float
    longitude: float
    altitude_km: float
    velocity_kms: float
    position_km: List[float]
    velocity_vector_kms: List[float]
    sgp4_error: int

class LiveSGP4Tracker:
    """Live SGP4 satellite tracking using direct sgp4 library"""
    
    def __init__(self):
        self.satellites_cache = {}
        self.tle_data = {}
        self.last_update = None
        
    async def fetch_tle_data(self, source: str = 'active') -> Dict[int, SatelliteData]:
        """Fetch TLE data from CELESTRAK"""
        try:
            sources = {
                'active': f'{config.CELESTRAK_BASE}/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
                'starlink': f'{config.CELESTRAK_BASE}/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle',
                'gps': f'{config.CELESTRAK_BASE}/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle',
                'weather': f'{config.CELESTRAK_BASE}/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle',
                'science': f'{config.CELESTRAK_BASE}/NORAD/elements/gp.php?GROUP=science&FORMAT=tle',
            }
            
            url = sources.get(source, sources['active'])
            
            # Check cache first
            cache_key = f"tle_data:{source}"
            cached_data = redis_client.get(cache_key) if redis_client else None
            
            if cached_data:
                logger.info(f"Using cached TLE data for {source}")
                return json.loads(cached_data)
            
            # Fetch fresh data
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            tle_lines = response.text.strip().split('\n')
            satellites = {}
            
            for i in range(0, len(tle_lines), 3):
                if i + 2 < len(tle_lines):
                    name = tle_lines[i].strip()
                    line1 = tle_lines[i + 1].strip()
                    line2 = tle_lines[i + 2].strip()
                    
                    try:
                        # Parse NORAD ID
                        norad_id = int(line1[2:7])
                        
                        # Parse orbital elements
                        epoch_year = int(line1[18:20])
                        epoch_day = float(line1[20:32])
                        
                        # Convert epoch to datetime
                        year = 2000 + epoch_year if epoch_year < 57 else 1900 + epoch_year
                        epoch = datetime(year, 1, 1) + timedelta(days=epoch_day - 1)
                        
                        # Extract orbital elements
                        inclination = float(line2[8:16])
                        raan = float(line2[17:25])
                        eccentricity = float('0.' + line2[26:33])
                        arg_perigee = float(line2[34:42])
                        mean_anomaly = float(line2[43:51])
                        mean_motion = float(line2[52:63])
                        
                        satellite_data = SatelliteData(
                            norad_id=norad_id,
                            name=name,
                            line1=line1,
                            line2=line2,
                            epoch=epoch,
                            mean_motion=mean_motion,
                            eccentricity=eccentricity,
                            inclination=inclination,
                            arg_perigee=arg_perigee,
                            raan=raan,
                            mean_anomaly=mean_anomaly
                        )
                        
                        satellites[norad_id] = satellite_data.dict()
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse TLE for {name}: {e}")
                        continue
            
            # Cache the results
            if redis_client:
                redis_client.setex(cache_key, config.CACHE_TTL, json.dumps(satellites))
            
            logger.info(f"Fetched {len(satellites)} satellites from {source}")
            return satellites
            
        except Exception as e:
            logger.error(f"Failed to fetch TLE data from {source}: {e}")
            return {}
    
    def calculate_position(self, satellite_data: SatelliteData, 
                          timestamp: Optional[datetime] = None) -> OrbitPosition:
        """Calculate satellite position using direct SGP4"""
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            
            # Create SGP4 satellite object
            satellite = Satrec.twoline2rv(satellite_data.line1, satellite_data.line2)
            
            # Convert timestamp to Julian date
            jd, fr = self._datetime_to_jd(timestamp)
            
            # Propagate
            error, position, velocity = satellite.sgp4(jd, fr)
            
            if error != 0:
                logger.warning(f"SGP4 error {error} for satellite {satellite_data.norad_id}")
            
            # Convert to lat/lon/alt
            lat, lon, alt = self._ecef_to_geodetic(position)
            
            # Calculate velocity magnitude
            velocity_magnitude = math.sqrt(sum(v**2 for v in velocity))
            
            return OrbitPosition(
                timestamp=timestamp,
                latitude=lat,
                longitude=lon,
                altitude_km=alt,
                velocity_kms=velocity_magnitude,
                position_km=list(position),
                velocity_vector_kms=list(velocity),
                sgp4_error=error
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate position for {satellite_data.name}: {e}")
            raise
    
    def _datetime_to_jd(self, dt):
        """Convert datetime to Julian date"""
        if dt.tzinfo is not None:
            dt = dt.utctimetuple()
        else:
            dt = dt.timetuple()
            
        year, month, day = dt.tm_year, dt.tm_mon, dt.tm_mday
        hour, minute, second = dt.tm_hour, dt.tm_min, dt.tm_sec
        
        # Julian day calculation
        if month <= 2:
            year -= 1
            month += 12
            
        a = int(year / 100)
        b = 2 - a + int(a / 4)
        
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        
        # Fractional part
        fr = (hour + minute/60.0 + second/3600.0) / 24.0
        
        return jd, fr
    
    def _ecef_to_geodetic(self, position):
        """Convert ECEF position to geodetic coordinates"""
        x, y, z = position
        
        # WGS84 constants
        a = 6378137.0  # meters
        f = 1/298.257223563
        e2 = 2*f - f*f
        
        # Convert km to meters
        x_m, y_m, z_m = x*1000, y*1000, z*1000
        
        # Longitude
        lon = math.atan2(y_m, x_m)
        
        # Latitude (iterative)
        p = math.sqrt(x_m*x_m + y_m*y_m)
        lat = math.atan2(z_m, p * (1 - e2))
        
        for _ in range(5):
            N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
            h = p / math.cos(lat) - N
            lat = math.atan2(z_m, p * (1 - e2 * N / (N + h)))
            
        # Altitude
        N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        alt = p / math.cos(lat) - N
        
        return math.degrees(lat), math.degrees(lon), alt/1000

# Initialize tracker
tracker = LiveSGP4Tracker()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        redis_status = "healthy" if redis_client and redis_client.ping() else "unhealthy"
        
        data_freshness = "unknown"
        if tracker.last_update:
            age = datetime.now(timezone.utc) - tracker.last_update
            data_freshness = "fresh" if age.total_seconds() < 3600 else "stale"
        
        status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0-live-sgp4",
            "sgp4_engine": "direct",
            "services": {
                "redis": redis_status,
                "data_freshness": data_freshness,
                "satellites_loaded": len(tracker.tle_data)
            },
            "configuration": {
                "autonomous_mode": config.AUTONOMOUS_MODE,
                "max_satellites": config.MAX_SATELLITES,
                "threat_threshold_km": config.THREAT_THRESHOLD_KM
            }
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 503

@app.route('/satellites', methods=['GET'])
def get_satellites():
    """Get all satellite data"""
    try:
        source = request.args.get('source', 'active')
        limit = min(int(request.args.get('limit', config.MAX_SATELLITES)), config.MAX_SATELLITES)
        
        satellites = asyncio.run(tracker.fetch_tle_data(source))
        limited_satellites = dict(list(satellites.items())[:limit])
        
        return jsonify({
            "satellites": limited_satellites,
            "count": len(limited_satellites),
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sgp4_engine": "direct"
        })
        
    except Exception as e:
        logger.error(f"Failed to get satellites: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/satellites/<int:norad_id>/position', methods=['GET'])
def get_satellite_position(norad_id: int):
    """Get current position of a specific satellite"""
    try:
        timestamp_str = request.args.get('timestamp')
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = datetime.now(timezone.utc)
        
        satellites = asyncio.run(tracker.fetch_tle_data('active'))
        
        if norad_id not in satellites:
            return jsonify({"error": "Satellite not found"}), 404
        
        satellite_data = SatelliteData(**satellites[norad_id])
        position = tracker.calculate_position(satellite_data, timestamp)
        
        return jsonify(position.dict())
        
    except Exception as e:
        logger.error(f"Failed to get satellite position: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sgp4/validate', methods=['GET'])
def validate_sgp4():
    """Validate SGP4 implementation"""
    try:
        # Test with reference satellite
        line1 = "1 06251U 62025A   06176.82412014  .00002182  00000-0  13103-3 0  6091"
        line2 = "2 06251  58.0579  54.0425 0002329  75.6910 284.4861 14.84479601804021"
        
        # Create satellite and propagate at epoch
        satellite = Satrec.twoline2rv(line1, line2)
        error, position, velocity = satellite.sgp4(satellite.jdsatepoch, satellite.jdsatepochF)
        
        # Expected values from reference
        expected_pos = [-907, 4655, 4404]
        expected_vel = [-7.45, -2.15, 0.92]
        
        pos_error = math.sqrt(sum((p - e)**2 for p, e in zip(position, expected_pos)))
        vel_error = math.sqrt(sum((v - e)**2 for v, e in zip(velocity, expected_vel)))
        
        validation_result = {
            "satellite_id": "06251",
            "test_type": "reference_validation",
            "computed_position": list(position),
            "expected_position": expected_pos,
            "position_error_km": pos_error,
            "computed_velocity": list(velocity),
            "expected_velocity": expected_vel,
            "velocity_error_kms": vel_error,
            "sgp4_error_code": error,
            "sgp4_engine": "direct",
            "meets_2km_target": pos_error < 2.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return jsonify(validation_result)
        
    except Exception as e:
        logger.error(f"SGP4 validation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logger.error(f"Unhandled error: {error}\n{traceback.format_exc()}")
    return jsonify({
        "error": "Internal server error",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 500

if __name__ == '__main__':
    logger.info("Starting Live SGP4 Orbit Microservice")
    app.run(host='0.0.0.0', port=5000, debug=False)
