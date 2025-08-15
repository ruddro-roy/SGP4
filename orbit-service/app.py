"""
Orbit Microservice - Autonomous Satellite Tracking & Threat Analysis
Advanced orbital mechanics service using Skyfield, SGP4, and modern astronomical algorithms
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import traceback

import redis
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from skyfield.api import load, Topos
from skyfield.sgp4lib import EarthSatellite
from sgp4.api import Satrec, WGS84
from sgp4.io import twoline2rv
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

# Initialize Skyfield
ts = load.timescale()
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
    visible: bool
    next_pass: Optional[datetime] = None

class ThreatAssessment(BaseModel):
    """Threat assessment model"""
    primary_object: int
    secondary_object: int
    closest_approach_time: datetime
    minimum_distance_km: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    probability: float

class AutonomousSatelliteTracker:
    """Autonomous satellite tracking with advanced orbital mechanics"""
    
    def __init__(self):
        self.satellites_cache = {}
        self.tle_data = {}
        self.threat_assessments = []
        self.last_update = None
        
    async def fetch_tle_data(self, source: str = 'active') -> Dict[int, SatelliteData]:
        """Fetch TLE data from CELESTRAK with autonomous error handling"""
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
        """Calculate precise satellite position using Skyfield"""
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            
            # Create Skyfield satellite object
            satellite = EarthSatellite(
                satellite_data.line1, 
                satellite_data.line2, 
                satellite_data.name, 
                ts
            )
            
            # Get position at specified time
            t = ts.from_datetime(timestamp.replace(tzinfo=timezone.utc))
            geocentric = satellite.at(t)
            
            # Convert to lat/lon/alt
            subpoint = geocentric.subpoint()
            
            # Calculate velocity
            position = geocentric.position.km
            velocity_vector = geocentric.velocity.km_per_s
            velocity_magnitude = np.linalg.norm(velocity_vector)
            
            # Determine visibility (simplified)
            altitude_km = subpoint.elevation.km
            visible = altitude_km > 200  # Rough visibility threshold
            
            return OrbitPosition(
                timestamp=timestamp,
                latitude=subpoint.latitude.degrees,
                longitude=subpoint.longitude.degrees,
                altitude_km=altitude_km,
                velocity_kms=velocity_magnitude,
                visible=visible
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate position for {satellite_data.name}: {e}")
            raise
    
    def predict_passes(self, satellite_data: SatelliteData, 
                      observer_lat: float, observer_lon: float, 
                      hours_ahead: int = 24) -> List[Dict]:
        """Predict satellite passes over observer location"""
        try:
            # Create observer location
            observer = Topos(observer_lat, observer_lon)
            
            # Create satellite
            satellite = EarthSatellite(
                satellite_data.line1, 
                satellite_data.line2, 
                satellite_data.name, 
                ts
            )
            
            # Calculate passes
            t0 = ts.now()
            t1 = ts.from_datetime(datetime.now(timezone.utc) + timedelta(hours=hours_ahead))
            
            times, events = satellite.find_events(observer, t0, t1, altitude_degrees=10.0)
            
            passes = []
            for time, event in zip(times, events):
                if event == 0:  # Rise
                    pass_info = {
                        'rise_time': time.utc_datetime(),
                        'event': 'rise'
                    }
                elif event == 1:  # Culmination
                    pass_info = {
                        'culmination_time': time.utc_datetime(),
                        'event': 'culmination'
                    }
                elif event == 2:  # Set
                    pass_info = {
                        'set_time': time.utc_datetime(),
                        'event': 'set'
                    }
                
                passes.append(pass_info)
            
            return passes
            
        except Exception as e:
            logger.error(f"Failed to predict passes for {satellite_data.name}: {e}")
            return []
    
    def assess_conjunction_threat(self, sat1_data: SatelliteData, 
                                 sat2_data: SatelliteData,
                                 time_window_hours: int = 24) -> Optional[ThreatAssessment]:
        """Assess conjunction threat between two satellites"""
        try:
            # Create satellite objects
            sat1 = EarthSatellite(sat1_data.line1, sat1_data.line2, sat1_data.name, ts)
            sat2 = EarthSatellite(sat2_data.line1, sat2_data.line2, sat2_data.name, ts)
            
            # Time range for analysis
            t0 = ts.now()
            t1 = ts.from_datetime(datetime.now(timezone.utc) + timedelta(hours=time_window_hours))
            
            # Sample positions over time
            times = ts.linspace(t0, t1, 1000)  # 1000 samples
            
            pos1 = sat1.at(times)
            pos2 = sat2.at(times)
            
            # Calculate distances
            diff = pos1.position.km - pos2.position.km
            distances = np.sqrt(np.sum(diff**2, axis=0))
            
            # Find minimum distance
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            closest_approach_time = times[min_distance_idx].utc_datetime()
            
            # Assess risk level
            if min_distance < 1.0:
                risk_level = "CRITICAL"
                probability = 0.9
            elif min_distance < 5.0:
                risk_level = "HIGH"
                probability = 0.7
            elif min_distance < config.THREAT_THRESHOLD_KM:
                risk_level = "MEDIUM"
                probability = 0.4
            else:
                risk_level = "LOW"
                probability = 0.1
            
            # Only return if within threshold
            if min_distance <= config.THREAT_THRESHOLD_KM:
                return ThreatAssessment(
                    primary_object=sat1_data.norad_id,
                    secondary_object=sat2_data.norad_id,
                    closest_approach_time=closest_approach_time,
                    minimum_distance_km=min_distance,
                    risk_level=risk_level,
                    probability=probability
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to assess conjunction threat: {e}")
            return None

# Initialize tracker
tracker = AutonomousSatelliteTracker()

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check Redis
        redis_status = "healthy" if redis_client and redis_client.ping() else "unhealthy"
        
        # Check last data update
        data_freshness = "unknown"
        if tracker.last_update:
            age = datetime.now(timezone.utc) - tracker.last_update
            data_freshness = "fresh" if age.total_seconds() < 3600 else "stale"
        
        status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "services": {
                "redis": redis_status,
                "data_freshness": data_freshness,
                "satellites_loaded": len(tracker.tle_data),
                "threats_active": len(tracker.threat_assessments)
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
    """Get all satellite data with optional filtering"""
    try:
        source = request.args.get('source', 'active')
        limit = min(int(request.args.get('limit', config.MAX_SATELLITES)), config.MAX_SATELLITES)
        
        # Fetch data
        satellites = asyncio.run(tracker.fetch_tle_data(source))
        
        # Apply limit
        limited_satellites = dict(list(satellites.items())[:limit])
        
        return jsonify({
            "satellites": limited_satellites,
            "count": len(limited_satellites),
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get satellites: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/satellites/<int:norad_id>/position', methods=['GET'])
def get_satellite_position(norad_id: int):
    """Get current position of a specific satellite"""
    try:
        # Get timestamp from query param
        timestamp_str = request.args.get('timestamp')
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = datetime.now(timezone.utc)
        
        # Find satellite in cache
        satellites = asyncio.run(tracker.fetch_tle_data('active'))
        
        if norad_id not in satellites:
            return jsonify({"error": "Satellite not found"}), 404
        
        satellite_data = SatelliteData(**satellites[norad_id])
        position = tracker.calculate_position(satellite_data, timestamp)
        
        return jsonify(position.dict())
        
    except Exception as e:
        logger.error(f"Failed to get satellite position: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/satellites/<int:norad_id>/passes', methods=['GET'])
def predict_satellite_passes(norad_id: int):
    """Predict satellite passes over observer location"""
    try:
        # Get observer coordinates
        lat = float(request.args.get('lat', 0))
        lon = float(request.args.get('lon', 0))
        hours = int(request.args.get('hours', 24))
        
        # Find satellite
        satellites = asyncio.run(tracker.fetch_tle_data('active'))
        
        if norad_id not in satellites:
            return jsonify({"error": "Satellite not found"}), 404
        
        satellite_data = SatelliteData(**satellites[norad_id])
        passes = tracker.predict_passes(satellite_data, lat, lon, hours)
        
        return jsonify({
            "passes": passes,
            "observer": {"latitude": lat, "longitude": lon},
            "prediction_window_hours": hours
        })
        
    except Exception as e:
        logger.error(f"Failed to predict passes: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/threats/conjunctions', methods=['GET'])
def assess_conjunction_threats():
    """Assess conjunction threats between satellites"""
    try:
        time_window = int(request.args.get('hours', 24))
        
        # Get active satellites
        satellites = asyncio.run(tracker.fetch_tle_data('active'))
        
        threats = []
        satellite_list = list(satellites.values())
        
        # Check pairs of satellites (limit for performance)
        for i in range(min(100, len(satellite_list))):
            for j in range(i + 1, min(100, len(satellite_list))):
                sat1_data = SatelliteData(**satellite_list[i])
                sat2_data = SatelliteData(**satellite_list[j])
                
                threat = tracker.assess_conjunction_threat(sat1_data, sat2_data, time_window)
                if threat:
                    threats.append(threat.dict())
        
        return jsonify({
            "threats": threats,
            "analysis_window_hours": time_window,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to assess threats: {e}")
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
    logger.info("Starting Orbit Microservice")
    app.run(host='0.0.0.0', port=5000, debug=False)