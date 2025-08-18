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
from sgp4.api import Satrec
from sgp4 import exporter
import requests
from pydantic import BaseModel, Field
import structlog
import torch
from differentiable_sgp4_torch import DifferentiableSGP4
from skyfield.api import load, Topos, EarthSatellite
from constants import ISS_TLE

# Initialize Skyfield timescale
ts = load.timescale()

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
        try:
            self.redis_client = redis.from_url(config.REDIS_URL)
            self.redis_client.ping()
            logger.info("Redis connection successful.")
        except (redis.exceptions.RedisError, AttributeError) as e:
            logger.warning(f"Redis connection failed or not configured: {e}. Caching will be disabled.")
            self.redis_client = None
        
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
            try:
                cached_data = self.redis_client.get(cache_key) if self.redis_client else None
            except redis.exceptions.ConnectionError as e:
                logger.warning(f"Redis connection failed: {e}. Caching will be disabled.")
                self.redis_client = None
                cached_data = None
            
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
            if self.redis_client:
                try:
                    self.redis_client.setex(cache_key, config.CACHE_TTL, json.dumps(satellites))
                except redis.exceptions.RedisError as e:
                    logger.warning(f"Could not cache TLE data to Redis: {e}")
            
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

@app.route('/differentiable-sgp4/demo', methods=['GET'])
def differentiable_sgp4_demo():
    """Interactive demo of differentiable SGP4 with gradient computation"""
    try:
        # Get parameters from request
        norad_id = request.args.get('norad_id', '25544')  # Default to ISS
        time_points = int(request.args.get('time_points', 10))
        max_time_hours = float(request.args.get('max_time_hours', 24))
        
        # Get satellite data
        satellites = {}
        try:
            satellites = asyncio.run(tracker.fetch_tle_data('active'))
        except Exception as e:
            logger.warning(f"TLE fetch failed, using ISS fallback: {e}")
        norad_id_int = int(norad_id)

        if norad_id_int not in satellites:
            if norad_id_int == 25544:
                logger.warning("Using hardcoded ISS TLE fallback for demo endpoint.")
                satellite_data = ISS_TLE
            else:
                return jsonify({"error": f"Satellite {norad_id} not found and no fallback available"}), 404
        else:
            satellite_data = satellites[norad_id_int]
        
        # Create differentiable SGP4 instance
        dsgp4 = DifferentiableSGP4(satellite_data['line1'], satellite_data['line2'])
        
        # Generate time points for demonstration
        max_time_minutes = max_time_hours * 60
        time_values = torch.linspace(0, max_time_minutes, time_points, requires_grad=True)
        
        # Propagate positions
        positions = []
        velocities = []
        gradients = []
        
        for i, t in enumerate(time_values):
            # Forward pass
            r, v = dsgp4(t)
            
            # Calculate orbital radius (distance from Earth center)
            radius = torch.norm(r)
            
            # Compute gradient of radius w.r.t. time
            radius.backward(retain_graph=True)
            gradient = t.grad.item() if t.grad is not None else 0.0
            
            # Store results
            positions.append({
                'time_minutes': t.item(),
                'time_hours': t.item() / 60.0,
                'position_km': r.detach().numpy().tolist(),
                'velocity_km_s': v.detach().numpy().tolist(),
                'orbital_radius_km': radius.item(),
                'radius_gradient': gradient,
                'altitude_km': radius.item() - 6378.137  # Earth radius
            })
            
            # Clear gradients for next iteration
            if t.grad is not None:
                t.grad.zero_()
        
        # Test ML corrections
        dsgp4.train()  # Enable training mode for corrections
        corrected_positions = []
        
        for i, t in enumerate(time_values):
            r_corrected, v_corrected = dsgp4(t)
            corrected_positions.append({
                'time_minutes': t.item(),
                'position_km': r_corrected.detach().numpy().tolist(),
                'velocity_km_s': v_corrected.detach().numpy().tolist(),
                'orbital_radius_km': torch.norm(r_corrected).item()
            })
        
        dsgp4.eval()  # Disable training mode
        
        # Calculate correction magnitudes
        correction_analysis = []
        for i in range(len(positions)):
            baseline_pos = np.array(positions[i]['position_km'])
            corrected_pos = np.array(corrected_positions[i]['position_km'])
            correction_magnitude = np.linalg.norm(corrected_pos - baseline_pos)
            
            correction_analysis.append({
                'time_minutes': positions[i]['time_minutes'],
                'correction_magnitude_km': correction_magnitude,
                'correction_vector_km': (corrected_pos - baseline_pos).tolist()
            })
        
        # Batch propagation test
        batch_times = torch.tensor([0.0, 360.0, 720.0, 1080.0, 1440.0])
        r_batch, v_batch = dsgp4.propagate_batch(batch_times)
        
        batch_results = []
        for i, t in enumerate(batch_times):
            batch_results.append({
                'time_minutes': t.item(),
                'position_km': r_batch[i].detach().numpy().tolist(),
                'velocity_km_s': v_batch[i].detach().numpy().tolist(),
                'orbital_radius_km': torch.norm(r_batch[i]).item()
            })
        
        # Satellite metadata
        satellite_info = {
            'norad_id': norad_id_int,
            'name': satellite_data['name'],
            'epoch': satellite_data['epoch'],
            'mean_motion': satellite_data['mean_motion'],
            'inclination_deg': satellite_data['inclination'],
            'eccentricity': satellite_data['eccentricity']
        }
        
        return jsonify({
            'satellite_info': satellite_info,
            'demo_parameters': {
                'time_points': time_points,
                'max_time_hours': max_time_hours,
                'differentiable': True,
                'ml_corrections_enabled': True
            },
            'baseline_propagation': positions,
            'corrected_propagation': corrected_positions,
            'correction_analysis': correction_analysis,
            'batch_propagation': batch_results,
            'capabilities': {
                'gradient_computation': True,
                'ml_corrections': True,
                'batch_processing': True,
                'pytorch_autograd': True
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Differentiable SGP4 demo failed: {e}")
        import traceback
        traceback.print_exc()
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
    app.run(host='0.0.0.0', port=5001, debug=False)