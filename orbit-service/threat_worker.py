#!/usr/bin/env python3
"""
Threat Analysis Worker
Performs collision detection and space situational awareness analysis
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
import redis
from sgp4.api import Satrec, jday
from sgp4.conveniences import sat_epoch_datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('threat_worker.log')
    ]
)
logger = logging.getLogger(__name__)

# Environment configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
THREAT_ANALYSIS_ENABLED = os.getenv('THREAT_ANALYSIS_ENABLED', 'true').lower() == 'true'
COLLISION_THRESHOLD_KM = float(os.getenv('COLLISION_THRESHOLD_KM', '5.0'))
PREDICTION_DAYS = int(os.getenv('PREDICTION_DAYS', '7'))
ANALYSIS_INTERVAL = int(os.getenv('ANALYSIS_INTERVAL', '300'))  # 5 minutes

class ThreatAnalyzer:
    """Main class for satellite threat analysis"""
    
    def __init__(self):
        self.redis_client = self._init_redis()
        self.running = True
        self.satellites_cache = {}
        
    def _init_redis(self) -> redis.Redis:
        """Initialize Redis connection"""
        try:
            client = redis.from_url(REDIS_URL, decode_responses=True)
            client.ping()
            logger.info("Successfully connected to Redis")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
            
    def run_analysis_loop(self) -> None:
        """Main analysis loop"""
        logger.info("Starting threat analysis worker...")
        
        while self.running and THREAT_ANALYSIS_ENABLED:
            try:
                # Load satellite data
                self._update_satellite_cache()
                
                # Perform collision analysis
                threats = self._analyze_collision_threats()
                
                # Store results
                self._store_threat_results(threats)
                
                # Process high-priority threats
                self._process_critical_threats(threats)
                
                # Wait before next analysis
                time.sleep(ANALYSIS_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Worker interrupted by user")
                self.running = False
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}", exc_info=True)
                time.sleep(60)  # Wait before retrying
                
    def _update_satellite_cache(self) -> None:
        """Update local cache of satellite TLE data"""
        try:
            # Get all satellite TLE keys
            pattern = "satellite:tle:*"
            keys = self.redis_client.keys(pattern)
            
            logger.info(f"Loading {len(keys)} satellites for analysis")
            
            for key in keys:
                try:
                    tle_json = self.redis_client.get(key)
                    if tle_json:
                        tle_data = json.loads(tle_json)
                        sat_id = key.split(':')[-1]
                        
                        # Parse TLE and create SGP4 satellite object
                        satellite = self._create_satellite_from_tle(tle_data)
                        if satellite:
                            self.satellites_cache[sat_id] = {
                                'satellite': satellite,
                                'tle_data': tle_data,
                                'name': tle_data.get('OBJECT_NAME', 'Unknown')
                            }
                except Exception as e:
                    logger.error(f"Failed to load satellite {key}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to update satellite cache: {e}")
            
    def _create_satellite_from_tle(self, tle_data: Dict[str, Any]) -> Optional[Satrec]:
        """Create SGP4 satellite object from TLE data"""
        try:
            # Extract TLE lines
            line1 = tle_data.get('TLE_LINE1', '')
            line2 = tle_data.get('TLE_LINE2', '')
            
            if line1 and line2:
                satellite = Satrec.twoline2rv(line1, line2)
                return satellite
        except Exception as e:
            logger.error(f"Failed to create satellite from TLE: {e}")
        return None
        
    def _analyze_collision_threats(self) -> List[Dict[str, Any]]:
        """Analyze potential collision threats between satellites"""
        threats = []
        sat_ids = list(self.satellites_cache.keys())
        
        logger.info(f"Analyzing collision threats for {len(sat_ids)} satellites")
        
        # Compare each satellite pair
        for i in range(len(sat_ids)):
            for j in range(i + 1, len(sat_ids)):
                sat1_id = sat_ids[i]
                sat2_id = sat_ids[j]
                
                threat = self._check_collision_threat(sat1_id, sat2_id)
                if threat:
                    threats.append(threat)
                    
        logger.info(f"Found {len(threats)} potential collision threats")
        return threats
        
    def _check_collision_threat(self, sat1_id: str, sat2_id: str) -> Optional[Dict[str, Any]]:
        """Check collision threat between two satellites"""
        try:
            sat1_data = self.satellites_cache[sat1_id]
            sat2_data = self.satellites_cache[sat2_id]
            
            sat1 = sat1_data['satellite']
            sat2 = sat2_data['satellite']
            
            # Calculate positions over prediction period
            start_time = datetime.now(timezone.utc)
            end_time = start_time + timedelta(days=PREDICTION_DAYS)
            
            # Check multiple time points
            min_distance = float('inf')
            closest_time = None
            
            current_time = start_time
            while current_time < end_time:
                # Calculate Julian date
                jd, fr = jday(
                    current_time.year,
                    current_time.month,
                    current_time.day,
                    current_time.hour,
                    current_time.minute,
                    current_time.second
                )
                
                # Get positions
                e1, r1, v1 = sat1.sgp4(jd, fr)
                e2, r2, v2 = sat2.sgp4(jd, fr)
                
                if e1 == 0 and e2 == 0:  # No errors
                    # Calculate distance
                    distance = np.linalg.norm(np.array(r1) - np.array(r2))
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_time = current_time
                        
                current_time += timedelta(minutes=30)
                
            # Check if minimum distance is below threshold
            if min_distance < COLLISION_THRESHOLD_KM:
                threat_level = self._calculate_threat_level(min_distance)
                
                return {
                    'satellite1': {
                        'id': sat1_id,
                        'name': sat1_data['name']
                    },
                    'satellite2': {
                        'id': sat2_id,
                        'name': sat2_data['name']
                    },
                    'min_distance_km': min_distance,
                    'closest_approach_time': closest_time.isoformat(),
                    'threat_level': threat_level,
                    'analysis_time': datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error checking collision threat between {sat1_id} and {sat2_id}: {e}")
            
        return None
        
    def _calculate_threat_level(self, distance_km: float) -> str:
        """Calculate threat level based on distance"""
        if distance_km < 1.0:
            return 'CRITICAL'
        elif distance_km < 2.5:
            return 'HIGH'
        elif distance_km < 5.0:
            return 'MEDIUM'
        else:
            return 'LOW'
            
    def _store_threat_results(self, threats: List[Dict[str, Any]]) -> None:
        """Store threat analysis results in Redis"""
        try:
            # Store current threats
            key = 'threats:current'
            self.redis_client.setex(
                key,
                3600,  # 1 hour TTL
                json.dumps({
                    'threats': threats,
                    'analysis_time': datetime.now(timezone.utc).isoformat(),
                    'total_satellites': len(self.satellites_cache)
                })
            )
            
            # Store individual threat records
            for threat in threats:
                threat_key = f"threat:{threat['satellite1']['id']}:{threat['satellite2']['id']}"
                self.redis_client.setex(threat_key, 86400, json.dumps(threat))
                
            logger.info(f"Stored {len(threats)} threat records")
            
        except Exception as e:
            logger.error(f"Failed to store threat results: {e}")
            
    def _process_critical_threats(self, threats: List[Dict[str, Any]]) -> None:
        """Process critical threats requiring immediate attention"""
        critical_threats = [t for t in threats if t['threat_level'] in ['CRITICAL', 'HIGH']]
        
        for threat in critical_threats:
            try:
                # Create alert
                alert = {
                    'type': 'collision_threat',
                    'severity': threat['threat_level'],
                    'satellites': [threat['satellite1'], threat['satellite2']],
                    'details': threat,
                    'created_at': datetime.now(timezone.utc).isoformat()
                }
                
                # Push to alert queue
                self.redis_client.rpush('alerts:queue', json.dumps(alert))
                
                # Store in alerts history
                alert_key = f"alert:{datetime.now(timezone.utc).timestamp()}"
                self.redis_client.setex(alert_key, 604800, json.dumps(alert))  # 7 days
                
                logger.warning(f"Created {threat['threat_level']} alert for potential collision: "
                             f"{threat['satellite1']['name']} - {threat['satellite2']['name']}")
                             
            except Exception as e:
                logger.error(f"Failed to process critical threat: {e}")
                
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            self.redis_client.ping()
            return {
                'status': 'healthy',
                'worker_type': 'threat_analyzer',
                'threat_analysis_enabled': THREAT_ANALYSIS_ENABLED,
                'satellites_tracked': len(self.satellites_cache),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
    def shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("Shutting down threat analyzer...")
        self.running = False
        if self.redis_client:
            self.redis_client.close()
        logger.info("Threat analyzer shutdown complete")

def main():
    """Main entry point"""
    if not THREAT_ANALYSIS_ENABLED:
        logger.info("Threat analysis is disabled. Exiting.")
        return
        
    try:
        analyzer = ThreatAnalyzer()
        
        # Register shutdown handlers
        import signal
        signal.signal(signal.SIGTERM, lambda sig, frame: analyzer.shutdown())
        signal.signal(signal.SIGINT, lambda sig, frame: analyzer.shutdown())
        
        # Start analysis
        analyzer.run_analysis_loop()
        
    except Exception as e:
        logger.error(f"Threat analyzer failed to start: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()