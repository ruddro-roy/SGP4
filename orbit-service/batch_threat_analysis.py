#!/usr/bin/env python3
"""
Batch Threat Analysis Script
Performs comprehensive collision risk assessment for all satellites
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timezone, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import redis
from sgp4.api import Satrec, jday

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
THREAT_ANALYSIS_ENABLED = os.getenv('THREAT_ANALYSIS_ENABLED', 'true').lower() == 'true'
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
ANALYSIS_WINDOW_DAYS = int(os.getenv('ANALYSIS_WINDOW_DAYS', '14'))
HIGH_RISK_THRESHOLD_KM = float(os.getenv('HIGH_RISK_THRESHOLD_KM', '10.0'))
CRITICAL_THRESHOLD_KM = float(os.getenv('CRITICAL_THRESHOLD_KM', '2.0'))

class BatchThreatAnalyzer:
    """Batch processor for comprehensive threat analysis"""
    
    def __init__(self):
        self.redis_client = self._init_redis()
        self.analysis_start_time = datetime.now(timezone.utc)
        self.stats = {
            'total_satellites': 0,
            'pairs_analyzed': 0,
            'threats_found': 0,
            'critical_threats': 0,
            'high_threats': 0,
            'processing_time': 0
        }
        
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
            
    def run_batch_analysis(self) -> Dict[str, Any]:
        """Run comprehensive batch threat analysis"""
        if not THREAT_ANALYSIS_ENABLED:
            logger.info("Threat analysis is disabled")
            return self.stats
            
        logger.info("Starting batch threat analysis...")
        start_time = time.time()
        
        try:
            # Load all satellites
            satellites = self._load_all_satellites()
            self.stats['total_satellites'] = len(satellites)
            
            if len(satellites) < 2:
                logger.warning("Not enough satellites for analysis")
                return self.stats
                
            # Generate satellite pairs for analysis
            pairs = self._generate_satellite_pairs(satellites)
            total_pairs = len(pairs)
            logger.info(f"Analyzing {total_pairs} satellite pairs")
            
            # Process in batches with multiprocessing
            all_threats = []
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit batches for processing
                futures = []
                for i in range(0, total_pairs, BATCH_SIZE):
                    batch = pairs[i:i + BATCH_SIZE]
                    future = executor.submit(self._process_batch, batch, satellites)
                    futures.append(future)
                    
                # Collect results
                for future in as_completed(futures):
                    try:
                        batch_threats = future.result()
                        all_threats.extend(batch_threats)
                        self.stats['pairs_analyzed'] += BATCH_SIZE
                        
                        # Log progress
                        progress = (self.stats['pairs_analyzed'] / total_pairs) * 100
                        logger.info(f"Progress: {progress:.1f}% ({self.stats['pairs_analyzed']}/{total_pairs})")
                        
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
                        
            # Process and store results
            self._process_analysis_results(all_threats)
            
            # Generate report
            self._generate_threat_report(all_threats)
            
            self.stats['processing_time'] = time.time() - start_time
            logger.info(f"Batch analysis completed in {self.stats['processing_time']:.2f} seconds")
            logger.info(f"Final stats: {self.stats}")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}", exc_info=True)
            raise
            
    def _load_all_satellites(self) -> Dict[str, Dict[str, Any]]:
        """Load all satellite data from Redis"""
        satellites = {}
        
        try:
            # Get all satellite TLE keys
            pattern = "satellite:tle:*"
            keys = self.redis_client.keys(pattern)
            
            logger.info(f"Loading {len(keys)} satellites")
            
            for key in keys:
                try:
                    tle_json = self.redis_client.get(key)
                    if tle_json:
                        tle_data = json.loads(tle_json)
                        sat_id = key.split(':')[-1]
                        
                        # Create SGP4 satellite object
                        line1 = tle_data.get('TLE_LINE1', '')
                        line2 = tle_data.get('TLE_LINE2', '')
                        
                        if line1 and line2:
                            satellite = Satrec.twoline2rv(line1, line2)
                            satellites[sat_id] = {
                                'id': sat_id,
                                'name': tle_data.get('OBJECT_NAME', 'Unknown'),
                                'satellite': satellite,
                                'tle_data': tle_data,
                                'category': tle_data.get('CATEGORY', 'unknown')
                            }
                            
                except Exception as e:
                    logger.error(f"Failed to load satellite {key}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load satellites: {e}")
            
        return satellites
        
    def _generate_satellite_pairs(self, satellites: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Generate all unique satellite pairs for analysis"""
        sat_ids = list(satellites.keys())
        pairs = []
        
        # Prioritize certain satellite combinations
        priority_categories = ['stations', 'active', 'communications']
        
        for i in range(len(sat_ids)):
            for j in range(i + 1, len(sat_ids)):
                sat1 = satellites[sat_ids[i]]
                sat2 = satellites[sat_ids[j]]
                
                # Check if pair should be analyzed
                if self._should_analyze_pair(sat1, sat2, priority_categories):
                    pairs.append((sat_ids[i], sat_ids[j]))
                    
        return pairs
        
    def _should_analyze_pair(self, sat1: Dict[str, Any], sat2: Dict[str, Any], 
                           priority_categories: List[str]) -> bool:
        """Determine if satellite pair should be analyzed"""
        # Always analyze if either satellite is in priority category
        if (sat1['category'] in priority_categories or 
            sat2['category'] in priority_categories):
            return True
            
        # Skip if both are debris
        if (sat1['tle_data'].get('OBJECT_TYPE') == 'DEBRIS' and 
            sat2['tle_data'].get('OBJECT_TYPE') == 'DEBRIS'):
            return False
            
        # Analyze if orbits are similar (rough check)
        try:
            period1 = float(sat1['tle_data'].get('PERIOD', 0))
            period2 = float(sat2['tle_data'].get('PERIOD', 0))
            
            # If periods are within 10%, orbits might intersect
            if period1 > 0 and period2 > 0:
                period_diff = abs(period1 - period2) / max(period1, period2)
                return period_diff < 0.1
                
        except Exception:
            pass
            
        return True
        
    def _process_batch(self, pairs: List[Tuple[str, str]], 
                      satellites: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of satellite pairs"""
        threats = []
        
        for sat1_id, sat2_id in pairs:
            try:
                sat1 = satellites[sat1_id]
                sat2 = satellites[sat2_id]
                
                threat = self._analyze_pair_collision_risk(sat1, sat2)
                if threat:
                    threats.append(threat)
                    
            except Exception as e:
                logger.error(f"Failed to analyze pair {sat1_id}-{sat2_id}: {e}")
                
        return threats
        
    def _analyze_pair_collision_risk(self, sat1: Dict[str, Any], 
                                   sat2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze collision risk between two satellites"""
        try:
            sat1_obj = sat1['satellite']
            sat2_obj = sat2['satellite']
            
            # Time range for analysis
            start_time = self.analysis_start_time
            end_time = start_time + timedelta(days=ANALYSIS_WINDOW_DAYS)
            
            # Find closest approach
            min_distance = float('inf')
            closest_time = None
            closest_positions = None
            
            # Sample points throughout the analysis window
            time_step = timedelta(minutes=5)  # 5-minute intervals
            current_time = start_time
            
            while current_time < end_time:
                # Calculate positions
                jd, fr = jday(
                    current_time.year, current_time.month, current_time.day,
                    current_time.hour, current_time.minute, current_time.second
                )
                
                e1, r1, v1 = sat1_obj.sgp4(jd, fr)
                e2, r2, v2 = sat2_obj.sgp4(jd, fr)
                
                if e1 == 0 and e2 == 0:  # No propagation errors
                    # Calculate distance
                    r1_array = np.array(r1)
                    r2_array = np.array(r2)
                    distance = np.linalg.norm(r1_array - r2_array)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_time = current_time
                        closest_positions = {
                            'sat1': {'r': r1, 'v': v1},
                            'sat2': {'r': r2, 'v': v2}
                        }
                        
                current_time += time_step
                
            # Check if this is a threat
            if min_distance < HIGH_RISK_THRESHOLD_KM:
                # Calculate relative velocity at closest approach
                rel_velocity = 0
                if closest_positions:
                    v1_array = np.array(closest_positions['sat1']['v'])
                    v2_array = np.array(closest_positions['sat2']['v'])
                    rel_velocity = np.linalg.norm(v1_array - v2_array)
                    
                threat_level = self._calculate_threat_level(min_distance)
                
                return {
                    'satellite1': {
                        'id': sat1['id'],
                        'name': sat1['name'],
                        'category': sat1['category']
                    },
                    'satellite2': {
                        'id': sat2['id'],
                        'name': sat2['name'],
                        'category': sat2['category']
                    },
                    'min_distance_km': min_distance,
                    'closest_approach_time': closest_time.isoformat(),
                    'relative_velocity_km_s': rel_velocity,
                    'threat_level': threat_level,
                    'analysis_window': {
                        'start': start_time.isoformat(),
                        'end': end_time.isoformat()
                    },
                    'positions_at_closest': closest_positions
                }
                
        except Exception as e:
            logger.error(f"Error analyzing collision risk: {e}")
            
        return None
        
    def _calculate_threat_level(self, distance_km: float) -> str:
        """Calculate threat level based on distance"""
        if distance_km < 0.5:
            return 'EMERGENCY'
        elif distance_km < CRITICAL_THRESHOLD_KM:
            return 'CRITICAL'
        elif distance_km < 5.0:
            return 'HIGH'
        elif distance_km < HIGH_RISK_THRESHOLD_KM:
            return 'MEDIUM'
        else:
            return 'LOW'
            
    def _process_analysis_results(self, threats: List[Dict[str, Any]]) -> None:
        """Process and store analysis results"""
        try:
            # Count threat levels
            for threat in threats:
                self.stats['threats_found'] += 1
                
                if threat['threat_level'] in ['EMERGENCY', 'CRITICAL']:
                    self.stats['critical_threats'] += 1
                elif threat['threat_level'] == 'HIGH':
                    self.stats['high_threats'] += 1
                    
            # Sort threats by risk level and distance
            threats.sort(key=lambda x: (
                ['EMERGENCY', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].index(x['threat_level']),
                x['min_distance_km']
            ))
            
            # Store batch analysis results
            batch_key = f"batch_analysis:{self.analysis_start_time.strftime('%Y%m%d_%H%M%S')}"
            self.redis_client.setex(
                batch_key,
                86400 * 7,  # 7 days TTL
                json.dumps({
                    'analysis_time': self.analysis_start_time.isoformat(),
                    'stats': self.stats,
                    'threats': threats[:1000],  # Store top 1000 threats
                    'total_threats': len(threats)
                })
            )
            
            # Update current threats
            self.redis_client.setex(
                'threats:batch:current',
                86400,  # 24 hours TTL
                json.dumps({
                    'threats': threats[:100],  # Top 100 threats
                    'stats': self.stats,
                    'analysis_time': self.analysis_start_time.isoformat()
                })
            )
            
            # Create alerts for critical threats
            for threat in threats:
                if threat['threat_level'] in ['EMERGENCY', 'CRITICAL']:
                    self._create_threat_alert(threat)
                    
        except Exception as e:
            logger.error(f"Failed to process analysis results: {e}")
            
    def _create_threat_alert(self, threat: Dict[str, Any]) -> None:
        """Create alert for critical threat"""
        try:
            alert = {
                'type': 'batch_collision_threat',
                'severity': threat['threat_level'],
                'threat': threat,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'batch_analysis_time': self.analysis_start_time.isoformat()
            }
            
            # Push to alert queue
            self.redis_client.rpush('alerts:critical', json.dumps(alert))
            
            # Store individual alert
            alert_key = f"alert:batch:{threat['satellite1']['id']}:{threat['satellite2']['id']}"
            self.redis_client.setex(alert_key, 86400 * 3, json.dumps(alert))  # 3 days
            
            logger.warning(
                f"{threat['threat_level']} threat: {threat['satellite1']['name']} - "
                f"{threat['satellite2']['name']} at {threat['min_distance_km']:.2f} km"
            )
            
        except Exception as e:
            logger.error(f"Failed to create threat alert: {e}")
            
    def _generate_threat_report(self, threats: List[Dict[str, Any]]) -> None:
        """Generate comprehensive threat report"""
        try:
            report = {
                'report_time': datetime.now(timezone.utc).isoformat(),
                'analysis_start': self.analysis_start_time.isoformat(),
                'analysis_window_days': ANALYSIS_WINDOW_DAYS,
                'statistics': self.stats,
                'threat_summary': {
                    'emergency': len([t for t in threats if t['threat_level'] == 'EMERGENCY']),
                    'critical': len([t for t in threats if t['threat_level'] == 'CRITICAL']),
                    'high': len([t for t in threats if t['threat_level'] == 'HIGH']),
                    'medium': len([t for t in threats if t['threat_level'] == 'MEDIUM']),
                    'low': len([t for t in threats if t['threat_level'] == 'LOW'])
                },
                'top_threats': threats[:20],  # Top 20 threats
                'recommendations': self._generate_recommendations(threats)
            }
            
            # Store report
            report_key = f"threat_report:{datetime.now(timezone.utc).strftime('%Y%m%d')}"
            self.redis_client.setex(
                report_key,
                86400 * 30,  # 30 days TTL
                json.dumps(report)
            )
            
            logger.info(f"Generated threat report: {report['threat_summary']}")
            
        except Exception as e:
            logger.error(f"Failed to generate threat report: {e}")
            
    def _generate_recommendations(self, threats: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on threat analysis"""
        recommendations = []
        
        critical_count = len([t for t in threats if t['threat_level'] in ['EMERGENCY', 'CRITICAL']])
        
        if critical_count > 0:
            recommendations.append(
                f"URGENT: {critical_count} critical collision threats detected. "
                "Immediate action required for affected satellites."
            )
            
        if self.stats['high_threats'] > 10:
            recommendations.append(
                "High number of collision risks detected. Consider implementing "
                "automated collision avoidance maneuvers."
            )
            
        # Category-specific recommendations
        station_threats = [t for t in threats if 
                          t['satellite1']['category'] == 'stations' or 
                          t['satellite2']['category'] == 'stations']
        
        if station_threats:
            recommendations.append(
                f"{len(station_threats)} threats involve space stations. "
                "Priority monitoring and coordination with station operators recommended."
            )
            
        return recommendations

def main():
    """Main entry point"""
    if not THREAT_ANALYSIS_ENABLED:
        logger.info("Threat analysis is disabled. Exiting.")
        return
        
    try:
        analyzer = BatchThreatAnalyzer()
        stats = analyzer.run_batch_analysis()
        
        # Exit with appropriate code
        if stats['critical_threats'] > 0:
            logger.error(f"Critical threats detected: {stats['critical_threats']}")
            sys.exit(2)
        elif stats['threats_found'] > 0:
            logger.warning(f"Threats detected: {stats['threats_found']}")
            sys.exit(1)
        else:
            logger.info("No significant threats detected")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()