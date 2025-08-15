#!/usr/bin/env python3
"""
Satellite Data Processor Worker
Handles background processing of satellite orbital data
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import redis
import requests
from flask import Flask
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('worker.log')
    ]
)
logger = logging.getLogger(__name__)

# Environment configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
CELESTRAK_API_BASE = os.getenv('CELESTRAK_API_BASE', 'https://celestrak.org')
WORKER_TYPE = os.getenv('WORKER_TYPE', 'data_processor')
PROCESSING_INTERVAL = int(os.getenv('PROCESSING_INTERVAL', '60'))  # seconds

class SatelliteDataProcessor:
    """Main worker class for processing satellite data"""
    
    def __init__(self):
        self.redis_client = self._init_redis()
        self.session = self._create_session()
        self.running = True
        
    def _init_redis(self) -> redis.Redis:
        """Initialize Redis connection with retry logic"""
        try:
            client = redis.from_url(REDIS_URL, decode_responses=True)
            client.ping()
            logger.info("Successfully connected to Redis")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
            
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
        
    def process_satellite_data(self) -> None:
        """Main processing loop for satellite data"""
        while self.running:
            try:
                # Get pending satellite updates from queue
                task = self.redis_client.lpop('satellite_update_queue')
                
                if task:
                    task_data = json.loads(task)
                    self._process_task(task_data)
                else:
                    # No tasks, wait before checking again
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                logger.info("Worker interrupted by user")
                self.running = False
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                time.sleep(10)  # Wait before retrying
                
    def _process_task(self, task_data: Dict[str, Any]) -> None:
        """Process individual satellite update task"""
        try:
            satellite_id = task_data.get('satellite_id')
            task_type = task_data.get('type', 'update')
            
            logger.info(f"Processing {task_type} for satellite {satellite_id}")
            
            if task_type == 'update':
                self._update_satellite_data(satellite_id)
            elif task_type == 'predict':
                self._predict_satellite_position(satellite_id)
            elif task_type == 'collision_check':
                self._check_collision_risk(satellite_id)
                
            # Mark task as completed
            self._mark_task_completed(task_data)
            
        except Exception as e:
            logger.error(f"Failed to process task {task_data}: {e}")
            self._handle_task_failure(task_data, str(e))
            
    def _update_satellite_data(self, satellite_id: str) -> None:
        """Fetch and update satellite TLE data"""
        try:
            # Fetch latest TLE from CelesTrak
            url = f"{CELESTRAK_API_BASE}/NORAD/elements/gp.php?CATNR={satellite_id}&FORMAT=JSON"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            tle_data = response.json()
            if tle_data:
                # Store updated TLE in Redis
                key = f"satellite:tle:{satellite_id}"
                self.redis_client.setex(
                    key,
                    86400,  # 24 hour TTL
                    json.dumps(tle_data[0])
                )
                logger.info(f"Updated TLE data for satellite {satellite_id}")
                
        except Exception as e:
            logger.error(f"Failed to update satellite {satellite_id}: {e}")
            raise
            
    def _predict_satellite_position(self, satellite_id: str) -> None:
        """Calculate future positions for satellite"""
        # Implementation for orbital prediction
        logger.info(f"Calculating predictions for satellite {satellite_id}")
        # Add actual prediction logic here
        
    def _check_collision_risk(self, satellite_id: str) -> None:
        """Check collision risk with other satellites"""
        # Implementation for collision detection
        logger.info(f"Checking collision risk for satellite {satellite_id}")
        # Add actual collision detection logic here
        
    def _mark_task_completed(self, task_data: Dict[str, Any]) -> None:
        """Mark task as completed in Redis"""
        completed_key = f"task:completed:{task_data.get('id', 'unknown')}"
        self.redis_client.setex(completed_key, 3600, json.dumps({
            'task': task_data,
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'worker': WORKER_TYPE
        }))
        
    def _handle_task_failure(self, task_data: Dict[str, Any], error: str) -> None:
        """Handle failed tasks with retry logic"""
        retry_count = task_data.get('retry_count', 0)
        
        if retry_count < 3:
            # Requeue with increased retry count
            task_data['retry_count'] = retry_count + 1
            task_data['last_error'] = error
            self.redis_client.rpush('satellite_update_queue', json.dumps(task_data))
            logger.warning(f"Requeued task {task_data.get('id')} (retry {retry_count + 1})")
        else:
            # Move to dead letter queue
            self.redis_client.rpush('satellite_update_dlq', json.dumps({
                'task': task_data,
                'error': error,
                'failed_at': datetime.now(timezone.utc).isoformat()
            }))
            logger.error(f"Task {task_data.get('id')} moved to DLQ after {retry_count} retries")
            
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            self.redis_client.ping()
            return {
                'status': 'healthy',
                'worker_type': WORKER_TYPE,
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
        logger.info("Shutting down worker...")
        self.running = False
        if self.redis_client:
            self.redis_client.close()
        logger.info("Worker shutdown complete")

def main():
    """Main entry point"""
    logger.info(f"Starting {WORKER_TYPE} worker...")
    
    try:
        processor = SatelliteDataProcessor()
        
        # Register shutdown handlers
        import signal
        signal.signal(signal.SIGTERM, lambda sig, frame: processor.shutdown())
        signal.signal(signal.SIGINT, lambda sig, frame: processor.shutdown())
        
        # Start processing
        processor.process_satellite_data()
        
    except Exception as e:
        logger.error(f"Worker failed to start: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()