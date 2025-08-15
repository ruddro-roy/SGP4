#!/usr/bin/env python3
"""
Satellite Data Sync Script
Fetches latest TLE data from CelesTrak and updates Redis cache
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import redis
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
CELESTRAK_API_BASE = os.getenv('CELESTRAK_API_BASE', 'https://celestrak.org')

# Satellite categories to sync
SATELLITE_CATEGORIES = [
    {
        'name': 'active',
        'url': '/NORAD/elements/gp.php?GROUP=active&FORMAT=JSON',
        'description': 'All active satellites'
    },
    {
        'name': 'stations',
        'url': '/NORAD/elements/gp.php?GROUP=stations&FORMAT=JSON',
        'description': 'Space stations'
    },
    {
        'name': 'weather',
        'url': '/NORAD/elements/gp.php?GROUP=weather&FORMAT=JSON',
        'description': 'Weather satellites'
    },
    {
        'name': 'communications',
        'url': '/NORAD/elements/gp.php?GROUP=geo&FORMAT=JSON',
        'description': 'Communications satellites'
    },
    {
        'name': 'navigation',
        'url': '/NORAD/elements/gp.php?GROUP=gnss&FORMAT=JSON',
        'description': 'Navigation satellites (GPS, GLONASS, etc.)'
    },
    {
        'name': 'science',
        'url': '/NORAD/elements/gp.php?GROUP=science&FORMAT=JSON',
        'description': 'Scientific satellites'
    },
    {
        'name': 'starlink',
        'url': '/NORAD/elements/gp.php?GROUP=starlink&FORMAT=JSON',
        'description': 'Starlink satellites'
    }
]

class SatelliteDataSync:
    """Main class for syncing satellite data"""
    
    def __init__(self):
        self.redis_client = self._init_redis()
        self.session = self._create_session()
        self.stats = {
            'total_satellites': 0,
            'updated': 0,
            'failed': 0,
            'categories_synced': []
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
            
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'User-Agent': 'SatelliteTrackingPlatform/1.0'
        })
        return session
        
    def sync_all_categories(self) -> Dict[str, Any]:
        """Sync all satellite categories"""
        logger.info("Starting satellite data synchronization...")
        start_time = time.time()
        
        for category in SATELLITE_CATEGORIES:
            try:
                self._sync_category(category)
            except Exception as e:
                logger.error(f"Failed to sync category {category['name']}: {e}")
                
        # Store sync metadata
        self._store_sync_metadata()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Sync completed in {elapsed_time:.2f} seconds")
        logger.info(f"Stats: {self.stats}")
        
        return self.stats
        
    def _sync_category(self, category: Dict[str, str]) -> None:
        """Sync a specific satellite category"""
        logger.info(f"Syncing category: {category['name']} - {category['description']}")
        
        try:
            url = f"{CELESTRAK_API_BASE}{category['url']}"
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            satellites = response.json()
            logger.info(f"Retrieved {len(satellites)} satellites in category {category['name']}")
            
            # Process each satellite
            for sat_data in satellites:
                self._process_satellite(sat_data, category['name'])
                
            self.stats['categories_synced'].append(category['name'])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error syncing {category['name']}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {category['name']}: {e}")
            raise
            
    def _process_satellite(self, sat_data: Dict[str, Any], category: str) -> None:
        """Process and store individual satellite data"""
        try:
            sat_id = sat_data.get('NORAD_CAT_ID', '')
            if not sat_id:
                logger.warning("Satellite missing NORAD_CAT_ID, skipping")
                return
                
            # Add category and sync metadata
            sat_data['CATEGORY'] = category
            sat_data['LAST_SYNC'] = datetime.now(timezone.utc).isoformat()
            
            # Validate TLE data
            if not self._validate_tle_data(sat_data):
                logger.warning(f"Invalid TLE data for satellite {sat_id}")
                self.stats['failed'] += 1
                return
                
            # Store in Redis with TTL
            key = f"satellite:tle:{sat_id}"
            self.redis_client.setex(
                key,
                86400 * 2,  # 48 hour TTL
                json.dumps(sat_data)
            )
            
            # Update satellite index
            self._update_satellite_index(sat_id, sat_data)
            
            self.stats['updated'] += 1
            self.stats['total_satellites'] += 1
            
        except Exception as e:
            logger.error(f"Failed to process satellite: {e}")
            self.stats['failed'] += 1
            
    def _validate_tle_data(self, sat_data: Dict[str, Any]) -> bool:
        """Validate TLE data completeness"""
        required_fields = ['TLE_LINE1', 'TLE_LINE2', 'OBJECT_NAME', 'NORAD_CAT_ID']
        return all(field in sat_data and sat_data[field] for field in required_fields)
        
    def _update_satellite_index(self, sat_id: str, sat_data: Dict[str, Any]) -> None:
        """Update satellite search index"""
        try:
            # Create search index entry
            index_data = {
                'id': sat_id,
                'name': sat_data.get('OBJECT_NAME', ''),
                'category': sat_data.get('CATEGORY', ''),
                'object_type': sat_data.get('OBJECT_TYPE', ''),
                'country': sat_data.get('COUNTRY_CODE', ''),
                'launch_date': sat_data.get('LAUNCH_DATE', ''),
                'decay_date': sat_data.get('DECAY_DATE', ''),
                'period': sat_data.get('PERIOD', 0),
                'inclination': sat_data.get('INCLINATION', 0),
                'apogee': sat_data.get('APOGEE', 0),
                'perigee': sat_data.get('PERIGEE', 0)
            }
            
            # Store in satellite index
            index_key = f"satellite:index:{sat_id}"
            self.redis_client.setex(
                index_key,
                86400 * 7,  # 7 day TTL
                json.dumps(index_data)
            )
            
            # Add to category set
            category_set_key = f"satellites:category:{sat_data.get('CATEGORY', 'unknown')}"
            self.redis_client.sadd(category_set_key, sat_id)
            self.redis_client.expire(category_set_key, 86400 * 7)
            
            # Add to active satellites set
            if sat_data.get('OBJECT_TYPE') != 'DEBRIS':
                self.redis_client.sadd('satellites:active', sat_id)
                self.redis_client.expire('satellites:active', 86400 * 7)
                
        except Exception as e:
            logger.error(f"Failed to update satellite index for {sat_id}: {e}")
            
    def _store_sync_metadata(self) -> None:
        """Store synchronization metadata"""
        try:
            metadata = {
                'last_sync': datetime.now(timezone.utc).isoformat(),
                'stats': self.stats,
                'categories': SATELLITE_CATEGORIES
            }
            
            self.redis_client.setex(
                'sync:metadata',
                86400,  # 24 hour TTL
                json.dumps(metadata)
            )
            
            # Store sync history
            history_key = f"sync:history:{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.redis_client.setex(
                history_key,
                86400 * 30,  # 30 day TTL
                json.dumps(metadata)
            )
            
        except Exception as e:
            logger.error(f"Failed to store sync metadata: {e}")
            
    def cleanup_old_data(self) -> None:
        """Clean up old satellite data"""
        try:
            logger.info("Cleaning up old satellite data...")
            
            # Get all satellite keys
            pattern = "satellite:tle:*"
            keys = self.redis_client.keys(pattern)
            
            cleaned = 0
            for key in keys:
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # No TTL set
                    self.redis_client.expire(key, 86400 * 2)  # Set 48 hour TTL
                    cleaned += 1
                    
            logger.info(f"Cleaned up {cleaned} satellite records")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

def main():
    """Main entry point"""
    try:
        syncer = SatelliteDataSync()
        
        # Perform sync
        stats = syncer.sync_all_categories()
        
        # Cleanup old data
        syncer.cleanup_old_data()
        
        # Exit with appropriate code
        if stats['failed'] > 0:
            logger.warning(f"Sync completed with {stats['failed']} failures")
            sys.exit(1)
        else:
            logger.info("Sync completed successfully")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()