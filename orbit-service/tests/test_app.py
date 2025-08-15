#!/usr/bin/env python3
"""
Test suite for Orbit Service
"""

import unittest
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app


class TestOrbitService(unittest.TestCase):
    """Test cases for orbit service endpoints"""
    
    def setUp(self):
        """Set up test client"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('version', data)
        self.assertIn('timestamp', data)
        
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('service', data)
        self.assertEqual(data['service'], 'Orbit Prediction Service')
        
    def test_satellites_endpoint_no_data(self):
        """Test satellites endpoint when no data is available"""
        response = self.client.get('/api/satellites')
        # Should return 200 even with empty data
        self.assertIn(response.status_code, [200, 404])
        
    def test_predict_endpoint_missing_params(self):
        """Test predict endpoint with missing parameters"""
        response = self.client.post('/api/predict')
        self.assertEqual(response.status_code, 400)
        
    def test_predict_endpoint_invalid_tle(self):
        """Test predict endpoint with invalid TLE data"""
        data = {
            'tle_line1': 'invalid',
            'tle_line2': 'invalid',
            'start_time': '2024-01-01T00:00:00Z',
            'end_time': '2024-01-01T01:00:00Z'
        }
        response = self.client.post('/api/predict', 
                                  json=data,
                                  content_type='application/json')
        self.assertIn(response.status_code, [400, 422])
        
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = self.client.get('/health')
        self.assertIn('Access-Control-Allow-Origin', response.headers)
        

class TestWorkerFunctions(unittest.TestCase):
    """Test cases for worker functions"""
    
    def test_worker_import(self):
        """Test that worker modules can be imported"""
        try:
            import worker
            import threat_worker
            import sync_satellites
            import batch_threat_analysis
        except ImportError as e:
            self.fail(f"Failed to import worker module: {e}")
            
    def test_redis_url_environment(self):
        """Test Redis URL environment variable handling"""
        # Save original
        original = os.environ.get('REDIS_URL')
        
        # Test with custom URL
        os.environ['REDIS_URL'] = 'redis://test:6379'
        from worker import REDIS_URL
        self.assertEqual(REDIS_URL, 'redis://test:6379')
        
        # Restore original
        if original:
            os.environ['REDIS_URL'] = original
        else:
            os.environ.pop('REDIS_URL', None)


if __name__ == '__main__':
    unittest.main()