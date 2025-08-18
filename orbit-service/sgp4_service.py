"""
Production SGP4 Propagation Service
Differentiable SGP4 with ML enhancements for microservices architecture
"""

import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from sgp4.api import Satrec
import numpy as np
import redis
import json
import hashlib
from datetime import datetime, timezone
import logging
from typing import Dict, List, Tuple, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DifferentiableSGP4Service:
    """Production-ready differentiable SGP4 service with caching and ML corrections"""
    
    def __init__(self, redis_host='localhost', redis_port=6379, cache_ttl=3600):
        self.cache_ttl = cache_ttl
        
        # Initialize Redis cache
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis cache connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable: {e}")
            self.redis_client = None
        
        # Satellite cache
        self.satellites = {}
        self.ml_models = {}
        
    def load_satellite(self, line1: str, line2: str, sat_id: Optional[str] = None) -> str:
        """Load TLE and return satellite ID"""
        try:
            satellite = Satrec.twoline2rv(line1, line2)
            
            if sat_id is None:
                sat_id = str(satellite.satnum)
            
            # Create differentiable model
            model = DifferentiableSGP4Model(line1, line2)
            
            self.satellites[sat_id] = {
                'satellite': satellite,
                'model': model,
                'line1': line1,
                'line2': line2,
                'loaded_at': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"📡 Loaded satellite {sat_id}")
            return sat_id
            
        except Exception as e:
            logger.error(f"❌ Failed to load satellite: {e}")
            raise
    
    def propagate(self, sat_id: str, tsince_minutes: float, 
                 use_ml_corrections: bool = False) -> Tuple[List[float], List[float]]:
        """Propagate satellite position and velocity"""
        if sat_id not in self.satellites:
            raise ValueError(f"Satellite {sat_id} not loaded")
        
        # Check cache first
        cache_key = f"sgp4:{sat_id}:{tsince_minutes}:{use_ml_corrections}"
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                result = json.loads(cached)
                return result['position'], result['velocity']
        
        try:
            sat_data = self.satellites[sat_id]
            model = sat_data['model']
            
            # Set model mode
            if use_ml_corrections:
                model.train()
            else:
                model.eval()
            
            # Propagate
            tsince_tensor = torch.tensor(tsince_minutes, dtype=torch.float32)
            with torch.no_grad():
                r_tensor, v_tensor = model(tsince_tensor)
            
            position = r_tensor.cpu().numpy().tolist()
            velocity = v_tensor.cpu().numpy().tolist()
            
            # Cache result
            if self.redis_client:
                result = {'position': position, 'velocity': velocity}
                self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(result))
            
            return position, velocity
            
        except Exception as e:
            logger.error(f"❌ Propagation failed for {sat_id}: {e}")
            raise
    
    def propagate_batch(self, sat_id: str, tsince_list: List[float],
                       use_ml_corrections: bool = False) -> Tuple[List[List[float]], List[List[float]]]:
        """Batch propagation for efficiency"""
        if sat_id not in self.satellites:
            raise ValueError(f"Satellite {sat_id} not loaded")
        
        try:
            sat_data = self.satellites[sat_id]
            model = sat_data['model']
            
            # Set model mode
            if use_ml_corrections:
                model.train()
            else:
                model.eval()
            
            # Batch propagation
            tsince_batch = torch.tensor(tsince_list, dtype=torch.float32)
            with torch.no_grad():
                r_batch, v_batch = model.propagate_batch(tsince_batch)
            
            positions = r_batch.cpu().numpy().tolist()
            velocities = v_batch.cpu().numpy().tolist()
            
            return positions, velocities
            
        except Exception as e:
            logger.error(f"❌ Batch propagation failed for {sat_id}: {e}")
            raise
    
    def train_ml_corrections(self, sat_id: str, training_data: Dict) -> Dict:
        """Train ML corrections for improved accuracy"""
        if sat_id not in self.satellites:
            raise ValueError(f"Satellite {sat_id} not loaded")
        
        try:
            model = self.satellites[sat_id]['model']
            
            # Extract training data
            times = torch.tensor(training_data['times'], dtype=torch.float32)
            true_positions = torch.tensor(training_data['positions'], dtype=torch.float32)
            true_velocities = torch.tensor(training_data['velocities'], dtype=torch.float32)
            
            # Training setup
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.train()
            
            losses = []
            for epoch in range(100):
                optimizer.zero_grad()
                
                # Forward pass
                pred_positions = []
                pred_velocities = []
                
                for t in times:
                    r, v = model(t)
                    pred_positions.append(r)
                    pred_velocities.append(v)
                
                pred_positions = torch.stack(pred_positions)
                pred_velocities = torch.stack(pred_velocities)
                
                # Compute loss
                pos_loss = torch.mean((pred_positions - true_positions) ** 2)
                vel_loss = torch.mean((pred_velocities - true_velocities) ** 2)
                total_loss = pos_loss + vel_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                losses.append(total_loss.item())
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}: Loss = {total_loss.item():.6f}")
            
            # Save trained model
            self.ml_models[sat_id] = model.state_dict()
            
            return {
                'status': 'success',
                'final_loss': losses[-1],
                'epochs': len(losses)
            }
            
        except Exception as e:
            logger.error(f"❌ ML training failed for {sat_id}: {e}")
            raise
    
    def get_satellite_info(self, sat_id: str) -> Dict:
        """Get satellite orbital information"""
        if sat_id not in self.satellites:
            raise ValueError(f"Satellite {sat_id} not loaded")
        
        sat_data = self.satellites[sat_id]
        satellite = sat_data['satellite']
        
        return {
            'sat_id': sat_id,
            'catalog_number': satellite.satnum,
            'epoch_year': satellite.epochyr,
            'epoch_days': satellite.epochdays,
            'inclination_deg': np.degrees(satellite.inclo),
            'raan_deg': np.degrees(satellite.nodeo),
            'eccentricity': satellite.ecco,
            'arg_perigee_deg': np.degrees(satellite.argpo),
            'mean_anomaly_deg': np.degrees(satellite.mo),
            'mean_motion_rev_day': satellite.no_kozai,
            'bstar': satellite.bstar,
            'loaded_at': sat_data['loaded_at']
        }

class DifferentiableSGP4Model(nn.Module):
    """Differentiable SGP4 model for ML enhancements"""
    
    def __init__(self, line1: str, line2: str):
        super().__init__()
        self.satellite = Satrec.twoline2rv(line1, line2)
        
        # Learnable correction parameters
        self.drag_correction = nn.Parameter(torch.tensor(0.0))
        self.j2_correction = nn.Parameter(torch.tensor(0.0))
        
        # ML correction network
        self.correction_net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6)  # [dx, dy, dz, dvx, dvy, dvz]
        )
        
    def forward(self, tsince_minutes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation with ML corrections"""
        tsince_np = tsince_minutes.detach().cpu().numpy()
        
        # Base SGP4 propagation
        jd = self.satellite.jdsatepoch + tsince_np / 1440.0
        error, r_km, v_km_s = self.satellite.sgp4(jd, 0.0)
        
        if error != 0:
            return torch.zeros(3), torch.zeros(3)
        
        r_tensor = torch.tensor(r_km, dtype=torch.float32)
        v_tensor = torch.tensor(v_km_s, dtype=torch.float32)
        
        # Apply ML corrections if in training mode
        if self.training:
            features = torch.tensor([
                tsince_minutes.item(),
                self.satellite.inclo,
                self.satellite.nodeo,
                self.satellite.ecco,
                self.satellite.argpo,
                self.satellite.mo,
                self.satellite.no_kozai
            ], dtype=torch.float32)
            
            corrections = self.correction_net(features)
            r_tensor = r_tensor + corrections[:3]
            v_tensor = v_tensor + corrections[3:]
            
            # Apply parameter corrections
            drag_effect = self.drag_correction * tsince_minutes / 1440.0
            r_tensor = r_tensor * (1.0 + drag_effect)
        
        return r_tensor, v_tensor
    
    def propagate_batch(self, tsince_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch propagation"""
        results_r = []
        results_v = []
        
        for tsince in tsince_batch:
            r, v = self.forward(tsince)
            results_r.append(r)
            results_v.append(v)
        
        return torch.stack(results_r), torch.stack(results_v)

# Flask API
app = Flask(__name__)
sgp4_service = DifferentiableSGP4Service()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'differentiable-sgp4',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'satellites_loaded': len(sgp4_service.satellites)
    })

@app.route('/satellite/load', methods=['POST'])
def load_satellite():
    """Load satellite from TLE"""
    try:
        data = request.json
        line1 = data['line1']
        line2 = data['line2']
        sat_id = data.get('sat_id')
        
        sat_id = sgp4_service.load_satellite(line1, line2, sat_id)
        
        return jsonify({
            'status': 'success',
            'sat_id': sat_id,
            'message': f'Satellite {sat_id} loaded successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/satellite/<sat_id>/propagate', methods=['POST'])
def propagate_satellite(sat_id):
    """Propagate satellite position"""
    try:
        data = request.json
        tsince_minutes = data['tsince_minutes']
        use_ml = data.get('use_ml_corrections', False)
        
        position, velocity = sgp4_service.propagate(sat_id, tsince_minutes, use_ml)
        
        return jsonify({
            'status': 'success',
            'sat_id': sat_id,
            'tsince_minutes': tsince_minutes,
            'position_km': position,
            'velocity_km_s': velocity,
            'ml_corrections': use_ml
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/satellite/<sat_id>/propagate_batch', methods=['POST'])
def propagate_batch(sat_id):
    """Batch propagate satellite positions"""
    try:
        data = request.json
        tsince_list = data['tsince_minutes']
        use_ml = data.get('use_ml_corrections', False)
        
        positions, velocities = sgp4_service.propagate_batch(sat_id, tsince_list, use_ml)
        
        return jsonify({
            'status': 'success',
            'sat_id': sat_id,
            'count': len(tsince_list),
            'positions_km': positions,
            'velocities_km_s': velocities,
            'ml_corrections': use_ml
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/satellite/<sat_id>/info', methods=['GET'])
def get_satellite_info(sat_id):
    """Get satellite information"""
    try:
        info = sgp4_service.get_satellite_info(sat_id)
        return jsonify({
            'status': 'success',
            'satellite': info
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/satellite/<sat_id>/train', methods=['POST'])
def train_corrections(sat_id):
    """Train ML corrections"""
    try:
        data = request.json
        training_data = data['training_data']
        
        result = sgp4_service.train_ml_corrections(sat_id, training_data)
        
        return jsonify({
            'status': 'success',
            'sat_id': sat_id,
            'training_result': result
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"🚀 Starting Differentiable SGP4 Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
