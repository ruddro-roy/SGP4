#!/usr/bin/env python3
"""
Orbital Perturbation Analysis

Analyzes and detects deviations between predicted and observed satellite positions.
These deviations can indicate:
- TLE data staleness
- Unmodeled perturbations (solar radiation pressure, third-body effects)
- Atmospheric density variations
- Satellite maneuvers

This module provides tools for:
- Computing position residuals
- Classifying perturbation severity
- Identifying systematic vs. random errors
- Flagging when TLE updates are needed

Educational purpose: Understanding the limitations of analytical propagation models
and the importance of regular TLE updates for maintaining accuracy.
"""

import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PerturbationLevel(Enum):
    """Perturbation severity levels"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PerturbationAlert:
    """Alert for detected orbital perturbation"""

    def __init__(
        self,
        timestamp: datetime,
        satellite_id: int,
        deviation_km: float,
        level: PerturbationLevel,
        predicted_pos: np.ndarray,
        observed_pos: np.ndarray,
    ):
        self.timestamp = timestamp
        self.satellite_id = satellite_id
        self.deviation_km = deviation_km
        self.level = level
        self.predicted_pos = predicted_pos
        self.observed_pos = observed_pos


class PerturbationScanner:
    """Scanner for detecting orbital perturbations exceeding thresholds"""

    def __init__(self, error_threshold_km: float = 1.0):
        self.error_threshold_km = error_threshold_km

    def scan_satellite_perturbations(
        self,
        satellite_data: Dict[str, Any],
        predicted_positions: List[Tuple[datetime, np.ndarray]],
        observed_positions: List[Tuple[datetime, np.ndarray]],
    ) -> List[PerturbationAlert]:
        """
        Scan for perturbations by comparing predicted vs observed positions

        Args:
            satellite_data: Satellite metadata
            predicted_positions: List of (timestamp, position_vector) tuples for predictions
            observed_positions: List of (timestamp, position_vector) tuples for observations

        Returns:
            List of perturbation alerts
        """
        alerts = []

        # Match predicted and observed positions by timestamp
        for pred_time, pred_pos in predicted_positions:
            # Find closest observed position in time
            closest_obs = min(
                observed_positions,
                key=lambda x: abs((x[0] - pred_time).total_seconds()),
            )
            obs_time, obs_pos = closest_obs

            # Skip if time difference is too large (>5 minutes)
            time_diff = abs((obs_time - pred_time).total_seconds())
            if time_diff > 300:  # 5 minutes
                continue

            # Calculate position deviation
            deviation_vector = obs_pos - pred_pos
            deviation_km = np.linalg.norm(deviation_vector)

            # Check if deviation exceeds threshold
            if deviation_km > self.error_threshold_km:
                level = self._classify_perturbation_level(deviation_km)

                alert = PerturbationAlert(
                    timestamp=pred_time,
                    satellite_id=satellite_data.get("norad_id", 0),
                    deviation_km=deviation_km,
                    level=level,
                    predicted_pos=pred_pos,
                    observed_pos=obs_pos,
                )

                alerts.append(alert)

                logger.warning(
                    f"Perturbation detected for satellite {satellite_data.get('name', 'Unknown')}: "
                    f"{deviation_km:.2f} km deviation at {pred_time}"
                )

        return alerts

    def _classify_perturbation_level(self, deviation_km: float) -> PerturbationLevel:
        """Classify perturbation severity based on deviation magnitude"""
        if deviation_km < 1.0:
            return PerturbationLevel.LOW
        elif deviation_km < 5.0:
            return PerturbationLevel.MEDIUM
        elif deviation_km < 10.0:
            return PerturbationLevel.HIGH
        else:
            return PerturbationLevel.CRITICAL
