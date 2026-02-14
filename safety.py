"""
Safety monitoring module for dynamic delivery routing.
"""

import numpy as np
from typing import Dict, Tuple

class SafetyMonitor:
    """
    Monitor for safety constraints and risks in the routing system.
    """
    
    def __init__(self, speed_limit: float = 60.0, variance_threshold: float = 0.5):
        """
        Initialize SafetyMonitor.
        
        Args:
            speed_limit (float): Max allowed speed (km/h) for warning
            variance_threshold (float): Threshold for high traffic variance warning
        """
        self.speed_limit = speed_limit
        self.variance_threshold = variance_threshold
        
    def estimate_speed(self, distance_km: float, duration_minutes: float) -> float:
        """
        Estimate speed in km/h.
        
        Args:
            distance_km (float): Distance in km
            duration_minutes (float): Duration in minutes
                (Note: Environment often uses abstract units, so we assume
                 distance=1.0 ~ 1km and time=1.0 ~ 1 min for compatibility 
                 if not specified, but here we calculate raw ratio * 60)
        
        Returns:
            float: Speed in km/h
        """
        if duration_minutes <= 0.001:
            return 0.0
        
        # Speed = Distance / Time
        # If inputs are standard units (km, min):
        return (distance_km / duration_minutes) * 60.0

    def check_safety(self, state: Dict, action: int, estimated_speed: float = None) -> Dict[str, bool]:
        """
        Check for safety risks.
        
        Args:
            state (Dict): Environment state
            action (int): Proposed action (node index)
            estimated_speed (float): Speed calculated from previous step or prediction
            
        Returns:
            dict: Safety flags e.g., {'overspeed': False, 'congestion_risk': True}
        """
        risks = {
            'overspeed': False,
            'congestion_risk': False,
            'fallback_triggered': False
        }
        
        # 1. Overspeed Check
        if estimated_speed is not None and estimated_speed > self.speed_limit:
            risks['overspeed'] = True
            
        # 2. Congestion Risk (Traffic Variance)
        # Check variance of traffic from current node to all others
        traffic_row = state['traffic'][state['current_node']]
        variance = np.var(traffic_row)
        
        if variance > self.variance_threshold:
            risks['congestion_risk'] = True
            
        # 3. Fallback Trigger (Extreme Instability)
        if variance > self.variance_threshold * 1.5:
            risks['fallback_triggered'] = True
            
        return risks
