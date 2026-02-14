"""
Dynamic Delivery Environment with traffic-aware routing and deadlines.

This environment simulates a last-mile delivery problem with:
- Random 2D delivery locations
- Depot at center
- Delivery deadlines for each node
- Dynamic traffic multipliers
- Time tracking and late delivery penalties
"""

import numpy as np
from typing import Tuple, Dict, List


class DynamicDeliveryEnv:
    """
    Environment for dynamic delivery routing with traffic and deadlines.
    """
    
    def __init__(self, num_nodes=10, grid_size=10.0, seed=None, traffic_mode="simulated", traffic_api=None):
        """
        Initialize the delivery environment.
        
        Args:
            num_nodes (int): Number of delivery nodes (excluding depot)
            grid_size (float): Size of the delivery grid
            seed (int): Random seed for reproducibility
            traffic_mode (str): "simulated" or "real" traffic mode
            traffic_api: TrafficAPI instance for real traffic data (optional)
        """
        self.num_nodes = num_nodes
        self.grid_size = grid_size
        self.total_nodes = num_nodes + 1  # Including depot
        self.traffic_mode = traffic_mode
        self.traffic_api = traffic_api
        
        # Bangalore bounding box for real traffic
        self.bangalore_bbox = {
            'lat_min': 12.87,
            'lat_max': 13.07,
            'lon_min': 77.48,
            'lon_max': 77.72
        }
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize environment
        self.locations = None
        self.deadlines = None
        self.traffic_multiplier = None
        self.baseline_times = None  # For real traffic normalization
        self.delivered = None
        self.current_node = None
        self.time_elapsed = None
        
        self.reset()
    
    def reset(self) -> Dict:
        """
        Reset the environment to initial state.
        
        Returns:
            dict: Initial state containing locations, deadlines, traffic, etc.
        """
        # Generate random locations for depot and delivery nodes
        self.locations = np.random.uniform(0, self.grid_size, (self.total_nodes, 2))
        
        # Depot is always the first node (index 0)
        # Delivery nodes are indices 1 to num_nodes
        
        # Generate deadlines (time window for each delivery)
        # Deadline is when delivery must be completed
        self.deadlines = np.random.uniform(10, 30, self.num_nodes + 1)
        self.deadlines[0] = float('inf')  # Depot has no deadline
        
        # Generate traffic multipliers (simulated or will be replaced by real)
        self.traffic_multiplier = np.random.uniform(0.8, 1.5, (self.total_nodes, self.total_nodes))
        np.fill_diagonal(self.traffic_multiplier, 1.0)  # No traffic to self
        
        # Compute baseline times for real traffic normalization
        self.baseline_times = np.zeros((self.total_nodes, self.total_nodes))
        for i in range(self.total_nodes):
            for j in range(self.total_nodes):
                if i != j:
                    dist = self._calculate_distance(i, j)
                    # Base time: distance at 30 km/h (in same units as simulation)
                    self.baseline_times[i][j] = dist
        
        # Initialize tracking
        self.delivered = np.zeros(self.total_nodes, dtype=bool)
        self.delivered[0] = True  # Depot is "delivered" (starting point)
        self.current_node = 0  # Start at depot
        self.time_elapsed = 0.0
        
        return self._get_state()
    
    def _grid_to_latlon(self, grid_coord):
        """
        Convert grid coordinates to Bangalore lat/lon.
        
        Args:
            grid_coord (array): [x, y] in range [0, grid_size]
        
        Returns:
            tuple: (lat, lon) in Bangalore bounding box
        """
        # Normalize to [0, 1]
        norm_x = grid_coord[0] / self.grid_size
        norm_y = grid_coord[1] / self.grid_size
        
        # Map to Bangalore bbox
        lat = self.bangalore_bbox['lat_min'] + norm_y * (self.bangalore_bbox['lat_max'] - self.bangalore_bbox['lat_min'])
        lon = self.bangalore_bbox['lon_min'] + norm_x * (self.bangalore_bbox['lon_max'] - self.bangalore_bbox['lon_min'])
        
        return (lat, lon)
    
    def step(self, action):
        """
        Execute action and return next state.
        
        Args:
            action (int): Index of next node to visit
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if action < 0 or action >= self.total_nodes:
            raise ValueError(f"Invalid action: {action}")
        
        if self.delivered[action] and action != 0:
            raise ValueError(f"Node {action} already delivered")
        
        # Calculate travel time
        distance = self._calculate_distance(self.current_node, action)
        
        # Get traffic multiplier (real or simulated)
        # Get traffic multiplier (real or simulated via API)
        if self.traffic_api:
            # Convert grid coords to lat/lon
            origin_latlon = self._grid_to_latlon(self.locations[self.current_node])
            dest_latlon = self._grid_to_latlon(self.locations[action])
            
            # Determine if we should force simulation
            force_sim = (self.traffic_mode != "real")
            
            # Get normalized multiplier (handles shocks and variance in API)
            multiplier = self.traffic_api.get_traffic_multiplier(
                origin_latlon, 
                dest_latlon, 
                force_simulation=force_sim
            )
            
            # Update traffic matrix with real multiplier (for agent state)
            self.traffic_multiplier[self.current_node, action] = multiplier
            
            travel_time = distance * multiplier
        else:
            # Legacy fallback if no API instance provided
            traffic = self.traffic_multiplier[self.current_node, action]
            travel_time = distance * traffic
        
        self.time_elapsed += travel_time
        
        # Calculate late penalty
        late_penalty = 0
        if action != 0:  # Not depot
            if self.time_elapsed > self.deadlines[action]:
                late_penalty = (self.time_elapsed - self.deadlines[action]) * 2
        
        # Mark as delivered
        if action != 0:
            self.delivered[action] = True
        
        # Move to new node
        self.current_node = action
        
        # Check if done (all delivered and returned to depot)
        done = np.all(self.delivered[1:]) and action == 0
        
        # Reward: negative time + penalty
        reward = -(travel_time + late_penalty)
        
        # Additional info
        info = {
            'travel_time': travel_time,
            'late_penalty': late_penalty,
            'distance': distance,
            'traffic_multiplier': self.traffic_multiplier[self.current_node, action]
        }
        
        return self._get_state(), reward, done, info
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions (undelivered nodes).
        
        Returns:
            np.ndarray: Boolean array where True = valid action
        """
        valid = ~self.delivered
        
        # CRITICAL FIX: Only allow returning to depot when all deliveries are done
        # This prevents infinite depot → depot → depot loops
        all_deliveries_complete = self.delivered[1:].all()
        valid[0] = all_deliveries_complete  # Depot only valid at end
        
        return valid
    
    def _get_state(self) -> Dict:
        """
        Get current state representation.
        
        Returns:
            dict: Current state
        """
        # Calculate urgency for each undelivered node
        urgency = np.zeros(self.total_nodes)
        for i in range(self.total_nodes):
            if not self.delivered[i]:
                time_to_node = self._calculate_distance(self.current_node, i)
                arrival_time = self.time_elapsed + time_to_node
                urgency[i] = max(0, arrival_time - self.deadlines[i])
        
        return {
            'locations': self.locations.copy(),
            'current_node': self.current_node,
            'delivered': self.delivered.copy(),
            'deadlines': self.deadlines.copy(),
            'traffic': self.traffic_multiplier.copy(),
            'time_elapsed': self.time_elapsed,
            'urgency': urgency,
            'valid_actions': self.get_valid_actions()
        }
    
    def _calculate_distance(self, node1: int, node2: int) -> float:
        """
        Calculate Euclidean distance between two nodes.
        
        Args:
            node1 (int): First node index
            node2 (int): Second node index
        
        Returns:
            float: Euclidean distance
        """
        return np.sqrt(np.sum((self.locations[node1] - self.locations[node2]) ** 2))
    
    def render(self):
        """Print current environment state."""
        print(f"\n=== Environment State ===")
        print(f"Current node: {self.current_node}")
        print(f"Time elapsed: {self.time_elapsed:.2f}")
        print(f"Delivered: {np.where(self.delivered)[0].tolist()}")
        print(f"Remaining: {np.where(~self.delivered)[0].tolist()}")
