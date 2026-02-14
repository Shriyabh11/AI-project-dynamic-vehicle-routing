"""
Heuristic baseline for dynamic delivery routing.

This module provides a greedy urgency-based heuristic that serves as
a baseline for comparison with the DQN agent.

IMPORTANT: Pure Python implementation - NO PyTorch dependencies.
"""

import numpy as np
from typing import Tuple, List


def heuristic_route(locations: np.ndarray, 
                   deadlines: np.ndarray, 
                   traffic_levels: np.ndarray,
                   urgency_weight: float = 1.5) -> Tuple[List[int], float, float]:
    """
    Greedy heuristic for delivery routing with urgency-based scoring.
    
    Strategy:
    - Start at depot (node 0)
    - At each step, choose the unvisited node with the lowest score
    - Score = travel_time + urgency_weight * lateness_penalty
    - Return to depot when all nodes are visited
    
    Args:
        locations (np.ndarray): Array of shape (N, 2) with (x, y) coordinates
        deadlines (np.ndarray): Array of shape (N,) with deadline for each node
        traffic_levels (np.ndarray): Array of shape (N, N) with traffic multipliers
        urgency_weight (float): Weight for urgency penalty (default: 1.5)
    
    Returns:
        tuple: (route, total_travel_time, total_delay)
            - route: List of node indices in visit order
            - total_travel_time: Total time spent traveling
            - total_delay: Total lateness across all deliveries
    """
    num_nodes = len(locations)
    
    # Initialize state
    current_node = 0  # Start at depot
    current_time = 0.0
    visited = np.zeros(num_nodes, dtype=bool)
    visited[0] = True  # Depot is "visited"
    route = [0]  # Route starts at depot
    total_delay = 0.0
    
    # Visit all nodes
    while not visited[1:].all():
        best_node = None
        best_score = float('inf')
        
        # Evaluate all unvisited nodes
        for candidate in range(1, num_nodes):
            if visited[candidate]:
                continue
            
            # Calculate travel time to candidate
            distance = _calculate_distance(locations[current_node], locations[candidate])
            traffic = traffic_levels[current_node, candidate]
            travel_time = distance * traffic
            
            # Calculate arrival time
            arrival_time = current_time + travel_time
            
            # Calculate lateness penalty
            lateness = max(0, arrival_time - deadlines[candidate])
            
            # Compute score (lower is better)
            score = travel_time + urgency_weight * lateness
            
            if score < best_score:
                best_score = score
                best_node = candidate
        
        # Move to the best node
        if best_node is not None:
            distance = _calculate_distance(locations[current_node], locations[best_node])
            traffic = traffic_levels[current_node, best_node]
            travel_time = distance * traffic
            
            current_time += travel_time
            current_node = best_node
            visited[best_node] = True
            route.append(best_node)
            
            # Track delay
            lateness = max(0, current_time - deadlines[best_node])
            total_delay += lateness
    
    # Return to depot
    distance = _calculate_distance(locations[current_node], locations[0])
    traffic = traffic_levels[current_node, 0]
    travel_time = distance * traffic
    current_time += travel_time
    route.append(0)
    
    return route, current_time, total_delay


def _calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1 (np.ndarray): First point (x, y)
        point2 (np.ndarray): Second point (x, y)
    
    Returns:
        float: Euclidean distance
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))


def evaluate_route(route: List[int],
                   locations: np.ndarray,
                   deadlines: np.ndarray,
                   traffic_levels: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate a given route's performance.
    
    Args:
        route (List[int]): List of node indices
        locations (np.ndarray): Node locations
        deadlines (np.ndarray): Node deadlines
        traffic_levels (np.ndarray): Traffic multipliers
    
    Returns:
        tuple: (total_time, total_delay)
    """
    current_time = 0.0
    total_delay = 0.0
    
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        
        # Calculate travel time
        distance = _calculate_distance(locations[from_node], locations[to_node])
        traffic = traffic_levels[from_node, to_node]
        travel_time = distance * traffic
        current_time += travel_time
        
        # Check lateness
        if to_node != 0:  # Skip depot
            lateness = max(0, current_time - deadlines[to_node])
            total_delay += lateness
    
    return current_time, total_delay


class GreedyHeuristic:
    """
    Class wrapper for the greedy heuristic logic.
    """
    
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
    
    def solve(self, locations: np.ndarray, traffic_matrix: np.ndarray) -> List[int]:
        """
        Solve routing problem using greedy heuristic.
        
        Args:
            locations (np.ndarray): Node locations
            traffic_matrix (np.ndarray): Traffic multipliers
            
        Returns:
            list: Route indices
        """
        # Mock deadlines for heuristic (not critical for visualization flow)
        deadlines = np.full(len(locations), 1000.0) 
        
        route, _, _ = heuristic_route(locations, deadlines, traffic_matrix)
        return route
