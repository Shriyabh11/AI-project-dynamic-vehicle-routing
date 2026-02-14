import numpy as np
from typing import Tuple, List


def heuristic_route(locations: np.ndarray, 
                   deadlines: np.ndarray, 
                   traffic_levels: np.ndarray,
                   urgency_weight: float = 1.5) -> Tuple[List[int], float, float]:
    num_nodes = len(locations)
    
    current_node = 0
    current_time = 0.0
    visited = np.zeros(num_nodes, dtype=bool)
    visited[0] = True
    route = [0]
    total_delay = 0.0
    
    while not visited[1:].all():
        best_node = None
        best_score = float('inf')
        
        for candidate in range(1, num_nodes):
            if visited[candidate]:
                continue
            
            distance = _calculate_distance(locations[current_node], locations[candidate])
            traffic = traffic_levels[current_node, candidate]
            travel_time = distance * traffic
            arrival_time = current_time + travel_time
            lateness = max(0, arrival_time - deadlines[candidate])
            score = travel_time + urgency_weight * lateness
            
            if score < best_score:
                best_score = score
                best_node = candidate
        
        if best_node is not None:
            distance = _calculate_distance(locations[current_node], locations[best_node])
            traffic = traffic_levels[current_node, best_node]
            travel_time = distance * traffic
            
            current_time += travel_time
            current_node = best_node
            visited[best_node] = True
            route.append(best_node)
            
            lateness = max(0, current_time - deadlines[best_node])
            total_delay += lateness
    
    distance = _calculate_distance(locations[current_node], locations[0])
    traffic = traffic_levels[current_node, 0]
    travel_time = distance * traffic
    current_time += travel_time
    route.append(0)
    
    return route, current_time, total_delay


def _calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))


def evaluate_route(route: List[int],
                   locations: np.ndarray,
                   deadlines: np.ndarray,
                   traffic_levels: np.ndarray) -> Tuple[float, float]:
    current_time = 0.0
    total_delay = 0.0
    
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        
        distance = _calculate_distance(locations[from_node], locations[to_node])
        traffic = traffic_levels[from_node, to_node]
        travel_time = distance * traffic
        current_time += travel_time
        
        if to_node != 0:
            lateness = max(0, current_time - deadlines[to_node])
            total_delay += lateness
    
    return current_time, total_delay


class GreedyHeuristic:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
    
    def solve(self, locations: np.ndarray, traffic_matrix: np.ndarray) -> List[int]:
        deadlines = np.full(len(locations), 1000.0) 
        route, _, _ = heuristic_route(locations, deadlines, traffic_matrix)
        return route
