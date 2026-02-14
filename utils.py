"""
Utility functions for the Dynamic Routing project.

This module provides device handling and helper functions.
"""

import torch
import numpy as np

# CPU-first device handling - GPU is optional
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device():
    """
    Get the current device (CPU or CUDA).
    
    Returns:
        torch.device: The device to use for tensors and models
    """
    return device

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        point1 (tuple or np.array): (x, y) coordinates
        point2 (tuple or np.array): (x, y) coordinates
    
    Returns:
        float: Euclidean distance
    """
    if isinstance(point1, (list, tuple)):
        point1 = np.array(point1)
    if isinstance(point2, (list, tuple)):
        point2 = np.array(point2)
    
    return np.sqrt(np.sum((point1 - point2) ** 2))

def normalize_coordinates(coords, scale=1.0):
    """
    Normalize coordinates to [0, scale] range.
    
    Args:
        coords (np.array): Array of coordinates
        scale (float): Maximum value for normalization
    
    Returns:
        np.array: Normalized coordinates
    """
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    range_val = coords_max - coords_min
    range_val[range_val == 0] = 1  # Avoid division by zero
    
    normalized = (coords - coords_min) / range_val * scale
    return normalized

def print_route_info(route, total_time, total_delay, agent_type="Agent"):
    """
    Print formatted route information.
    
    Args:
        route (list): List of node indices in visit order
        total_time (float): Total travel time
        total_delay (float): Total lateness penalty
        agent_type (str): Type of agent (Heuristic/DQN)
    """
    print(f"\n{agent_type} Route:")
    print(f"  Nodes visited: {' → '.join(map(str, route))}")
    print(f"  Total travel time: {total_time:.4f}")
    print(f"  Total delay: {total_delay:.4f}")
    print(f"  Combined cost: {total_time + total_delay:.4f}")
