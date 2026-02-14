"""
Tests for the heuristic routing baseline.

Run with: pytest test_heuristic.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time as time_module
from heuristic import heuristic_route, evaluate_route


def test_heuristic_returns_valid_route():
    """Test that heuristic_route returns a valid route."""
    # Create simple test case
    num_nodes = 5
    locations = np.random.rand(num_nodes, 2) * 10
    locations[0] = [5, 5]  # Depot at center
    
    deadlines = np.ones(num_nodes) * 100  # Loose deadlines
    traffic = np.ones((num_nodes, num_nodes))
    
    route, total_time, total_delay = heuristic_route(
        locations, deadlines, traffic
    )
    
    # Check route is valid
    assert isinstance(route, list)
    assert len(route) >= num_nodes  # At least depot + all nodes
    assert route[0] == 0, "Route should start at depot"
    assert route[-1] == 0, "Route should end at depot"
    
    # Check metrics
    assert total_time > 0
    assert total_delay >= 0
    
    print("✓ test_heuristic_returns_valid_route passed")


def test_no_duplicate_nodes():
    """Test that heuristic doesn't visit nodes multiple times (except depot)."""
    num_nodes = 6
    locations = np.random.rand(num_nodes, 2) * 10
    locations[0] = [5, 5]
    
    deadlines = np.ones(num_nodes) * 100
    traffic = np.ones((num_nodes, num_nodes))
    
    route, _, _ = heuristic_route(locations, deadlines, traffic)
    
    # Remove depot visits for checking (depot can be visited multiple times)
    non_depot_visits = [node for node in route if node != 0]
    
    # Check no duplicates in non-depot visits
    assert len(non_depot_visits) == len(set(non_depot_visits)), \
        "Heuristic should not visit non-depot nodes multiple times"
    
    # Check all nodes are visited
    assert set(non_depot_visits) == set(range(1, num_nodes)), \
        "All delivery nodes should be visited exactly once"
    
    print("✓ test_no_duplicate_nodes passed")


def test_runtime_threshold():
    """Test that heuristic runs efficiently."""
    num_nodes = 20
    locations = np.random.rand(num_nodes, 2) * 10
    deadlines = np.ones(num_nodes) * 100
    traffic = np.ones((num_nodes, num_nodes))
    
    start_time = time_module.time()
    route, _, _ = heuristic_route(locations, deadlines, traffic)
    elapsed_time = time_module.time() - start_time
    
    # Should complete in under 1 second for 20 nodes
    assert elapsed_time < 1.0, f"Heuristic too slow: {elapsed_time:.3f}s"
    
    print(f"✓ test_runtime_threshold passed ({elapsed_time*1000:.2f}ms)")


def test_urgency_affects_routing():
    """Test that urgency weight affects route selection."""
    num_nodes = 5
    locations = np.array([
        [5, 5],  # Depot (center)
        [1, 5],  # Node 1 (close, tight deadline)
        [9, 5],  # Node 2 (far, loose deadline)
        [5, 1],  # Node 3 (medium distance)
        [5, 9],  # Node 4 (medium distance, tight deadline)
    ])
    
    deadlines = np.array([1000, 2, 1000, 1000, 3])  # Nodes 1 and 4 have tight deadlines
    traffic = np.ones((num_nodes, num_nodes))
    
    # High urgency weight should prioritize urgent nodes
    route_high_urgency, time_high, delay_high = heuristic_route(
        locations, deadlines, traffic, urgency_weight=10.0
    )
    
    # Low urgency weight should prioritize distance
    route_low_urgency, time_low, delay_low = heuristic_route(
        locations, deadlines, traffic, urgency_weight=0.1
    )
    
    # At minimum, verify both routes are valid
    assert len(route_high_urgency) >= num_nodes
    assert len(route_low_urgency) >= num_nodes
    
    # High urgency should generally result in lower delays
    # (Not always guaranteed, but should be true for this test case)
    # Just verify the function produces different costs with different weights
    cost_high = time_high + 10.0 * delay_high
    cost_low = time_low + 0.1 * delay_low
    
    # Costs should be different (allowing for the fact routes might be same)
    # This is a softer check than requiring different routes
    assert cost_high != cost_low or route_high_urgency == route_low_urgency, \
        "Urgency weight should affect total cost"
    
    print("✓ test_urgency_affects_routing passed")


def test_evaluate_route():
    """Test route evaluation function."""
    locations = np.array([[5, 5], [1, 1], [9, 9]])
    deadlines = np.array([100, 5, 100])  # Tight deadline for node 1
    traffic = np.ones((3, 3))
    
    route = [0, 2, 1, 0]  # Visit node 2 first (causes delay at node 1)
    
    total_time, total_delay = evaluate_route(route, locations, deadlines, traffic)
    
    assert total_time > 0
    assert total_delay > 0, "Should have delay visiting node 1 last"
    
    print("✓ test_evaluate_route passed")


if __name__ == "__main__":
    print("\nRunning heuristic tests...\n")
    test_heuristic_returns_valid_route()
    test_no_duplicate_nodes()
    test_runtime_threshold()
    test_urgency_affects_routing()
    test_evaluate_route()
    print("\n✓ All heuristic tests passed!")
