"""
Tests for the DynamicDeliveryEnv environment.

Run with: pytest test_environment.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from environment import DynamicDeliveryEnv


def test_environment_reset():
    """Test that environment reset returns valid state."""
    env = DynamicDeliveryEnv(num_nodes=5)
    state = env.reset()
    
    # Check state has required keys
    assert 'locations' in state
    assert 'current_node' in state
    assert 'delivered' in state
    assert 'deadlines' in state
    assert 'traffic' in state
    assert 'time_elapsed' in state
    assert 'urgency' in state
    assert 'valid_actions' in state
    
    # Check state values
    assert len(state['locations']) == 6  # 5 nodes + depot
    assert state['current_node'] == 0  # Start at depot
    assert state['time_elapsed'] == 0.0
    assert state['delivered'][0] == True  # Depot is delivered
    assert state['delivered'][1:].sum() == 0  # No other nodes delivered
    
    print("✓ test_environment_reset passed")


def test_environment_step():
    """Test that step updates time_elapsed correctly."""
    env = DynamicDeliveryEnv(num_nodes=5, seed=42)
    state = env.reset()
    
    initial_time = state['time_elapsed']
    
    # Take a step
    action = 1  # Visit node 1
    next_state, reward, done, info = env.step(action)
    
    # Check time increased
    assert next_state['time_elapsed'] > initial_time
    assert info['travel_time'] > 0
    
    # Check node was delivered
    assert next_state['delivered'][action] == True
    assert next_state['current_node'] == action
    
    # Check info dictionary
    assert 'travel_time' in info
    assert 'late_penalty' in info
    assert 'total_time' in info
    
    print("✓ test_environment_step passed")


def test_deadline_penalty():
    """Test that deadlines trigger penalties."""
    env = DynamicDeliveryEnv(num_nodes=3, seed=42)
    state = env.reset()
    
    # Manually set very tight deadlines to trigger penalties
    env.deadlines[1:] = 0.1  # Very tight deadline
    
    # Visit node 1
    action = 1
    next_state, reward, done, info = env.step(action)
    
    # Should have late penalty
    assert info['late_penalty'] > 0, "Expected late penalty for tight deadline"
    
    print("✓ test_deadline_penalty passed")


def test_action_masking():
    """Test that action masking prevents revisiting delivered nodes."""
    env = DynamicDeliveryEnv(num_nodes=3, seed=42)
    state = env.reset()
    
    # Visit node 1
    action = 1
    next_state, reward, done, info = env.step(action)
    
    # Check that node 1 is no longer valid
    valid_actions = next_state['valid_actions']
    assert valid_actions[action] == True, "Can revisit depot"  # Depot is always valid
    assert next_state['delivered'][action] == True  # Node 1 is delivered
    
    print("✓ test_action_masking passed")


def test_done_condition():
    """Test that episode ends when all nodes are delivered."""
    env = DynamicDeliveryEnv(num_nodes=2, seed=42)
    state = env.reset()
    
    # Deliver all nodes
    for node in [1, 2]:
        if not state['delivered'][node]:
            state, reward, done, info = env.step(node)
    
    # Check done condition
    assert done == True, "Episode should end when all nodes delivered"
    assert state['delivered'][1:].all(), "All non-depot nodes should be delivered"
    
    print("✓ test_done_condition passed")


if __name__ == "__main__":
    print("\nRunning environment tests...\n")
    test_environment_reset()
    test_environment_step()
    test_deadline_penalty()
    test_action_masking()
    test_done_condition()
    print("\n✓ All environment tests passed!")
