"""
Tests for the Hybrid Controller.

Run with: python test_hybrid.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from hybrid_controller import HybridController
from environment import DynamicDeliveryEnv
from agent import DQNAgent


def test_complexity_score():
    """Test complexity scoring function."""
    controller = HybridController(mode="adaptive")
    env = DynamicDeliveryEnv(num_nodes=5, seed=42)
    state = env.reset()
    
    complexity = controller.compute_complexity_score(state)
    
    assert 0 <= complexity <= 1, f"Complexity should be in [0,1], got {complexity}"
    assert isinstance(complexity, (float, np.float64)), "Complexity should be float"
    
    print("✓ test_complexity_score passed")


def test_heuristic_scores():
    """Test heuristic scoring computation."""
    controller = HybridController(mode="adaptive")
    env = DynamicDeliveryEnv(num_nodes=5, seed=42)
    state = env.reset()
    
    scores = controller.compute_heuristic_scores(state)
    
    assert len(scores) == 6, "Should have scores for all nodes"
    assert scores.min() >= -1e9, "Invalid nodes should have very negative scores"
    assert scores[state['current_node']] < 0, "Current node should be invalid"
    
    print("✓ test_heuristic_scores passed")


def test_fast_mode():
    """Test fast (heuristic-only) mode."""
    controller = HybridController(mode="fast")
    env = DynamicDeliveryEnv(num_nodes=3, seed=42)
    state = env.reset()
    
    action, policy = controller.select_action(state, agent=None)
    
    assert policy == "heuristic", f"Expected heuristic policy, got {policy}"
    assert state['valid_actions'][action], f"Selected invalid action {action}"
    
    print("✓ test_fast_mode passed")


def test_adaptive_mode():
    """Test adaptive mode complexity-based switching."""
    controller = HybridController(mode="adaptive")
    env = DynamicDeliveryEnv(num_nodes=5, seed=42)
    
    # Create simple agent for testing
    action_size = 6
    state_size = 6 + 6 + 6 + 1
    agent = DQNAgent(state_size, action_size)
    
    state = env.reset()
    action, policy = controller.select_action(state, agent)
    
    assert policy in ["heuristic", "hybrid"], f"Expected heuristic or hybrid, got {policy}"
    assert state['valid_actions'][action], f"Selected invalid action {action}"
    
    print("✓ test_adaptive_mode passed")


def test_controller_modes():
    """Test all controller modes execute without errors."""
    env = DynamicDeliveryEnv(num_nodes=3, seed=42)
    action_size = 4
    state_size = 4 + 4 + 4 + 1
    agent = DQNAgent(state_size, action_size)
    
    modes = ["fast", "adaptive", "rl"]
    
    for mode in modes:
        controller = HybridController(mode=mode)
        state = env.reset()
        
        try:
            action, policy = controller.select_action(state, agent if mode != "fast" else None)
            assert state['valid_actions'][action], f"Mode {mode} selected invalid action"
        except Exception as e:
            raise AssertionError(f"Mode {mode} failed: {e}")
    
    print("✓ test_controller_modes passed")


def test_heuristic_bias():
    """Test that heuristic bias affects guided actions."""
    env = DynamicDeliveryEnv(num_nodes=5, seed=42)
    action_size = 6
    state_size = 6 + 6 + 6 + 1
    agent = DQNAgent(state_size, action_size)
    
    # Test with different bias values
    controller_low = HybridController(mode="adaptive", heuristic_bias=0.1)
    controller_high = HybridController(mode="adaptive", heuristic_bias=0.5)
    
    state = env.reset()
    
    # Both should select valid actions
    action1, _ = controller_low.select_action(state, agent)
    action2, _ = controller_high.select_action(state, agent)
    
    assert state['valid_actions'][action1], "Low bias selected invalid action"
    assert state['valid_actions'][action2], "High bias selected invalid action"
    
    print("✓ test_heuristic_bias passed")


if __name__ == "__main__":
    print("\nRunning hybrid controller tests...\n")
    test_complexity_score()
    test_heuristic_scores()
    test_fast_mode()
    test_adaptive_mode()
    test_controller_modes()
    test_heuristic_bias()
    print("\n✓ All hybrid controller tests passed!")
