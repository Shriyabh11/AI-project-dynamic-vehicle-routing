"""
Tests for the DQN agent.

Run with: pytest test_agent.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from agent import DQNAgent, DQNNetwork, ReplayBuffer
from environment import DynamicDeliveryEnv


def test_network_forward_pass():
    """Test that DQN forward pass returns correct shape."""
    state_size = 50
    action_size = 10
    batch_size = 32
    
    network = DQNNetwork(state_size, action_size)
    
    # Create random input
    x = torch.randn(batch_size, state_size)
    
    # Forward pass
    q_values = network(x)
    
    # Check output shape
    assert q_values.shape == (batch_size, action_size), \
        f"Expected shape {(batch_size, action_size)}, got {q_values.shape}"
    
    print("✓ test_network_forward_pass passed")


def test_replay_buffer():
    """Test replay buffer operations."""
    buffer = ReplayBuffer(capacity=100)
    
    # Add some transitions
    for i in range(50):
        state = np.random.rand(10)
        action = i % 5
        reward = np.random.rand()
        next_state = np.random.rand(10)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
    
    # Check buffer size
    assert len(buffer) == 50
    
    # Sample batch
    batch_size = 10
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    # Check batch shapes
    assert states.shape == (batch_size, 10)
    assert len(actions) == batch_size
    assert len(rewards) == batch_size
    
    print("✓ test_replay_buffer passed")


def test_epsilon_greedy_selection():
    """Test that epsilon-greedy selects valid actions."""
    env = DynamicDeliveryEnv(num_nodes=5, seed=42)
    state = env.reset()
    
    action_size = 6  # 5 nodes + depot
    state_size = 6 + 6 + 6 + 1  # onehot + delivered + urgency + time
    
    agent = DQNAgent(state_size, action_size)
    
    # Test with high exploration (should select random valid actions)
    agent.epsilon = 1.0
    for _ in range(10):
        action = agent.select_action(state)
        assert state['valid_actions'][action], \
            f"Selected invalid action {action}"
    
    # Test with no exploration (should select Q-value based actions)
    agent.epsilon = 0.0
    for _ in range(10):
        action = agent.select_action(state)
        assert state['valid_actions'][action], \
            f"Selected invalid action {action}"
    
    print("✓ test_epsilon_greedy_selection passed")


def test_state_flattening():
    """Test that state dictionary is correctly flattened."""
    env = DynamicDeliveryEnv(num_nodes=3, seed=42)
    state = env.reset()
    
    action_size = 4  # 3 nodes + depot
    state_size = 4 + 4 + 4 + 1  # onehot + delivered + urgency + time
    
    agent = DQNAgent(state_size, action_size)
    flat_state = agent.flatten_state(state)
    
    # Check flattened state shape
    assert flat_state.shape == (state_size,), \
        f"Expected shape {(state_size,)}, got {flat_state.shape}"
    
    # Check values are reasonable
    assert not np.isnan(flat_state).any(), "Flattened state contains NaN"
    assert not np.isinf(flat_state).any(), "Flattened state contains Inf"
    
    print("✓ test_state_flattening passed")


def test_training_step():
    """Test that agent can perform a training step."""
    state_size = 20
    action_size = 5
    
    agent = DQNAgent(state_size, action_size)
    
    # Add some experiences to replay buffer
    for _ in range(100):
        state = np.random.rand(state_size)
        action = np.random.randint(0, action_size)
        reward = np.random.rand()
        next_state = np.random.rand(state_size)
        done = False
        
        agent.memory.push(state, action, reward, next_state, done)
    
    # Perform training step (should not crash)
    agent.train_step(batch_size=32)
    
    print("✓ test_training_step passed")


def test_target_network_update():
    """Test that target network can be updated."""
    state_size = 20
    action_size = 5
    
    agent = DQNAgent(state_size, action_size)
    
    # Get initial target network weights
    initial_weights = agent.target_network.state_dict()['network.0.weight'].clone()
    
    # Modify Q-network weights
    with torch.no_grad():
        agent.q_network.network[0].weight += 1.0
    
    # Update target network
    agent.update_target_network()
    
    # Check target network was updated
    updated_weights = agent.target_network.state_dict()['network.0.weight']
    
    assert not torch.equal(initial_weights, updated_weights), \
        "Target network was not updated"
    
    print("✓ test_target_network_update passed")


def test_epsilon_decay():
    """Test that epsilon decays correctly."""
    agent = DQNAgent(state_size=20, action_size=5)
    
    initial_epsilon = agent.epsilon
    agent.decay_epsilon()
    
    assert agent.epsilon < initial_epsilon, "Epsilon should decrease"
    assert agent.epsilon >= agent.epsilon_end, "Epsilon should not go below minimum"
    
    # Decay many times
    for _ in range(1000):
        agent.decay_epsilon()
    
    assert agent.epsilon == agent.epsilon_end, \
        f"Epsilon should reach minimum: {agent.epsilon} vs {agent.epsilon_end}"
    
    print("✓ test_epsilon_decay passed")


if __name__ == "__main__":
    print("\nRunning agent tests...\n")
    test_network_forward_pass()
    test_replay_buffer()
    test_epsilon_greedy_selection()
    test_state_flattening()
    test_training_step()
    test_target_network_update()
    test_epsilon_decay()
    print("\n✓ All agent tests passed!")
