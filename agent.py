"""
Advanced DQN Agent for dynamic delivery routing.

This module implements state-of-the-art RL improvements over the standard DQN:
1. Dueling Network: Separates Value and Advantage streams for better state evaluation
2. Double DQN: Reduces Q-value overestimation by decoupling selection and evaluation
3. Prioritized Experience Replay (PER): Focuses learning on high-error transitions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple
from collections import deque

from utils import get_device

class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN Network Architecture.
    
    Splits the network into two streams:
    - Value stream V(s): Estimates value of the state itself
    - Advantage stream A(s,a): Estimates advantage of each action
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DuelingDQNNetwork, self).__init__()
        
        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine V and A
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return qvals


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) Buffer.
    
    Stores transitions with priorities based on TD-errors.
    Samples important transitions more frequently.
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        Args:
            capacity (int): Max buffer size
            alpha (float): Priority exponent (0 = uniform, 1 = full priority)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add transition with max priority."""
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample batch based on priorities.
        
        Args:
            batch_size (int): Batch size
            beta (float): Importance sampling exponent (correction factor)
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.buffer) == 0:
            return None
        
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), indices, weights)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities after TD-error calculation."""
        self.priorities[indices] = priorities + 1e-5  # Add small epsilon
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Advanced DQNAgent using Dueling DQN + Double DQN + PER.
    Renamed to DQNAgent to maintain compatibility with existing code.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        
        self.device = get_device()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Dueling Networks
        self.q_network = DuelingDQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DuelingDQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Priority Replay
        self.memory = PrioritizedReplayBuffer()
        
        # PER Beta (anneals from 0.4 to 1.0)
        self.beta = 0.4
        self.beta_increment = 0.001
    
    def flatten_state(self, state: Dict) -> np.ndarray:
        """
        Convert state dictionary to flat numpy array.
        Includes location and traffic data.
        """
        current_node_onehot = np.zeros(self.action_size)
        current_node_onehot[state['current_node']] = 1.0
        
        delivered = state['delivered'].astype(float)
        urgency = state['urgency']
        time_elapsed = np.array([state['time_elapsed']])
        
        flat_state = np.concatenate([
            current_node_onehot,
            delivered,
            urgency,
            time_elapsed,
            state['locations'].flatten(),
            state['traffic'][state['current_node']]
        ])
        return flat_state
    
    def select_action(self, state: Dict) -> int:
        valid_actions = state['valid_actions']
        valid_indices = np.where(valid_actions)[0]
        
        if len(valid_indices) == 0:
            raise RuntimeError("No valid actions available!")
        
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_indices)
        
        flat_state = self.flatten_state(state)
        state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        # Mask invalid actions
        q_values[~valid_actions] = -1e9
        
        selected_action = np.argmax(q_values)
        return selected_action
    
    def train_step(self, batch_size: int = 32):
        if len(self.memory) < batch_size:
            return
        
        # Sample with PER
        samples = self.memory.sample(batch_size, self.beta)
        states, actions, rewards, next_states, dones, indices, weights = samples
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Double DQN Logic
        # 1. Select best action using Online Network
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            # 2. Evaluate that action using Target Network
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # TD Error
        td_errors = torch.abs(target_q_values - current_q_values).detach().cpu().numpy()
        
        # Weighted MSE Loss
        loss = (weights * (current_q_values - target_q_values) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
