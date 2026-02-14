"""
Hybrid Heuristic + Deep RL Controller for Dynamic Routing.

This module implements a novel hybrid decision intelligence system that combines:
- Fast heuristic reasoning for simple scenarios
- Deep RL adaptation for complex traffic patterns

Key Innovation:
"Hybrid Lightweight Decision Intelligence combining heuristic reasoning 
with Deep RL adaptation while maintaining CPU deployability."

Design Philosophy:
- FAST MODE: CPU-only heuristic routing (instant deployment)
- ADAPTIVE MODE: Intelligent hybrid switching based on environment complexity
- RL MODE: Pure DQN for maximum learning capability
"""

import numpy as np
import torch
from typing import Dict, Tuple
from .heuristic import heuristic_route
from tests.safety import SafetyMonitor


class HybridController:
    """
    Adaptive controller that intelligently combines heuristic and DQN policies.
    
    Decision Logic:
    - Low complexity (simple traffic, loose deadlines) → Heuristic
    - High complexity (variable traffic, tight deadlines) → DQN
    - Mixed scenarios → Heuristic-guided DQN (Q-values biased by heuristic scores)
    
    Research Alignment:
    - Jiang (2025): Stability-aware decision making
    - Aslan Yildiz (2025): Lightweight action filtering for dynamic traffic
    """
    
    
    def __init__(self, mode="adaptive", heuristic_bias=0.15, use_action_mask=True, top_k=3, stability_threshold=0.15, n_nodes=5):
        """
        Initialize hybrid controller.
        
        Args:
            mode (str): Operation mode - "fast", "adaptive", "rl", or "stable_fast"
            heuristic_bias (float): Weight for heuristic guidance in Q-values (0.1-0.2)
            use_action_mask (bool): Enable heuristic action masking (RESEARCH NOVELTY)
            top_k (int): Number of top heuristic actions to consider (3-5 recommended)
            stability_threshold (float): Traffic stability threshold for stable_fast mode
            n_nodes (int): Number of nodes (for DQN initialization)
        """
        self.mode = mode
        self.heuristic_bias = heuristic_bias
        self.use_action_mask = use_action_mask
        self.top_k = top_k
        self.stability_threshold = stability_threshold
        
        # Adaptive thresholds
        self.traffic_variance_threshold = 0.15
        self.urgency_threshold = 0.5
        self.complexity_threshold = 0.3  # Below this: heuristic, above: RL
        
        # Track stability (Jiang 2025-inspired)
        # Initialize with default value to avoid "UNKNOWN" on first steps
        self.travel_time_history = [1.0]
        
        # Initialize DQN agent if not in fast_mode
        self.agent = None
        if mode in ["adaptive", "rl"]:
            try:
                from .agent import DQNAgent
                state_dim = n_nodes * 3 + 2  # Simplified state
                action_dim = n_nodes
                self.agent = DQNAgent(state_dim, action_dim)
                # Force CPU device to avoid CUDA/CPU mismatch
                self.agent.device = 'cpu'
                self.agent.q_network = self.agent.q_network.to('cpu')
                self.agent.target_network = self.agent.target_network.to('cpu')
            except Exception as e:
                print(f"DQN agent init skipped: {e}")
                self.agent = None
        
        # Safety Monitor (integrated from safety.py)
        self.safety_monitor = SafetyMonitor(speed_limit=60.0, variance_threshold=0.5)
        self.prev_node = 0  # Track previous node for speed estimation
        self.last_distance = 0.0  # Track last segment distance
        self.last_travel_time = 0.0  # Track last travel time
    
    def compute_complexity_score(self, state: Dict) -> float:
        """
        Compute environment complexity score to guide policy selection.
        
        High complexity = variable traffic + tight deadlines
        Low complexity = uniform traffic + loose deadlines
        
        Args:
            state (Dict): Environment state
        
        Returns:
            float: Complexity score (0=simple, 1=complex)
        """
        # Traffic variability
        traffic = state['traffic']
        traffic_variance = np.var(traffic)
        traffic_score = min(traffic_variance / 0.3, 1.0)  # Normalize
        
        # Deadline urgency (how many nodes have tight deadlines)
        urgency = state['urgency']
        delivered = state['delivered']
        undelivered_urgency = urgency[~delivered]
        
        if len(undelivered_urgency) > 0:
            avg_urgency = np.mean(undelivered_urgency)
            urgency_score = min(avg_urgency / 10.0, 1.0)  # Normalize
        else:
            urgency_score = 0.0
        
        # Combined complexity (weighted average)
        complexity = 0.6 * traffic_score + 0.4 * urgency_score
        
        return complexity
    
    def compute_stability(self, recent_window=5) -> Tuple[float, str]:
        """
        Compute traffic stability score based on travel time variance.
        
        RESEARCH ALIGNMENT (Jiang 2025): Stability-aware routing decisions.
        Low variance = stable traffic → use faster heuristic
        High variance = unstable traffic → use adaptive RL
        
        Args:
            recent_window (int): Number of recent steps to analyze
        
        Returns:
            tuple: (stability_score, stability_category)
                   stability_category ∈ {"LOW", "MEDIUM", "HIGH"}
        """
        # Handle edge case: if only one value, assume low stability (stable)
        if len(self.travel_time_history) < 2:
            return 0.0, "LOW"
        
        # Use recent window for stability
        recent_times = self.travel_time_history[-recent_window:]
        
        # Compute variance (lower = more stable)
        variance = np.var(recent_times)
        
        # Normalize to 0-1 range (assuming max variance ~0.5)
        stability_score = min(variance / 0.5, 1.0)
        
        # Categorize
        if stability_score < 0.1:
            category = "LOW"  # Stable traffic
        elif stability_score < 0.3:
            category = "MEDIUM"  # Moderate variability
        else:
            category = "HIGH"  # Unstable traffic
        
        return stability_score, category
    
    def compute_heuristic_scores(self, state: Dict) -> np.ndarray:
        """
        Compute heuristic preference scores for each action.
        
        Args:
            state (Dict): Environment state
        
        Returns:
            np.ndarray: Heuristic scores for each action (higher = better)
        """
        locations = state['locations']
        current_node = state['current_node']
        deadlines = state['deadlines']
        traffic = state['traffic']
        time_elapsed = state['time_elapsed']
        delivered = state['delivered']
        
        num_nodes = len(locations)
        scores = np.zeros(num_nodes)
        
        # Score each undelivered node
        for node in range(num_nodes):
            if delivered[node] and node != 0:
                scores[node] = -1e9  # Invalid
                continue
            
            if node == current_node:
                scores[node] = -1e9  # Don't stay at current
                continue
            
            # Distance to node
            distance = np.linalg.norm(locations[current_node] - locations[node])
            travel_time = distance * traffic[current_node, node]
            
            # Urgency penalty
            arrival_time = time_elapsed + travel_time
            lateness = max(0, arrival_time - deadlines[node])
            
            # Heuristic score: lower is better, so negate
            # (we want higher scores to be better for consistency)
            raw_score = travel_time + 2.0 * lateness
            scores[node] = -raw_score
        
        # Normalize scores to [0, 1] range for biasing
        # BUT preserve invalid markers
        valid_mask = scores > -1e8  # Valid scores
        if valid_mask.any():
            valid_scores = scores[valid_mask]
            if valid_scores.max() > valid_scores.min():
                # Normalize only valid scores
                normalized = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min())
                scores[valid_mask] = normalized
        
        return scores
    
    def get_top_k_actions(self, state: Dict, k: int = 3) -> np.ndarray:
        """
        Get top-k actions based on heuristic scores.
        
        RESEARCH NOVELTY: Heuristic-guided action filtering for RL.
        This restricts RL agent to explore only promising actions,
        reducing search space and improving sample efficiency.
        
        Args:
            state (Dict): Environment state
            k (int): Number of top actions to return
        
        Returns:
            np.ndarray: Boolean mask where True = action in top-k
        """
        scores = self.compute_heuristic_scores(state)
        valid_actions = state['valid_actions']
        
        # Only consider valid actions
        scores_valid = scores.copy()
        scores_valid[~valid_actions] = -1e9
        
        # Get top-k indices
        top_k_indices = np.argsort(scores_valid)[-k:]
        
        # Create mask
        mask = np.zeros(len(scores), dtype=bool)
        mask[top_k_indices] = True
        
        # Ensure at least valid actions if k is too large
        mask = mask & valid_actions
        
        return mask & valid_actions
    
    def update_stability(self, travel_time: float):
        """
        Update travel time history for stability tracking.
        
        Args:
            travel_time (float): Latest travel time
        """
        self.travel_time_history.append(travel_time)
        # Keep only recent history (last 20 steps)
        if len(self.travel_time_history) > 20:
            self.travel_time_history.pop(0)
    
    def select_action(self, state: Dict, agent=None, travel_time: float = None) -> Tuple[int, str, str, float, float, str]:
        """
        Select action using hybrid policy with explainability.
        
        Args:
            state (Dict): Environment state
            agent: DQN agent (required for adaptive/rl modes)
            travel_time (float): Optional, latest travel time for stability tracking
        
        Returns:
            tuple: (action, policy_used, decision_reason, complexity_score, 
                    stability_score, stability_category)
        """
        # Update stability tracking if travel time provided
        if travel_time is not None:
            self.update_stability(travel_time)
            
            # Safety / Speed Estimation
            if self.safety_monitor:
                # Calculate distance from prev_node to current_node
                locations = state['locations']
                prev_loc = locations[self.prev_node]
                curr_loc = locations[state['current_node']]
                dist = np.linalg.norm(prev_loc - curr_loc)
                
                # Estimate speed (assuming dist=1.0 ~ 1km, time=1.0 ~ 1min for demo)
                speed = self.safety_monitor.estimate_speed(dist, travel_time)
                
                # Check safety
                risks = self.safety_monitor.check_safety(state, 0, speed) # Action 0 dummy for check
                
                if risks['overspeed']:
                     print(f"[Safety] WARNING: Overspeed detected ({speed:.1f} km/h)")
                
                if risks['congestion_risk']:
                     print(f"[Safety] High traffic volatility → increased safety/delay risk")
                
                if risks.get('fallback_triggered', False):
                     print(f"[Safety] Fallback triggered → Using HEURISTIC")
                     self.prev_node = state['current_node']
                     return self._select_heuristic_action(state, np.where(state['valid_actions'])[0]), "heuristic", "Safety Fallback", 0.0, stability_score, stability_category

        self.prev_node = state['current_node']
        
        stability_score, stability_category = self.compute_stability()
        valid_actions = state['valid_actions']
        valid_indices = np.where(valid_actions)[0]
        
        if len(valid_indices) == 0:
            raise RuntimeError("No valid actions available!")
        
        # Compute complexity for decision reasoning
        complexity = self.compute_complexity_score(state)
        
        # FAST MODE: Heuristic only (CPU deployment)
        if self.mode == "fast":
            action = self._select_heuristic_action(state, valid_indices)
            reason = "Fast mode: Pure heuristic (CPU-optimized)"
            return action, "heuristic", reason, complexity, stability_score, stability_category
        
        # STABLE_FAST MODE: Stability-aware switching (Jiang 2025-inspired)
        if self.mode == "stable_fast":
            if stability_score < self.stability_threshold or stability_category == "LOW":
                action = self._select_heuristic_action(state, valid_indices)
                reason = f"Stable traffic ({stability_category}, {stability_score:.3f}): Heuristic selected"
                return action, "heuristic", reason, complexity, stability_score, stability_category
            else:
                if agent is None:
                    action = self._select_heuristic_action(state, valid_indices)
                    reason = "Unstable traffic but no agent: Fallback to heuristic"
                    return action, "heuristic", reason, complexity, stability_score, stability_category
                action = self._select_guided_action(state, agent)
                reason = f"Unstable traffic ({stability_category}, {stability_score:.3f}): Adaptive RL"
                return action, "hybrid", reason, complexity, stability_score, stability_category
        
        # RL MODE: Pure DQN (with optional action masking)
        if self.mode == "rl":
            if agent is None:
                raise ValueError("Agent required for RL mode")
            action = self._select_rl_action(state, agent)
            mask_info = f" (top-{self.top_k} masked)" if self.use_action_mask else ""
            reason = f"RL mode: Pure DQN{mask_info}"
            return action, "dqn", reason, complexity, stability_score, stability_category
        
        # ADAPTIVE MODE: Intelligence hybrid switching
        if self.mode == "adaptive":
            # Low complexity → Heuristic (faster, simpler)
            if complexity < self.complexity_threshold:
                action = self._select_heuristic_action(state, valid_indices)
                reason = f"Low complexity ({complexity:.2f} < {self.complexity_threshold}): Heuristic selected"
                return action, "heuristic", reason, complexity, stability_score, stability_category
            
            # High complexity → RL-based decision
            if agent is None:
                action = self._select_heuristic_action(state, valid_indices)
                reason = "No agent available: Fallback to heuristic"
                return action, "heuristic", reason, complexity, stability_score, stability_category
            
            action = self._select_guided_action(state, agent)
            mask_info = f" with top-{self.top_k} masking" if self.use_action_mask else ""
            reason = f"High complexity ({complexity:.2f} ≥ {self.complexity_threshold}): Hybrid RL{mask_info}"
            return action, "hybrid", reason, complexity, stability_score, stability_category
        
        raise ValueError(f"Unknown mode: {self.mode}")
    
    def _select_heuristic_action(self, state: Dict, valid_indices: np.ndarray) -> int:
        """Select action using pure heuristic scoring."""
        scores = self.compute_heuristic_scores(state)
        
        # Mask invalid actions
        scores[~state['valid_actions']] = -1e9
        
        return np.argmax(scores)
    
    def _select_rl_action(self, state: Dict, agent) -> int:
        """
        Select action using DQN with optional top-k action masking.
        
        RESEARCH NOVELTY: Heuristic-guided action space reduction.
        """
        # Get DQN Q-values
        flat_state = agent.flatten_state(state)
        state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(agent.device)
        
        with torch.no_grad():
            q_values = agent.q_network(state_tensor).cpu().numpy()[0]
        
        # Apply heuristic action masking if enabled
        if self.use_action_mask:
            top_k_mask = self.get_top_k_actions(state, k=self.top_k)
            # Mask actions not in top-k
            q_values[~top_k_mask] = -1e9
        else:
            # Mask only invalid actions
            q_values[~state['valid_actions']] = -1e9
        
        return np.argmax(q_values)
    
    def _select_guided_action(self, state: Dict, agent) -> int:
        """
        Select action using heuristic-guided DQN.
        
        RESEARCH NOVELTY: Combines Q-value biasing with action space reduction.
        This dual approach provides both guidance (bias) and restriction (masking).
        """
        # Get DQN Q-values
        flat_state = agent.flatten_state(state)
        state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(agent.device)
        
        with torch.no_grad():
            q_values = agent.q_network(state_tensor).cpu().numpy()[0]
        
        # Get heuristic scores
        heuristic_scores = self.compute_heuristic_scores(state)
        
        # Normalize Q-values for stable biasing
        if q_values.max() > q_values.min():
            q_norm = (q_values - q_values.min()) / (q_values.max() - q_values.min())
        else:
            q_norm = q_values
        
        # RESEARCH FEATURE 1: Q-value biasing
        guided_values = q_norm + self.heuristic_bias * heuristic_scores
        
        # RESEARCH FEATURE 2: Top-k action masking (if enabled)
        if self.use_action_mask:
            top_k_mask = self.get_top_k_actions(state, k=self.top_k)
            guided_values[~top_k_mask] = -1e9
        else:
            # Mask only invalid actions
            guided_values[~state['valid_actions']] = -1e9
        
        return np.argmax(guided_values)
    
    def get_stats(self) -> Dict:
        """Return controller statistics."""
        return {
            'mode': self.mode,
            'heuristic_bias': self.heuristic_bias,
            'traffic_threshold': self.traffic_variance_threshold,
            'urgency_threshold': self.urgency_threshold
        }

    def select_policy(self, state: Dict, recent_travel_times: list = None) -> str:
        """
        Determine which policy to use based on state and history.
        
        Args:
            state (Dict): Environment state
            recent_travel_times (list): List of recent travel times (floats)
            
        Returns:
            str: "heuristic" or "dqn"
        """
        # Update history if provided
        if recent_travel_times:
             # extend history with new values
             # Note: current implementation appends one by one, 
             # here we just extend and keep last 20
             self.travel_time_history.extend(recent_travel_times)
             if len(self.travel_time_history) > 20:
                 self.travel_time_history = self.travel_time_history[-20:]

        stability_score, stability_category = self.compute_stability()
        
        # Decide based on stability
        # High stability -> Heuristic
        # Low stability -> DQN (adaptive)
        
        if stability_score < self.stability_threshold:
            print(f"[Controller] Stability={stability_score:.2f} ({stability_category}) → Using HEURISTIC")
            return "heuristic"
        else:
            print(f"[Controller] Stability={stability_score:.2f} ({stability_category}) → Using DQN")
            return "dqn"

    def check_safety(self, state: Dict, action: int) -> Dict[str, bool]:
        """
        Check safety risks for the proposed action.
        
        Args:
            state (Dict): Environment state
            action (int): Proposed action (node index)
            
        Returns:
            dict: Safety warnings {'overspeed': bool, 'congestion_risk': bool, ...}
        """
        # Estimate speed based on last segment
        estimated_speed = None
        if self.last_travel_time > 0:
            estimated_speed = self.safety_monitor.estimate_speed(
                self.last_distance, 
                self.last_travel_time
            )
        
        # Run safety checks
        risks = self.safety_monitor.check_safety(state, action, estimated_speed)
        
        return risks
    
    def set_safety_monitor(self, monitor):
        """Set custom safety monitor (for testing/demo)."""
        self.safety_monitor = monitor



def evaluate_hybrid(controller, agent, env, num_episodes=10, verbose=False):
    """
    Evaluate hybrid controller performance.
    
    Args:
        controller (HybridController): Hybrid controller
        agent: DQN agent (required for adaptive/rl modes)
        env: Environment
        num_episodes (int): Number of evaluation episodes
        verbose (bool): Print detailed statistics
    
    Returns:
        dict: Performance statistics including policy usage breakdown
    """
    total_rewards = []
    total_times = []
    policy_counts = {'heuristic': 0, 'dqn': 0, 'hybrid': 0}
    
    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_time = 0
        done = False
        steps = 0
        max_steps = env.num_nodes * 3
        
        while not done and steps < max_steps:
            action, policy = controller.select_action(state, agent)
            policy_counts[policy] += 1
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_time += info['travel_time']
            state = next_state
            steps += 1
        
        total_rewards.append(episode_reward)
        total_times.append(episode_time)
    
    stats = {
        'avg_reward': np.mean(total_rewards),
        'avg_time': np.mean(total_times),
        'policy_usage': policy_counts,
        'total_decisions': sum(policy_counts.values())
    }
    
    if verbose:
        print(f"\nHybrid Controller Stats:")
        print(f"  Mode: {controller.mode}")
        print(f"  Avg Reward: {stats['avg_reward']:.2f}")
        print(f"  Avg Time: {stats['avg_time']:.2f}")
        print(f"  Policy Usage:")
        total = stats['total_decisions']
        for policy, count in policy_counts.items():
            pct = (count / total * 100) if total > 0 else 0
            print(f"    {policy}: {count} ({pct:.1f}%)")
    
    return stats


def print_comparison_table(results):
    """
    Print formatted comparison table for multiple policies.
    
    Args:
        results (dict): Dictionary with policy names as keys and dicts containing:
                       - 'times': list of travel times
                       - 'delays': list of delays
                       - 'episodes': number of episodes
    
    Example:
        results = {
            'DQN': {'times': [25.3, 24.8, ...], 'delays': [1.2, 1.5, ...], 'episodes': 20},
            'Heuristic': {'times': [27.8, 28.1, ...], 'delays': [0.0, 0.0, ...], 'episodes': 20},
            'Hybrid': {'times': [24.9, 25.2, ...], 'delays': [0.9, 1.1, ...], 'episodes': 20}
        }
    """
    print(f"\n{'='*70}")
    print("POLICY COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Policy':<15} | {'Avg Travel Time':<20} | {'Avg Delay':<12} | Episodes")
    print(f"{'-'*70}")
    
    for policy_name, data in results.items():
        times = data['times']
        delays = data['delays']
        episodes = data['episodes']
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_delay = np.mean(delays)
        
        print(f"{policy_name:<15} | {avg_time:>6.2f} ± {std_time:<8.2f} | {avg_delay:>10.2f} | {episodes:>8}")
    
    print(f"{'='*70}\n")
