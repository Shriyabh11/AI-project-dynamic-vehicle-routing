"""
Training script for DQN-based dynamic delivery routing.

This script trains a DQN agent and supports hybrid heuristic + Deep RL evaluation.

Usage:
    python train.py                  # Default: pure DQN training + comparison
    python train.py --mode fast      # Heuristic-only evaluation (CPU deployment)
    python train.py --mode adaptive  # Hybrid adaptive controller
    python train.py --mode rl        # Pure RL evaluation
"""

import numpy as np
import argparse
from environment import DynamicDeliveryEnv
from heuristic import heuristic_route
from agent import DQNAgent
from hybrid_controller import HybridController, evaluate_hybrid, print_comparison_table
from utils import set_seed, print_route_info


def train_dqn(num_episodes=500, num_nodes=10, print_every=50):
    """
    Train DQN agent on delivery routing task.
    
    Args:
        num_episodes (int): Number of training episodes
        num_nodes (int): Number of delivery nodes
        print_every (int): Print stats every N episodes
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize environment
    env = DynamicDeliveryEnv(num_nodes=num_nodes)
    
    # Calculate state and action sizes
    action_size = num_nodes + 1  # All nodes + depot
    # State: onehot current + delivered + urgency + time + locations(2d) + traffic_row
    state_size = action_size * 6 + 1
    
    # Initialize agent
    agent = DQNAgent(state_size, action_size)
    
    # Training statistics
    episode_rewards = []
    episode_times = []
    episode_delays = []
    
    print(f"\n{'='*60}")
    print(f"Training DQN Agent on {num_nodes}-node delivery problem")
    print(f"{'='*60}\n")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_time = 0
        episode_delay = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            flat_state = agent.flatten_state(state)
            flat_next_state = agent.flatten_state(next_state)
            agent.memory.push(flat_state, action, reward, flat_next_state, done)
            
            # Train agent
            agent.train_step()
            
            # Update statistics
            episode_reward += reward
            episode_time += info['travel_time']
            episode_delay += info['late_penalty']
            
            state = next_state
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Store statistics
        episode_rewards.append(episode_reward)
        episode_times.append(episode_time)
        episode_delays.append(episode_delay)
        
        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_time = np.mean(episode_times[-print_every:])
            avg_delay = np.mean(episode_delays[-print_every:])
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Travel Time: {avg_time:.4f}")
            print(f"  Avg Delay: {avg_delay:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}\n")
    
    return agent, episode_rewards, episode_times, episode_delays


def evaluate_agent(agent, num_episodes=10, num_nodes=10, debug=False):
    """
    Evaluate trained DQN agent.
    
    Args:
        agent (DQNAgent): Trained DQN agent
        num_episodes (int): Number of evaluation episodes
        num_nodes (int): Number of delivery nodes
        debug (bool): Print debugging information
    
    Returns:
        dict: Evaluation statistics
    """
    env = DynamicDeliveryEnv(num_nodes=num_nodes)
    
    total_rewards = []
    total_times = []
    total_delays = []
    
    # Disable exploration for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    print(f"Evaluating on {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_time = 0
        episode_delay = 0
        done = False
        steps = 0
        max_steps = num_nodes * 3  # Safety limit: 3x expected steps
        
        if debug and ep == 0:
            print(f"\n  DEBUG Episode {ep+1}:")
        
        while not done and steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            if debug and ep == 0 and steps < 5:
                valid = np.where(state['valid_actions'])[0]
                delivered = np.where(state['delivered'])[0]
                print(f"    Step {steps+1}: current={state['current_node']}, action={action}, " +
                      f"valid={valid.tolist()}, delivered={delivered.tolist()}, reward={reward:.2f}")
            
            episode_reward += reward
            episode_time += info['travel_time']
            episode_delay += info['late_penalty']
            
            state = next_state
            steps += 1
        
        if steps >= max_steps:
            print(f"  ⚠ WARNING: Episode {ep+1} hit max steps limit (stuck in loop!)")
        
        total_rewards.append(episode_reward)
        total_times.append(episode_time)
        total_delays.append(episode_delay)
        
        # Progress indicator
        status = "⚠ STUCK" if steps >= max_steps else "✓"
        print(f"  {status} Episode {ep+1}/{num_episodes}: Steps={steps}, Reward={episode_reward:.2f}, Time={episode_time:.2f}")
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    return {
        'avg_reward': np.mean(total_rewards),
        'avg_time': np.mean(total_times),
        'avg_delay': np.mean(total_delays)
    }



def compare_with_heuristic(agent, num_nodes=10, num_trials=10):
    """
    Compare DQN agent with heuristic baseline.
    
    Args:
        agent (DQNAgent): Trained DQN agent
        num_nodes (int): Number of delivery nodes
        num_trials (int): Number of comparison trials
    """
    env = DynamicDeliveryEnv(num_nodes=num_nodes)
    
    dqn_times = []
    dqn_delays = []
    heuristic_times = []
    heuristic_delays = []
    
    # Disable exploration
    agent.epsilon = 0.0
    
    print(f"\nRunning {num_trials} comparison trials...")
    
    for trial in range(num_trials):
        print(f"  Trial {trial+1}/{num_trials}...", end=" ", flush=True)
        
        # Get environment state
        state = env.reset()
        locations = state['locations']
        deadlines = state['deadlines']
        traffic = state['traffic']
        
        # DQN agent
        dqn_time = 0
        dqn_delay = 0
        done = False
        steps = 0
        max_steps = num_nodes * 3  # Safety limit
        
        while not done and steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            dqn_time += info['travel_time']
            dqn_delay += info['late_penalty']
            state = next_state
            steps += 1
        
        dqn_times.append(dqn_time)
        dqn_delays.append(dqn_delay)
        
        # Heuristic baseline
        route, h_time, h_delay = heuristic_route(locations, deadlines, traffic)
        heuristic_times.append(h_time)
        heuristic_delays.append(h_delay)
        
        print(f"Done (DQN: {dqn_time+dqn_delay:.2f}, Heuristic: {h_time+h_delay:.2f})")

    
    print(f"\n{'='*60}")
    print(f"Comparison: DQN vs Heuristic ({num_trials} trials)")
    print(f"{'='*60}\n")
    
    print(f"DQN Agent:")
    print(f"  Avg Travel Time: {np.mean(dqn_times):.4f} (±{np.std(dqn_times):.4f})")
    print(f"  Avg Delay: {np.mean(dqn_delays):.4f} (±{np.std(dqn_delays):.4f})")
    print(f"  Total Cost: {np.mean(dqn_times) + np.mean(dqn_delays):.4f}\n")
    
    print(f"Heuristic Baseline:")
    print(f"  Avg Travel Time: {np.mean(heuristic_times):.4f} (±{np.std(heuristic_times):.4f})")
    print(f"  Avg Delay: {np.mean(heuristic_delays):.4f} (±{np.std(heuristic_delays):.4f})")
    print(f"  Total Cost: {np.mean(heuristic_times) + np.mean(heuristic_delays):.4f}\n")
    
    # Calculate improvement
    dqn_cost = np.mean(dqn_times) + np.mean(dqn_delays)
    heuristic_cost = np.mean(heuristic_times) + np.mean(heuristic_delays)
    improvement = ((heuristic_cost - dqn_cost) / heuristic_cost) * 100
    
    if improvement > 0:
        print(f"✓ DQN is {improvement:.2f}% better than heuristic")
    else:
        print(f"✗ Heuristic is {abs(improvement):.2f}% better than DQN")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Train DQN agent
    agent, rewards, times, delays = train_dqn(
        num_episodes=500,
        num_nodes=10,
        print_every=100
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train and evaluate DQN agent for dynamic routing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='standard',
        choices=['standard', 'fast', 'adaptive', 'rl'],
        help='''Evaluation mode:
            standard - Traditional DQN vs Heuristic comparison (default)
            fast - Heuristic-only (CPU deployment mode)
            adaptive - Hybrid adaptive controller
            rl - Pure DQN evaluation'''
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and load existing model'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='dqn_agent.pth',
        help='Path to save/load model'
    )
    
    args = parser.parse_args()
    
    # Training phase (unless skipped)
    if not args.skip_training:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Training DQN Agent")
        print(f"{'='*60}\n")
        
        agent, rewards, times, delays = train_dqn(
            num_episodes=500,
            num_nodes=10,
            print_every=100
        )
        
        # Save trained agent
        agent.save(args.model_path)
        print(f"\n✓ Agent saved to '{args.model_path}'")
    else:
        # Load existing agent
        print(f"\nLoading agent from '{args.model_path}'...")
        env = DynamicDeliveryEnv(num_nodes=10)
        action_size = 11
        state_size = 11 * 6 + 1
        agent = DQNAgent(state_size, action_size)
        agent.load(args.model_path)
        print("✓ Agent loaded successfully")
    
    # Evaluation phase
    print(f"\n{'='*60}")
    print(f"PHASE 2: Evaluation ({args.mode.upper()} mode)")
    print(f"{'='*60}\n")
    
    if args.mode == 'standard':
        # Traditional comparison
        print("Evaluating trained agent...")
        eval_stats = evaluate_agent(agent, num_episodes=20, num_nodes=10, debug=False)
        print(f"\nEvaluation Results:")
        print(f"  Avg Reward: {eval_stats['avg_reward']:.4f}")
        print(f"  Avg Time: {eval_stats['avg_time']:.4f}")
        print(f"  Avg Delay: {eval_stats['avg_delay']:.4f}\n")
        
        # Compare with heuristic
        compare_with_heuristic(agent, num_nodes=10, num_trials=20)
    
    else:
        # Hybrid controller evaluation
        controller = HybridController(mode=args.mode if args.mode != 'standard' else 'rl')
        env = DynamicDeliveryEnv(num_nodes=10)
        
        print(f"🚀 Hybrid Lightweight Decision Intelligence")
        print(f"   Mode: {controller.mode.upper()}")
        print(f"   CPU-Deployable: {'Yes' if args.mode == 'fast' else 'Adaptive'}\n")
        
        stats = evaluate_hybrid(
            controller, 
            agent if args.mode != 'fast' else None,
            env,
            num_episodes=20,
            verbose=True
        )
        
        # Compare with traditional approaches
        print(f"\n{'='*60}")
        print("Comparison with Baseline Policies")
        print(f"{'='*60}\n")
        
        # Heuristic-only baseline
        heur_controller = HybridController(mode='fast')
        heur_stats = evaluate_hybrid(heur_controller, None, env, num_episodes=20)
        
        print(f"Heuristic-Only:")
        print(f"  Avg Reward: {heur_stats['avg_reward']:.2f}")
        print(f"  Avg Time: {heur_stats['avg_time']:.2f}\n")
        
        print(f"{controller.mode.capitalize()} Mode:")
        print(f"  Avg Reward: {stats['avg_reward']:.2f}")
        print(f"  Avg Time: {stats['avg_time']:.2f}\n")
        
        # Calculate improvement
        if stats['avg_time'] < heur_stats['avg_time']:
            improvement = (heur_stats['avg_time'] - stats['avg_time']) / heur_stats['avg_time'] * 100
            print(f"✓ {controller.mode.capitalize()} mode is {improvement:.2f}% faster than heuristic")
        
        print(f"\n{'='*60}\n")
