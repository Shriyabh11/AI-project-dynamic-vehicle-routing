"""
Final Verification Script.
Tests:
1. Environment Bug Fix
2. Advanced Agent Initialization & Forward Pass
3. Traffic API Shock Injection
"""
import sys
import os
import torch
import numpy as np

# Add parent dir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dynamic_routing.environment import DynamicDeliveryEnv
from dynamic_routing.advanced_agent import AdvancedDQNAgent
from dynamic_routing.traffic_api import TrafficAPI

def test_all():
    print("running final verification...")
    
    # 1. Environment
    try:
        env = DynamicDeliveryEnv(num_nodes=5)
        env.reset()
        print("✓ Environment Reset Fixed")
    except Exception as e:
        print(f"❌ Environment Failed: {e}")
        return

    # 2. Advanced Agent
    try:
        agent = AdvancedDQNAgent(state_size=10, action_size=5)
        state = torch.randn(1, 10).to(agent.device)
        q_vals = agent.q_network(state)
        assert q_vals.shape == (1, 5)
        print("✓ Advanced Agent Dueling Network Works")
    except Exception as e:
        print(f"❌ Agent Failed: {e}")
        return

    # 3. Traffic API
    try:
        api = TrafficAPI()
        api.inject_traffic_shock("high")
        assert api.shock_active == True
        print("✓ Traffic Shock Injection Works")
    except Exception as e:
        print(f"❌ Traffic API Failed: {e}")
        return

    print("\n✅ ALL SYSTEMS GO")

if __name__ == "__main__":
    test_all()
