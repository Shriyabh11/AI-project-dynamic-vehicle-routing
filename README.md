# Dynamic Vehicle Routing with Deep Q-Learning

Real-time traffic-aware delivery routing using Deep Reinforcement Learning

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements_viz.txt

# Run dashboard
streamlit run app/streamlit_app.py

# Train DQN agent
python scripts/train.py
```

## 📁 Project Structure

```
AI-project-dynamic-vehicle-routing/
├── app/
│   └── streamlit_app.py          # Interactive dashboard
├── src/
│   ├── agent.py                  # DQN implementation
│   ├── environment.py            # MDP formulation
│   ├── heuristic.py              # Greedy & 2-opt algorithms
│   ├── hybrid_controller.py      # Adaptive policy switching
│   └── traffic_api.py            # OSRM + traffic integration
├── scripts/
│   ├── train.py                  # Training script
│   ├── trainer.py                # Training utilities
│   └── verify_hybrid.py          # Testing hybrid controller
├── utils/
│   ├── utils.py                  # Helper functions
│   ├── visualizer.py             # Plotting utilities
│   └── vrp.py                    # VRP solver
├── models/
│   └── dqn_agent.pth            # Trained DQN weights
├── tests/
│   ├── test_all.py              # Unit tests
│   ├── test_osrm.py             # OSRM integration tests  
│   └── safety.py                # Safety validation
├── README.md
├── requirements_viz.txt
└── .gitignore
```

## 🧠 RL Approach

**Algorithm:** Deep Q-Network (DQN) with Dueling Architecture

**State:** `[current_pos, visited_mask, progress, traffic]`  
**Action:** Select next delivery location  
**Reward:** `-travel_time - delay_penalties`

**Hybrid Controller:** Adaptive policy based on traffic variance
- Variance < 0.02 → Heuristic (fast)
- Variance 0.02-0.08 → Adaptive blend
- Variance > 0.08 → DQN (learned policy)

## 🎯 Features

✅ Deep Q-Learning with Dueling architecture  
✅ Real curved roads via OSRM  
✅ Live traffic visualization  
✅ Hybrid controller (adaptive switching)  
✅ 2-opt route optimization  
✅ Interactive Streamlit dashboard

## 📊 Demo

1. Add locations (click map or search)
2. Click "Route" to generate optimal path
3. View metrics, policy, and traffic analysis

## 🛠️ Tech Stack

- **RL:** PyTorch
- **Dashboard:** Streamlit + Folium
- **Routing:** OSRM (OpenStreetMap)
- **Optimization:** 2-opt, greedy heuristics

## 👤 Author

Shriya - Autonomous Navigation RL Project
