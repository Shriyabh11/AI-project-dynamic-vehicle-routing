# Dynamic Vehicle Routing with Deep Q-Learning

**Real-time traffic-aware delivery routing using Deep Reinforcement Learning**

## 🚀 Features

- **Deep Q-Network (DQN)** with Dueling Architecture
- **Hybrid Controller** - Adaptive policy switching based on traffic variance
- **Real Road Geometry** via OSRM (OpenStreetMap)
- **Live Traffic Visualization** with color-coded segments
- **Interactive Dashboard** built with Streamlit
- **2-Opt Route Optimization** for local improvements

## 🧠 RL Approach

### Algorithm: Deep Q-Learning (DQN)

**State Representation:**
- Current delivery position (one-hot encoded)
- Visited/unvisited nodes (binary mask)
- Route progress (normalized: remaining stops / total stops)
- Real-time traffic conditions

**Action Space:**
- Select next delivery location from unvisited nodes

**Reward Function:**
```
R = -(travel_time + delay_penalties + traffic_cost)
```

**Q-Learning Update:**
```
Q(s,a) ← Q(s,a) + α[r + γ·maxₐ' Q(s',a') - Q(s,a)]
```

### Network Architecture

**Dueling DQN:**
```
Q(s,a) = V(s) + [A(s,a) - mean(A(s,·))]
```
- **Value Stream V(s):** Estimates state quality
- **Advantage Stream A(s,a):** Estimates action advantage

**Training Features:**
- Prioritized Experience Replay (PER)
- Target Network for stable learning
- ε-greedy exploration (ε: 1.0 → 0.08)

### Hybrid Controller

Intelligently switches between policies based on traffic conditions:

| Traffic Variance | Policy | Rationale |
|------------------|--------|-----------|
| < 0.02 | Heuristic | Stable traffic → fast greedy sufficient |
| 0.02 - 0.08 | Adaptive | Moderate variance → blend approaches |
| > 0.08 | DQN | High variance → learned policy needed |

## 🎯 Quick Start

### Installation

```bash
pip install streamlit folium numpy torch requests
```

### Run Dashboard

```bash
streamlit run streamlit_app.py
```

### Train DQN Agent

```bash
python train.py
```

## 📁 Project Structure

```
dynamic_routing/
├── streamlit_app.py       # Interactive dashboard
├── agent.py               # DQN implementation
├── environment.py         # MDP formulation
├── hybrid_controller.py   # Adaptive policy switching
├── traffic_api.py         # OSRM + traffic integration
├── train.py               # Training script
└── dqn_agent.pth         # Trained model (500 episodes)
```

## 🎓 Academic Context

**Course Project:** Autonomous Navigation using Reinforcement Learning

**Key Concepts Demonstrated:**
- Q-Learning with function approximation
- Markov Decision Processes (MDPs)
- Policy iteration vs value iteration
- Exploration-exploitation tradeoff
- Deep RL state representation

## 📊 Demo Flow

1. **Add Locations** - Click map or search addresses
2. **Generate Route** - System selects appropriate policy
3. **View Metrics** - ETA, distance, traffic breakdown
4. **Analyze Policy** - See which approach was used and why

## 🛠️ Technologies

- **RL Framework:** PyTorch
- **Dashboard:** Streamlit
- **Maps:** Folium + OSRM
- **Traffic:** Real-time API + simulated fallback

## 📈 Results

**Training (500 episodes):**
- Average Reward: -369 → -354 (improved)
- Travel Time: 64.7 → 62.5 minutes (optimized)
- Epsilon: 0.60 → 0.08 (exploitation)

**Demo Performance:**
- Routes follow real curved streets
- Traffic-aware path selection
- Dynamic policy adaptation
- Sub-second routing for 5-10 stops

## 🎤 For Viva/Demo

**Explain Q-Learning:**
> "Our DQN learns Q(s,a) - the expected cumulative reward for taking action 'a' in state 's'. The Dueling architecture separates state value from action advantages, leading to better generalization across similar states."

**Explain Hybrid Approach:**
> "Rather than always using DQN, our hybrid controller adapts to traffic conditions. When traffic is stable, we use fast heuristics. When variance is high, DQN's learned policy handles the complexity."

**Explain Real Roads:**
> "Routes follow actual street networks via OSRM, not straight lines. This provides realistic travel time estimates and visually compelling demonstrations."

## 📝 License

Academic project for educational purposes.

## 👤 Author

Shriya - Autonomous Navigation RL Demo
