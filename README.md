# Dynamic Vehicle Routing with Deep Q-Learning

**Academic Course Project: Reinforcement Learning for Autonomous Systems**

A demonstration of Deep Reinforcement Learning applied to real-time vehicle routing optimization, combining neural network-based decision making with classical heuristics for practical deployment.

---

## 🎯 Project Overview

This project implements a **Hybrid Deep Q-Network (DQN)** system for dynamic vehicle routing that adapts to real-time traffic conditions. The system intelligently switches between fast heuristic algorithms and learned RL policies based on environmental complexity.

### Key Innovations

1. **Dueling DQN Architecture**: Separates state value and action advantage estimation for better generalization
2. **Hybrid Controller**: Adaptive policy switching based on traffic stability (heuristic ↔ DQN)
3. **Real Road Geometry**: OSRM API integration for actual street paths and travel times
4. **OSRM-Based Traffic Data**: Real routing times with congestion detection via traffic multipliers
5. **Heuristic-Guided Action Masking**: Top-k action filtering for efficient RL exploration
6. **Safety Monitoring**: Integrated overspeed and congestion risk detection

---

## 🧠 Reinforcement Learning Approach

### Algorithm: Deep Q-Learning

**Markov Decision Process (MDP) Formulation:**

- **State Space**: `s = [position, visited_mask, progress, traffic_conditions]`
- **Action Space**: `A = {next_delivery_location}` (from unvisited nodes)
- **Reward Function**: `R(s,a) = -(travel_time + delay_penalties + traffic_cost)`

**Q-Learning Update Rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
                      a'
```

### Neural Network Architecture

**Dueling DQN Structure:**
```
Input State → Feature Layer → Split into:
                            ├─ Value Stream V(s)
                            └─ Advantage Stream A(s,a)

Q(s,a) = V(s) + [A(s,a) - mean(A(s,·))]
```

**Training Features:**
- Prioritized Experience Replay (PER)
- Target Network (updated every 10 episodes)
- ε-greedy Exploration (ε: 1.0 → 0.08 over 500 episodes)

### Hybrid Controller Logic

| Traffic Variance | Policy Used | Reasoning |
|------------------|-------------|-----------|
| < 0.02 | **Heuristic (Greedy)** | Stable conditions → fast solution sufficient |
| 0.02 - 0.08 | **Adaptive Blend** | Moderate variance → combine approaches |
| > 0.08 | **DQN (Learned)** | High variance → learned policy handles complexity |

---

## 📁 Project Structure

```
AI-project-dynamic-vehicle-routing/
├── app/
│   └── app.py                    # Streamlit dashboard (main entry point)
├── src/
│   ├── agent.py                  # DQN implementation (Dueling architecture)
│   ├── environment.py            # MDP environment definition
│   ├── heuristic.py              # Greedy nearest-neighbor & 2-opt
│   ├── hybrid_controller.py      # Adaptive policy switching logic
│   └── traffic_api.py            # OSRM integration + traffic simulation
├── scripts/
│   ├── train.py                  # Training script (500 episodes)
│   └── verify_hybrid.py          # Testing/validation script
├── models/
│   └── dqn_agent.pth            # Pre-trained DQN weights
├── tests/
│   └── safety.py                # Safety monitoring utilities
├── utils/
│   ├── utils.py                 # Helper functions
│   ├── visualizer.py            # Route plotting utilities
│   └── vrp.py                   # VRP solver algorithms
├── README.md
├── requirements_viz.txt
└── .gitignore
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

```bash
# Clone repository
git clone https://github.com/Shriyabh11/AI-project-dynamic-vehicle-routing.git
cd AI-project-dynamic-vehicle-routing

# Install dependencies
pip install -r requirements_viz.txt

# Run dashboard
streamlit run app/app.py
```

### Dependencies
- `streamlit` - Interactive web dashboard
- `folium` - Map visualization
- `torch` - Deep learning framework (PyTorch)
- `numpy` - Numerical computations
- `requests` - HTTP requests for OSRM API
- `streamlit-folium` - Folium integration for Streamlit

---

## 🎮 Demo Instructions

### Running the Dashboard

1. **Start the application:**
   ```bash
   streamlit run app/app.py
   ```

2. **Add delivery locations:**
   - Click directly on the map, OR
   - Use the search bar (e.g., "Koramangala, Bangalore")

3. **Generate optimal route:**
   - Click "🚀 Route" button
   - System selects appropriate policy based on problem size and traffic

4. **View results:**
   - **Route Metrics**: ETA, distance, policy used
   - **Visual Route**: Color-coded traffic segments
     - 🟢 Green: Smooth (≤ 1.2x normal time)
     - 🟡 Orange: Moderate (1.2-1.5x)
     - 🔴 Red: Heavy (> 1.5x)
   - **Stability Trace**: Shows decision-making process

### Training a New Model

```bash
# Train DQN agent (500 episodes, ~5-10 minutes)
python scripts/train.py

# Trained model saved to: models/dqn_agent.pth
```

---

## 📊 Experimental Results

**Training Performance (500 episodes, 10-node problem):**

| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| Average Reward | -369.97 | -354.02 | +4.3% |
| Travel Time (min) | 64.69 | 62.49 | -3.4% |
| Epsilon (exploration) | 1.0 | 0.08 | Converged |

**Policy Comparison (20 test episodes):**
- **Pure DQN**: Slower but handles complex scenarios
- **Pure Heuristic**: Fastest but misses optimization opportunities  
- **Hybrid Controller**: Best balance (93% of DQN performance, 10x faster)

---

## 🎓 Academic Context

### Course Alignment
- **Reinforcement Learning**: Q-learning, DQN, experience replay
- **Optimization**: Vehicle Routing Problem (VRP), 2-opt local search
- **Systems Integration**: Real-time APIs, web-based visualization

### Key Concepts Demonstrated

1. **Markov Decision Processes**: State-action-reward formulation
2. **Value-Based RL**: Q-function approximation with neural networks
3. **Exploration-Exploitation**: ε-greedy strategy with decay
4. **Transfer Learning**: Pre-trained model deployment
5. **Hybrid Architectures**: Combining classical & modern approaches

### Potential Extensions

- **Multi-vehicle routing**: Extend to fleet optimization
- **Enhanced traffic APIs**: Integrate Google Maps/Mapbox for richer traffic data
- **Advanced RL algorithms**: A3C, PPO, transformer-based policies
- **Constraint handling**: Time windows, vehicle capacity, priorities

---

## 🛠️ Technical Details

### OSRM Integration
Uses OpenStreetMap Routing Machine for realistic route geometry:
- **Endpoint**: `http://router.project-osrm.org/route/v1/driving/`
- Returns actual curved street paths (not straight lines)
- Provides distance and duration estimates

### Traffic Integration
**Primary Source:** OSRM (OpenStreetMap Routing Machine) API
- Real routing times based on current road conditions
- Actual curved street paths (not straight lines)
- Travel time and distance estimates

**Traffic Multiplier Calculation:**
- Compares OSRM real time vs. expected time at 40 km/h free flow
- Multiplier = real_time / expected_time
- Range: 1.0x (smooth) to 3.0x (heavy congestion)
- Used by hybrid controller for adaptive policy selection

**Fallback:** Simulated traffic with multipliers 0.8x-1.5x when API unavailable

### 2-Opt Optimization
Local search refinement applied after initial route construction:
- Swaps route segments to reduce total distance
- Runs for max 50 iterations or until no improvement
- Typically improves route by 5-15%

---

## 📝 License

Academic project for educational purposes. Code provided as-is for learning and demonstration.

---

## 🙏 Acknowledgments

- **OSRM**: OpenStreetMap routing engine
- **Streamlit**: Interactive dashboard framework
- **PyTorch**: Deep learning library
- Course instructors and teaching assistants for guidance

---

## 📧 Contact

For questions about this implementation or academic inquiries, please open an issue on the GitHub repository.
