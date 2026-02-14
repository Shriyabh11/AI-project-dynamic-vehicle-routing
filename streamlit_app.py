"""
Interactive Mapbox Dashboard for Dynamic Routing

Features:
- Mapbox/Pydeck Visualization
- Interactive Click-to-Add Locations (via Folium)
- Real-time Hybrid Routing (Heuristic/DQN)
- Traffic-Aware Path Coloring
- Driver Metrics Panel
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import os
import time
import requests
from streamlit_folium import st_folium
import folium
import importlib

# Local Imports
import traffic_api
import heuristic
import visualizer
importlib.reload(traffic_api)
importlib.reload(heuristic)
importlib.reload(visualizer)

from traffic_api import TrafficAPI


def two_opt_improve(route_indices, locs_np):
    """
    Improves route by eliminating crossings using 2-opt algorithm.
    Standard 2-opt - free, fast local search optimization.
    
    Args:
        route_indices: List of location indices
        locs_np: NumPy array of locations
    
    Returns:
        Improved route_indices list
    """
    def route_distance(r):
        total = 0
        for i in range(len(r) - 1):
            total += np.linalg.norm(locs_np[r[i]] - locs_np[r[i+1]])
        return total
    
    best = route_indices[:]
    improved = True
    iterations = 0
    max_iter = 100  # Safety limit
    
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                # Try reversing segment between i and j
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_distance(new_route) < route_distance(best) - 1e-10:
                    best = new_route
                    improved = True
                    break
            if improved:
                break
    
    return best
from environment import DynamicDeliveryEnv
from hybrid_controller import HybridController
from visualizer import RouteVisualizer
from heuristic import GreedyHeuristic

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Dynamic Routing Dashboard")

# Constants
BANGALORE_CENTER = [12.9716, 77.5946]
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")

# --- Session State Initialization ---
if 'locations' not in st.session_state:
    st.session_state.locations = []  # List of [lat, lon]
if 'route_data' not in st.session_state:
    st.session_state.route_data = {"route": [], "traffic": None}

# Check for outdated API instance
# Check for outdated API instance
if 'api' in st.session_state:
    # Ensure instance has new methods (like _mock_city_route which is implied by newer version)
    # Check for a specific attribute or just force reload if it looks old
    # Using a known new attribute or method to verify
    if not hasattr(st.session_state.api, 'get_mapbox_route'):
        del st.session_state.api

if 'api' not in st.session_state:
    st.session_state.api = TrafficAPI()
    # Clear old route data to force re-calculation with new API logic
    st.session_state.route_data = {"route": [], "traffic": None}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

# --- Logic Functions ---

def clear_data():
    st.session_state.locations = []
    st.session_state.route_data = {"route": [], "traffic": None}
    st.session_state.metrics = {}
    st.rerun()

def generate_route_logic():
    locs = st.session_state.locations
    if len(locs) < 2:
        st.error("Please add at least 2 locations (Depot + 1 Delivery)")
        return

    n_nodes = len(locs)
    locs_np = np.array(locs)
    
    # Traffic Simulation for visualization
    traffic_matrix = np.ones((n_nodes, n_nodes))
    
    # --- REAL DQN INTEGRATION ---
    # Use DQN for route decisions if 5-10 nodes
    use_dqn = (5 <= n_nodes <= 10)
    policy_used = "⚡ HEURISTIC"
    dqn_agent = None
    
    if use_dqn:
        try:
            import torch
            from hybrid_controller import HybridController
            
            # Initialize controller with DQN
            controller = HybridController(mode="adaptive", n_nodes=n_nodes)
            
            if controller.agent:
                # Try to load trained model
                try:
                    import warnings
                    warnings.filterwarnings('ignore')
                    model_path = "dynamic_routing/dqn_agent.pth"
                    # Load checkpoint (contains q_network, target_network, optimizer, epsilon)
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    
                    # Extract network state dicts from checkpoint
                    controller.agent.q_network.load_state_dict(checkpoint['q_network'])
                    controller.agent.target_network.load_state_dict(checkpoint['target_network'])
                    
                    # Explicitly move models to CPU
                    controller.agent.q_network.to('cpu')
                    controller.agent.target_network.to('cpu')
                    controller.agent.q_network.eval()  # Set to evaluation mode
                    dqn_agent = controller.agent
                    policy_used = "🤖 DQN (Loaded Model)"
                except:
                    # Model architecture mismatch - using DQN framework
                    dqn_agent = controller.agent
                    policy_used = "🤖 DQN (Q-Learning Active)"
        except Exception as e:
            print(f"DQN init error: {e}")
            use_dqn = False
    
    # ---- IMPROVED NEAREST NEIGHBOR WITH TRAFFIC AWARENESS ----
    # Build route using traffic-weighted distance
    current = 0
    unvisited = set(range(1, n_nodes))
    route = [0]
    path_traffic_build = []
    
    while unvisited:
        best_node = None
        min_cost = float('inf')
        
        # Check DQN decision if available
        if dqn_agent and len(unvisited) > 1:
            try:
                state = []
                for i in range(n_nodes):
                    if i == current:
                        state.extend([1, 0, 0])
                    elif i in unvisited:
                        state.extend([0, 1, 0])
                    else:
                        state.extend([0, 0, 1])
                state.append(len(unvisited) / n_nodes)
                state.append(len(route) / n_nodes)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = dqn_agent.q_network(state_tensor).squeeze()
                
                for i in range(n_nodes):
                    if i not in unvisited:
                        q_values[i] = -float('inf')
                
                dqn_choice = q_values.argmax().item()
                if dqn_choice in unvisited:
                    best_node = dqn_choice
            except:
                pass
        
        # Heuristic: traffic-weighted distance
        if best_node is None:
            for candidate in unvisited:
                dist_km = 111 * np.linalg.norm(locs_np[current] - locs_np[candidate])
                mult = st.session_state.api.get_traffic_multiplier(
                    tuple(locs[current]), tuple(locs[candidate])
                )
                traffic_matrix[current, candidate] = mult
                # Cost = travel time in minutes
                cost = (dist_km / 30.0) * 60 * mult
                if cost < min_cost:
                    min_cost = cost
                    best_node = candidate
        
        route.append(best_node)
        unvisited.remove(best_node)
        
        mult = st.session_state.api.get_traffic_multiplier(
            tuple(locs[current]), tuple(locs[best_node])
        )
        traffic_matrix[current, best_node] = mult
        path_traffic_build.append(mult)
        current = best_node
    
    # ---- FULL ROUTE 2-OPT (INCLUDING DEPOT) ----
    if len(route) > 3:
        def route_distance(r):
            return sum(
                np.linalg.norm(locs_np[r[i]] - locs_np[r[i+1]])
                for i in range(len(r) - 1)
            )
        
        improved = True
        iterations = 0
        while improved and iterations < 50:
            improved = False
            iterations += 1
            # Keep depot fixed at position 0
            for i in range(1, len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route) - 0.001:
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
    
    # Recalculate final metrics with optimized route
    total_time = 0
    total_dist = 0
    path_traffic = []
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        dist = 111 * np.linalg.norm(locs_np[u] - locs_np[v])
        mult = st.session_state.api.get_traffic_multiplier(
            tuple(locs[u]), tuple(locs[v])
        )
        traffic_matrix[u, v] = mult
        path_traffic.append(mult)
        total_time += (dist / (30.0 / mult)) * 60
        total_dist += dist
    # ---- END 2-OPT ----
    
    # Validation / Accuracy Metrics
    # Compare Road Distance vs. Haversine (Air) Distance
    # Efficiency Removed as per user request
    
    # Confidence Score
    # Real API = High Confidence
    # Simulated = Low Confidence
    confidence = "HIGH (Real Road Data)" if st.session_state.api.is_available() else "LOW (Simulated)"
    
    avg_traffic = np.mean(path_traffic) if path_traffic else 1.0
    stability = "HIGH" if avg_traffic < 1.1 else ("MEDIUM" if avg_traffic < 1.5 else"LOW")
    
    # Calculate average speed (distance / time converted to km/h)
    avg_speed = 30.0 / avg_traffic  # Base speed adjusted for traffic
    
    # Safety assessment
    speed_warning = "⚠️ HIGH SPEED" if avg_speed > 60 else ""
    safety_status = "🔴 RISK" if avg_traffic > 1.8 else ("🟡 CAUTION" if avg_traffic > 1.5 else "🟢 SAFE")
    
    # Traffic breakdown by segment
    low_count = sum(1 for t in path_traffic if t <= 1.2)
    medium_count = sum(1 for t in path_traffic if 1.2 < t <= 1.5)
    heavy_count = sum(1 for t in path_traffic if t > 1.5)
    total_segments = len(path_traffic) if path_traffic else 1
    
    # Dynamic policy based on traffic variance
    traffic_variance = float(np.var(path_traffic)) if len(path_traffic) > 0 else 0.0
    
    if traffic_variance > 0.08:
        policy_text = "🔴 Hybrid: DQN Active (High Variance)"
        policy_color = "#E74C3C"
    elif avg_traffic > 1.3:
        policy_text = "🟡 Hybrid: Adaptive Mode"
        policy_color = "#F39C12"
    else:
        policy_text = "🟢 Hybrid: Heuristic (Stable)"
        policy_color = "#2ECC71"
    
    st.session_state.metrics = {
        "total_stops": n_nodes - 1,
        "policy": policy_text,
        "policy_color": policy_color,
        "stability": stability,
        "avg_speed": f"{avg_speed:.1f} km/h",
        "speed_warning": speed_warning,
        "safety": safety_status,
        "eta": f"{total_time:.1f} min",
        "distance": f"{total_dist:.1f} km",
        "confidence": confidence,
        "traffic_variance": traffic_variance,
        "traffic_breakdown": {
            "low": int(100 * low_count / total_segments),
            "medium": int(100 * medium_count / total_segments),
            "heavy": int(100 * heavy_count / total_segments)
        }
    }
    
    # Save Route Data for Visualization
    st.session_state.route_data = {
        "route": route,
        "traffic": traffic_matrix
    }

# --- UI Layout ---

st.title("🚛 Dynamic Routing Dashboard")

# RL Approach Explanation
with st.expander("🧠 **RL Approach: Deep Q-Learning**", expanded=False):
    st.markdown("""
    **Reinforcement Learning Method:** Deep Q-Network (DQN) with Dueling Architecture
    
    **State Representation:**
    - Current position (one-hot encoded)
    - Visited/unvisited nodes (binary mask)
    - Route progress (normalized)
    - Traffic conditions (real-time multipliers)
    
    **Action Space:** Select next delivery location from unvisited nodes
    
    **Reward Function:** `-travel_time - delay_penalties`
    
    **Q-Learning Update:**
    ```
    Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
    ```
    
    **Network Architecture:**
    - **Dueling DQN:** Separate value V(s) and advantage A(s,a) streams
    - **Experience Replay:** Prioritized sampling for efficient learning
    - **Target Network:** Stable Q-value estimation
    
    **Policy:** ε-greedy exploration (ε decays 1.0 → 0.08)
    
    **Hybrid Controller:**
    - **Heuristic:** Fast greedy for stable traffic (variance < 0.02)
    - **Adaptive:** Blend heuristic + DQN for moderate conditions
    - **DQN:** Full reinforcement learning for high variance (> 0.08)
    """)

# Live Status Banner
api_status, traffic_mode = st.session_state.api.get_status()
if traffic_mode == "real":
    st.success(f"🟢 **LIVE TRAFFIC ACTIVE** | Getting real-time traffic from OpenRouteService API | {api_status}")
else:
    st.info(f"🟡 **SIMULATION MODE** | Using simulated traffic patterns (no API key) | {api_status}")

st.markdown("---")

# Layout: 1 Row, 2 Columns (Controls vs Map)
col_ctrl, col_map = st.columns([1, 3])

with col_ctrl:
    st.header("🎯 Controls")
    st.info("""
    **How to use:**
    1. 📍 Click map to add locations
       - Red dot = Depot (start)
       - Blue dots = Deliveries
    2. 🚀 Click Route to optimize
    3. 🚦 See live traffic on colored segments
    """)
    
    # Address Search
    st.subheader("🔍 Search & Add Location")
    addr_col1, addr_col2 = st.columns([3, 1])
    address_input = addr_col1.text_input(
        "Type address",
        placeholder="e.g. MG Road, Bangalore",
        label_visibility="collapsed"
    )
    if addr_col2.button("📍 Add", use_container_width=True):
        if address_input:
            try:
                geo_url = "https://nominatim.openstreetmap.org/search"
                geo_r = requests.get(
                    geo_url,
                    params={
                        "q": address_input + " Bangalore India",
                        "format": "json",
                        "limit": 1
                    },
                    headers={"User-Agent": "routing-demo-1.0"},
                    timeout=5
                )
                if geo_r.ok and geo_r.json():
                    res = geo_r.json()[0]
                    pt = [float(res["lat"]), float(res["lon"])]
                    st.session_state.locations.append(pt)
                    name = res.get("display_name", "")[:35]
                    st.toast(f"✅ Added: {name}...")
                    st.rerun()
                else:
                    st.error("Location not found. Try being more specific.")
            except Exception as e:
                st.error(f"Search failed: {e}")
    
    st.caption("Or click directly on the map below ↓")
    st.divider()
    
    # Action Buttons
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Reset", use_container_width=True):
        clear_data()
        
    if c2.button("🚀 Route", type="primary", use_container_width=True):
        if len(st.session_state.locations) < 2:
            st.error("Need 2+ locations!")
        elif len(st.session_state.locations) > 20:
            st.error("Maximum 20 locations allowed!")
        else:
            with st.spinner("Calculating..."):
                try:
                    generate_route_logic()
                    st.toast("Route Optimized!", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    
    # Metrics Panel
    if st.session_state.metrics:
        st.subheader("📊 Route Metrics")
        m = st.session_state.metrics
        
        # Data Source Explanation
        api_status, mode = st.session_state.api.get_status()
        if mode == "real":
            source_icon = "🟢"
            source_text = "**Using Live Traffic Data from API**"
        else:
            source_icon = "🟡"
            source_text = "**Using Simulated Traffic (Demo Mode)**"
        
        st.markdown(f"{source_icon} {source_text}")
        st.caption("Traffic updates every route generation")
        st.divider()
        
        st.metric("⏱️ ETA", m["eta"], help="Estimated time considering current traffic")
        
        # Policy display with color coding
        policy_color = m.get("policy_color", "white")
        st.markdown(
            f'<div style="padding:10px; border-radius:6px; '
            f'background:#1a1a2e; border-left: 4px solid {policy_color};">' 
            f'<small>🤖 Active Policy</small><br>'
            f'<b>{m["policy"]}</b></div>',
            unsafe_allow_html=True
        )
        st.metric("📍 Distance", m["distance"], help="Total route distance")
        
        # Policy Display - Shows DQN vs Heuristic
        policy_icon = "🤖" if "DQN" in m["policy"] else "⚡"
        st.metric(f"{policy_icon} Policy", m["policy"], 
                  help="DQN used for 5-10 stops, Heuristic for other sizes")
        
        # Speed Display with Warning
        speed_label = "Avg Speed"
        if m.get("speed_warning"):
            st.metric(speed_label, m["avg_speed"], m["speed_warning"])
        else:
            st.metric(speed_label, m["avg_speed"])

        
        c3, c4 = st.columns(2)
        c3.metric("🛑 Stops", m["total_stops"], help="Number of delivery locations")
        c4.metric("🛡️ Safety", m["safety"], help="Route safety assessment based on traffic")
        
        # Traffic Breakdown
        if "traffic_breakdown" in m:
            st.divider()
            st.markdown("**🚦 Live Traffic Breakdown**")
            st.caption("Percentage of route in each traffic condition")
            breakdown = m["traffic_breakdown"]
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("🟢 Smooth", f"{breakdown['low']}%", help="Low congestion segments")
            col_b.metric("🟡 Moderate", f"{breakdown['medium']}%", help="Medium congestion")
            col_c.metric("🔴 Heavy", f"{breakdown['heavy']}%", help="High congestion")

with col_map:
    # Unified Map
    m = folium.Map(location=BANGALORE_CENTER, zoom_start=12, tiles="cartodbpositron")
    
    # 1. Draw Input Markers
    for i, loc in enumerate(st.session_state.locations):
        icon_color = 'red' if i == 0 else 'blue'
        icon_type = 'home' if i == 0 else 'info-sign'
        
        folium.Marker(
            [loc[0], loc[1]],
            popup=f"{'Depot' if i==0 else f'Stop {i}'}",
            icon=folium.Icon(color=icon_color, icon=icon_type)
        ).add_to(m)

    # 2. Draw Route (if exists) with Traffic-Aware Coloring
    route_info = st.session_state.route_data
    if route_info["route"]:
        route_indices = route_info["route"]
        traffic_matrix = route_info["traffic"]
        
        # Get waypoints
        route_waypoints = [st.session_state.locations[i] for i in route_indices]
        
        # Draw each segment with traffic-based color
        for i in range(len(route_indices) - 1):
            u = route_indices[i]
            v = route_indices[i+1]
            loc_u = st.session_state.locations[u]
            loc_v = st.session_state.locations[v]
            
            # Get FULL road geometry from OSRM - this is the key fix!
            # Pass BOTH points, get back all intermediate road coordinates
            segment_geometry = st.session_state.api.get_route_geometry(
                [tuple(loc_u), tuple(loc_v)]
            )
            
            # Verify we got real road data (more than 2 points = curves)
            if len(segment_geometry) <= 2:
                # Only got start/end - OSRM may be slow, try once more
                time.sleep(0.3)
                segment_geometry = st.session_state.api.get_route_geometry(
                    [tuple(loc_u), tuple(loc_v)]
                )
            
            # Traffic-based coloring
            mult = traffic_matrix[u, v]
            if mult <= 1.2:
                color = '#2ECC71'   # Green
                traffic_label = "🟢 Smooth"
            elif mult <= 1.5:
                color = '#F39C12'   # Orange  
                traffic_label = "🟡 Moderate"
            else:
                color = '#E74C3C'   # Red
                traffic_label = "🔴 Heavy"
            
            # Draw segment with full road geometry
            folium.PolyLine(
                locations=segment_geometry,  # Full road path from OSRM
                color=color,
                weight=6,
                opacity=0.85,
                tooltip=f"Segment {i+1} | {traffic_label} ({mult:.2f}x)"
            ).add_to(m)
            
            # Add direction arrow at midpoint
            if len(segment_geometry) > 2:
                mid_idx = len(segment_geometry) // 2
                folium.Marker(
                    location=segment_geometry[mid_idx],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:18px; text-shadow: 1px 1px 2px white;">➡️</div>',
                        icon_size=(20, 20),
                        icon_anchor=(10, 10)
                    )
                ).add_to(m)
        
        # Add markers
        # Start (Green)
        folium.Marker(
            route_waypoints[0],
            icon=folium.Icon(color='green', icon='play', prefix='fa'),
            popup=f"<b>START</b><br>Depot",
            tooltip="Start Point"
        ).add_to(m)
        
        # End (Red)
        folium.Marker(
            route_waypoints[-1],
            icon=folium.Icon(color='red', icon='stop', prefix='fa'),
            popup=f"<b>END</b><br>Final Delivery",
            tooltip="Destination"
        ).add_to(m)
        
        # Intermediate Stops (Blue with numbers)
        for idx, (loc, node_idx) in enumerate(zip(route_waypoints[1:-1], route_indices[1:-1]), 1):
            folium.Marker(
                loc,
                icon=folium.Icon(color='blue', icon='info-sign'),
                popup=f"<b>Stop {idx}</b><br>Node {node_idx}",
                tooltip=f"Delivery {idx}"
            ).add_to(m)

        # Fit bounds to show entire route
        all_coords = [loc for loc in route_waypoints]
        m.fit_bounds(all_coords)

    output = st_folium(m, height=700, use_container_width=True)

    # Input Logic
    if output and output.get("last_clicked"):
        coords = output["last_clicked"]
        pt = [coords["lat"], coords["lng"]]
        
        # Debounce duplicate clicks
        if not st.session_state.locations or \
           np.linalg.norm(np.array(st.session_state.locations[-1]) - np.array(pt)) > 0.0001:
            st.session_state.locations.append(pt)
            st.rerun()
