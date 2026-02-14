"""
Visualization Module for Dynamic Routing Demo.

Features:
- Traffic Heatmap: Color-coded routes based on congestion
- Policy Confidence Meter: Heuristic vs DQN confidence comparison
- Stability Score: Real-time tracking of system stability
- Route Comparison: Before/After shock injection
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import io
import pydeck as pdk

class RouteVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.traffic_cmap = plt.cm.RdYlGn_r  # Red=High Traffic, Green=Low
    
    def create_pydeck_layers(self, locations, route, traffic_matrix):
        """
        Create Pydeck layers for 3D map visualization.
        
        Args:
            locations (list): List of (lat, lon) tuples
            route (list): List of node indices
            traffic_matrix (np.array): (N, N) traffic multipliers
            
        Returns:
            list: List of pydeck.Layer objects
        """
        if not locations:
            return []
            
        # 1. Scatter Layer for Nodes
        # Separate depot (index 0) from others for styling
        data_nodes = []
        for i, loc in enumerate(locations):
            color = [0, 0, 255, 160] if i != 0 else [255, 0, 0, 200] # Blue for delivery, Red for depot
            radius = 100 if i != 0 else 200
            data_nodes.append({
                "coordinates": [loc[1], loc[0]], # Pydeck uses [lon, lat]
                "color": color,
                "radius": radius,
                "id": str(i)
            })
            
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data_nodes,
            get_position="coordinates",
            get_color="color",
            get_radius="radius",
            pickable=True,
        )
        
        # 2. Path Layer for Route
        if not route or len(route) < 2:
            return [scatter_layer]
            
        data_paths = []
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            start_loc = locations[u]
            end_loc = locations[v]
            
            traffic = traffic_matrix[u, v]
            normalized = min(1.0, max(0.0, (traffic - 0.8) / 1.7))
            
            # Color Mapping (Green -> Yellow -> Red)
            # Simple interpolation
            r = int(255 * normalized)
            g = int(255 * (1 - normalized))
            color = [r, g, 0]
            
            data_paths.append({
                "path": [[start_loc[1], start_loc[0]], [end_loc[1], end_loc[0]]],
                "color": color,
                "width": 5 + normalized * 10
            })
            
        path_layer = pdk.Layer(
            "PathLayer",
            data_paths,
            get_path="path",
            get_color="color",
            get_width="width",
            width_min_pixels=3,
            pickable=True
        )
        
        return [scatter_layer, path_layer]
    
    def plot_route(self, locations, route, traffic_matrix, current_node=None):
        """
        Plot the current route with traffic-aware coloring.
        
        Args:
            locations (np.array): (N, 2) array of coordinates
            route (list): List of node indices
            traffic_matrix (np.array): (N, N) traffic multipliers
            current_node (int): Current agent position
        """
        self.ax.clear()
        
        # Plot nodes
        self.ax.scatter(locations[:, 0], locations[:, 1], c='blue', s=100, label='Delivery Node')
        self.ax.scatter(locations[0, 0], locations[0, 1], c='red', s=200, marker='*', label='Depot')
        
        if current_node is not None:
            self.ax.scatter(locations[current_node, 0], locations[current_node, 1], 
                           c='gold', s=300, marker='o', edgecolors='black', label='Agent')
        
        # Plot edges
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            traffic = traffic_matrix[u, v]
            
            # Color based on traffic (0.8=Green, 1.5+=Red)
            normalized_traffic = min(1.0, max(0.0, (traffic - 0.8) / 1.7))
            color = self.traffic_cmap(normalized_traffic)
            
            width = 2 + normalized_traffic * 3  # Thicker lines for heavy traffic
            
            self.ax.plot([locations[u, 0], locations[v, 0]], 
                        [locations[u, 1], locations[v, 1]], 
                        c=color, linewidth=width, alpha=0.7)
            
            # Add traffic label if heavy
            if traffic > 1.3:
                mid_x = (locations[u, 0] + locations[v, 0]) / 2
                mid_y = (locations[u, 1] + locations[v, 1]) / 2
                self.ax.text(mid_x, mid_y, f"{traffic:.1f}x", fontsize=8, color='red', fontweight='bold')
                
        self.ax.set_title("Real-Time Route Optimization with Traffic Awareness")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        return self.fig

    def plot_confidence_meter(self, dqn_q_values, heuristic_score):
        """
        Visualize policy confidence.
        
        Args:
            dqn_q_values (np.array): Q-values from DQN
            heuristic_score (float): Score from heuristic
        """
        # Create a separate figure for the meter
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.set_axis_off()
        
        # Normalize scores to 0-1 range (simplified for visual)
        dqn_conf = np.max(dqn_q_values) - np.mean(dqn_q_values)
        heur_conf = 1.0 / (heuristic_score + 1e-5)
        
        total = dqn_conf + heur_conf
        p_dqn = dqn_conf / total
        p_heur = heur_conf / total
        
        # Draw bar
        ax.barh(0, p_dqn, color='purple', label='DQN Confidence')
        ax.barh(0, p_heur, left=p_dqn, color='orange', label='Heuristic Confidence')
        
        ax.text(p_dqn/2, 0, f"DQN\n{p_dqn:.0%}", ha='center', va='center', color='white', fontweight='bold')
        ax.text(p_dqn + p_heur/2, 0, f"Heuristic\n{p_heur:.0%}", ha='center', va='center', color='black', fontweight='bold')
        
        ax.set_title("Policy Confidence Meter")
        
        return fig

    def plot_stability_score(self, stability_history):
        """
        Plot system stability over time.
        
        Args:
            stability_history (list): List of stability scores (0-100)
        """
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(stability_history, c='teal')
        ax.set_ylim(0, 100)
        ax.set_ylabel("Stability Score")
        ax.set_xlabel("Time Step")
        ax.fill_between(range(len(stability_history)), stability_history, alpha=0.2, color='teal')
        return fig
