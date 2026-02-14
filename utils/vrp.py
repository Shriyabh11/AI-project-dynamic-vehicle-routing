"""Defines the main task for the VRP.

The VRP is defined by the following traits:
    1. Each city has a demand in [1, 9], which must be serviced by the vehicle
    2. Each vehicle has a capacity (depends on problem), the must visit all cities
    3. When the vehicle load is 0, it __must__ return to the depot to refill
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class VehicleRoutingDataset(Dataset):
    def __init__(self, num_samples, input_size, max_load=20, max_demand=9,
                 seed=None):
        super(VehicleRoutingDataset, self).__init__()

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.max_load = max_load
        self.max_demand = max_demand

        # Depot location will be the first node in each
        locations = torch.rand((num_samples, 2, input_size + 1))
        self.static = locations

        # Traffic multipliers for each edge (NxN matrix per sample)
        # Values in range [0.8, 1.5] representing traffic conditions
        # 0.8 = light traffic (faster), 1.5 = heavy traffic (slower)
        traffic_shape = (num_samples, input_size + 1, input_size + 1)
        self.traffic_multiplier = torch.rand(traffic_shape) * 0.7 + 0.8
        
        # Deadlines for each node (scaled by expected max tour length)
        # Depot has no deadline (set to large value)
        # Estimated max tour length ≈ sqrt(2) * num_nodes (worst case diagonal traversal)
        max_tour_time = np.sqrt(2) * input_size * 1.5  # With traffic buffer
        deadline_shape = (num_samples, input_size + 1)
        self.deadlines = torch.rand(deadline_shape) * max_tour_time * 0.8 + max_tour_time * 0.2
        self.deadlines[:, 0] = max_tour_time * 10  # Depot has very large deadline

        # All states will broadcast the drivers current load
        # Note that we only use a load between [0, 1] to prevent large
        # numbers entering the neural network
        dynamic_shape = (num_samples, 1, input_size + 1)
        loads = torch.full(dynamic_shape, 1.)

        # All states will have their own intrinsic demand in [1, max_demand), 
        # then scaled by the maximum load. E.g. if load=10 and max_demand=30, 
        # demands will be scaled to the range (0, 3)
        demands = torch.randint(1, max_demand + 1, dynamic_shape)
        demands = demands / float(max_load)

        demands[:, 0, 0] = 0  # depot starts with a demand of 0
        
        # Time elapsed (starts at 0 for all samples)
        time_elapsed = torch.zeros(dynamic_shape)
        
        # Urgency feature: normalized (deadline - time_elapsed) / max_deadline
        # Higher urgency = closer to deadline
        urgency = self.deadlines.unsqueeze(1) / max_tour_time  # Normalize deadlines
        
        # Dynamic state now has 4 features: [load, demand, time_elapsed, urgency]
        self.dynamic = torch.tensor(np.concatenate((loads, demands, time_elapsed, urgency), axis=1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1])

    def update_mask(self, mask, dynamic, chosen_idx=None):
        """Updates the mask used to hide non-valid states.

        Parameters
        ----------
        dynamic: torch.autograd.Variable of size (1, num_feats, seq_len)
            Now expects 4 features: [load, demand, time_elapsed, urgency]
        """

        # Convert floating point to integers for calculations
        loads = dynamic.data[:, 0]  # (batch_size, seq_len)
        demands = dynamic.data[:, 1]  # (batch_size, seq_len)
        # Note: time_elapsed is dynamic[:, 2] and urgency is dynamic[:, 3]

        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        if demands.eq(0).all():
            return demands * 0.

        # Otherwise, we can choose to go anywhere where demand is > 0
        new_mask = demands.ne(0) * demands.lt(loads)

        # We should avoid traveling to the depot back-to-back
        repeat_home = chosen_idx.ne(0)

        if repeat_home.any():
            new_mask[repeat_home.nonzero(), 0] = 1.
        if (1 - repeat_home).any():
            new_mask[(1 - repeat_home).nonzero(), 0] = 0.

        # ... unless we're waiting for all other samples in a minibatch to finish
        has_no_load = loads[:, 0].eq(0).float()
        has_no_demand = demands[:, 1:].sum(1).eq(0).float()

        combined = (has_no_load + has_no_demand).gt(0)
        if combined.any():
            new_mask[combined.nonzero(), 0] = 1.
            new_mask[combined.nonzero(), 1:] = 0.

        return new_mask.float()

    def update_dynamic(self, dynamic, chosen_idx):
        """Updates the (load, demand, time_elapsed, urgency) dataset values."""
        
        batch_size = dynamic.size(0)
        num_nodes = dynamic.size(2)

        # Update the dynamic elements differently for if we visit depot vs. a city
        visit = chosen_idx.ne(0)
        depot = chosen_idx.eq(0)

        # Clone the dynamic variable so we don't mess up graph
        all_loads = dynamic[:, 0].clone()
        all_demands = dynamic[:, 1].clone()
        all_time_elapsed = dynamic[:, 2].clone()
        all_urgency = dynamic[:, 3].clone()

        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))

        # Calculate travel time for this step (using traffic multipliers)
        # We need to track previous position to calculate travel
        # For simplicity, we'll use approximate travel time = base_distance * traffic
        # Since we don't have access to previous position here, we estimate travel time
        # as average edge length * traffic multiplier
        # A more accurate implementation would track previous_idx in the environment
        
        # Simplified travel time: assume average distance ~0.2 per step with traffic
        # This is a placeholder - in production, you'd track actual edges traversed
        avg_travel_time = 0.2  # Approximate average Euclidean distance
        # We'll apply traffic later when we have actual edge information
        # For now, just increment time by a normalized amount
        all_time_elapsed = all_time_elapsed + avg_travel_time
        
        # Across the minibatch - if we've chosen to visit a city, try to satisfy
        # as much demand as possible
        if visit.any():

            new_load = torch.clamp(load - demand, min=0)
            new_demand = torch.clamp(demand - load, min=0)

            # Broadcast the load to all nodes, but update demand seperately
            visit_idx = visit.nonzero().squeeze()

            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
            all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1)

        # Return to depot to fill vehicle load
        if depot.any():
            all_loads[depot.nonzero().squeeze()] = 1.
            all_demands[depot.nonzero().squeeze(), 0] = 0.
        
        # Update urgency: higher values = more urgent (closer to deadline)
        # urgency = max(0, 1 - (deadline - time_elapsed) / deadline)
        # Normalized so that urgency ∈ [0, 1]
        max_time = 30.0  # Approximate max tour time for normalization
        for i in range(batch_size):
            time_to_deadline = self.deadlines[i % self.num_samples] - all_time_elapsed[i]
            all_urgency[i] = torch.clamp(1.0 - time_to_deadline / max_time, min=0.0, max=1.0)

        tensor = torch.cat((all_loads.unsqueeze(1), 
                          all_demands.unsqueeze(1),
                          all_time_elapsed.unsqueeze(1),
                          all_urgency.unsqueeze(1)), 1)
        return torch.tensor(tensor.data, device=dynamic.device)


def reward(static, tour_indices, dynamic=None, deadlines=None, penalty_weight=2.0):
    """
    Calculate reward as negative tour length plus late delivery penalties.
    
    Parameters
    ----------
    static: Tensor of city locations
    tour_indices: Indices of cities visited in order
    dynamic: Optional dynamic state tensor (for extracting time information)
    deadlines: Optional deadlines tensor
    penalty_weight: Weight for late delivery penalty (default: 2.0)
    
    Returns
    -------
    Total cost (distance + penalties). Lower is better.
    """

    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Ensure we're always returning to the depot - note the extra concat
    # won't add any extra loss, as the euclidean distance between consecutive
    # points is 0
    start = static.data[:, :, 0].unsqueeze(1)
    y = torch.cat((start, tour, start), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
    
    total_distance = tour_len.sum(1)
    
    # If deadlines are provided, add late delivery penalty
    # Note: This is a simplified penalty calculation
    # In practice, you'd track actual arrival times per node during the tour
    if deadlines is not None and dynamic is not None:
        # Extract time_elapsed from dynamic state (feature index 2)
        # This is approximate - actual implementation would need per-node arrival tracking
        time_elapsed = dynamic[:, 2].max(dim=1)[0]  # Max time across all nodes
        
        # Simplified penalty: penalize if total tour time exceeds average deadline
        avg_deadline = deadlines.mean(dim=1)
        lateness = torch.clamp(time_elapsed - avg_deadline, min=0)
        penalty = penalty_weight * lateness
        
        return total_distance + penalty
    
    return total_distance


def heuristic_route(locations, deadlines, traffic_levels, urgency_weight=1.0, max_load=20, demands=None):
    """
    Greedy heuristic baseline for VRP with deadlines and traffic.
    
    This function is SEPARATE from the RL loop and can be used for comparison.
    
    Parameters
    ----------
    locations: numpy array or torch.Tensor of shape (2, N)
        (x, y) coordinates of nodes. Node 0 is the depot.
    deadlines: numpy array or torch.Tensor of shape (N,)
        Deadline for each node
    traffic_levels: numpy array or torch.Tensor of shape (N, N)
        Traffic multiplier matrix for edges [from_node, to_node]
    urgency_weight: float
        Weight for urgency penalty in route selection (default: 1.0)
    max_load: int
        Vehicle capacity (default: 20)
    demands: optional numpy array of shape (N,)
        Demand at each node. If None, assumes all demands = 1
    
    Returns
    -------
    route: list of int
        Order of nodes visited (excluding depot returns)
    total_time: float
        Total travel time including traffic
    """
    
    # Convert to numpy for easier manipulation
    if torch.is_tensor(locations):
        locations = locations.cpu().numpy()
    if torch.is_tensor(deadlines):
        deadlines = deadlines.cpu().numpy()
    if torch.is_tensor(traffic_levels):
        traffic_levels = traffic_levels.cpu().numpy()
    
    num_nodes = locations.shape[1]
    
    # Initialize demands if not provided
    if demands is None:
        demands = np.ones(num_nodes)
        demands[0] = 0  # Depot has no demand
    elif torch.is_tensor(demands):
        demands = demands.cpu().numpy()
    
    # Track state
    current_node = 0  # Start at depot
    current_load = max_load
    current_time = 0.0
    route = []
    visited = np.zeros(num_nodes, dtype=bool)
    visited[0] = True  # Depot is always "visited"
    remaining_demands = demands.copy()
    
    while not visited[1:].all():  # While there are unvisited customer nodes
        best_node = None
        best_score = float('inf')
        
        # Consider all unvisited nodes
        for candidate in range(1, num_nodes):
            if visited[candidate] or remaining_demands[candidate] == 0:
                continue
            
            # Check capacity constraint
            if remaining_demands[candidate] > current_load:
                continue
            
            # Calculate travel time to candidate
            dx = locations[0, candidate] - locations[0, current_node]
            dy = locations[1, candidate] - locations[1, current_node]
            base_distance = np.sqrt(dx**2 + dy**2)
            travel_time = base_distance * traffic_levels[current_node, candidate]
            
            # Estimate arrival time
            arrival_time = current_time + travel_time
            
            # Calculate urgency penalty (how late we'll be)
            lateness = max(0, arrival_time - deadlines[candidate])
            
            # Total score: travel time + urgency penalty
            score = travel_time + urgency_weight * lateness
            
            if score < best_score:
                best_score = score
                best_node = candidate
        
        # If no valid node found, return to depot
        if best_node is None:
            # Return to depot
            dx = locations[0, 0] - locations[0, current_node]
            dy = locations[1, 0] - locations[1, current_node]
            base_distance = np.sqrt(dx**2 + dy**2)
            travel_time = base_distance * traffic_levels[current_node, 0]
            current_time += travel_time
            current_node = 0
            current_load = max_load
            route.append(0)  # Mark depot visit
            continue
        
        # Move to best node
        dx = locations[0, best_node] - locations[0, current_node]
        dy = locations[1, best_node] - locations[1, current_node]
        base_distance = np.sqrt(dx**2 + dy**2)
        travel_time = base_distance * traffic_levels[current_node, best_node]
        
        current_time += travel_time
        current_load -= remaining_demands[best_node]
        remaining_demands[best_node] = 0
        visited[best_node] = True
        current_node = best_node
        route.append(best_node)
    
    # Return to depot at end
    dx = locations[0, 0] - locations[0, current_node]
    dy = locations[1, 0] - locations[1, current_node]
    base_distance = np.sqrt(dx**2 + dy**2)
    travel_time = base_distance * traffic_levels[current_node, 0]
    current_time += travel_time
    
    return route, current_time


def render(static, tour_indices, save_path):
    """Plots the found solution."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        # Assign each subtour a different colour & label in order traveled
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]

        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)

        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)


'''
def render(static, tour_indices, save_path):
    """Plots the found solution."""

    path = 'C:/Users/Matt/Documents/ffmpeg-3.4.2-win64-static/bin/ffmpeg.exe'
    plt.rcParams['animation.ffmpeg_path'] = path

    plt.close('all')

    num_plots = min(int(np.sqrt(len(tour_indices))), 3)
    fig, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                             sharex='col', sharey='row')
    axes = [a for ax in axes for a in ax]

    all_lines = []
    all_tours = []
    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        cur_tour = np.vstack((x, y))

        all_tours.append(cur_tour)
        all_lines.append(ax.plot([], [])[0])

        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

    from matplotlib.animation import FuncAnimation

    tours = all_tours

    def update(idx):

        for i, line in enumerate(all_lines):

            if idx >= tours[i].shape[1]:
                continue

            data = tours[i][:, idx]

            xy_data = line.get_xydata()
            xy_data = np.vstack((xy_data, np.atleast_2d(data)))

            line.set_data(xy_data[:, 0], xy_data[:, 1])
            line.set_linewidth(0.75)

        return all_lines

    anim = FuncAnimation(fig, update, init_func=None,
                         frames=100, interval=200, blit=False,
                         repeat=False)

    anim.save('line.mp4', dpi=160)
    plt.show()

    import sys
    sys.exit(1)
'''
