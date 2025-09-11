#!/usr/bin/env python3
"""
Facility Location Problem visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import japanize_matplotlib
from typing import List, Tuple, Dict, Any


def visualize_facility_location(fl_data: Dict[str, Any], 
                              facility_locations: List[int],
                              assignments: Dict[int, int] = None,
                              title: str = "Facility Location Solution",
                              save_path: str = None) -> str:
    """
    Visualize facility location solution
    
    Args:
        fl_data: Facility location problem data with coordinates and demands
        facility_locations: List of facility location indices
        assignments: Dictionary mapping demand points to assigned facilities
        title: Plot title
        save_path: Path to save the plot (optional)
    
    Returns:
        Path where the plot was saved
    """
    coordinates = fl_data.get('coordinates')
    demands = fl_data.get('demands')
    if coordinates is None:
        raise ValueError("FL data must contain 'coordinates' for visualization")
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Get all coordinates and demands
    n = fl_data['num_locations']
    x_coords = [coordinates[i][0] for i in range(n)]
    y_coords = [coordinates[i][1] for i in range(n)]
    demand_values = [demands[i] for i in range(n)]
    
    # Create colormap for facilities (each facility gets a unique color) - use more vibrant colors
    facility_colors = plt.cm.tab10(np.linspace(0, 0.9, len(facility_locations)))
    facility_color_map = {fac: color for fac, color in zip(facility_locations, facility_colors)}
    
    # Plot assignment lines first (so they appear behind points)
    if assignments:
        for demand_point, facility in assignments.items():
            if demand_point != facility:  # Don't draw line to self
                x_start, y_start = coordinates[demand_point]
                x_end, y_end = coordinates[facility]
                color = facility_color_map.get(facility, 'gray')
                plt.plot([x_start, x_end], [y_start, y_end], 
                        color=color, alpha=0.5, linewidth=1.5, zorder=1)
    
    # Normalize demand values for point sizes
    min_demand = min(demand_values)
    max_demand = max(demand_values)
    demand_range = max_demand - min_demand if max_demand > min_demand else 1
    
    # Plot demand points
    demand_colors = []
    point_sizes = []
    
    for i in range(n):
        demand = demand_values[i]
        
        # Size based on demand (larger demand = larger point) - more dramatic sizing, larger overall
        normalized_demand = (demand - min_demand) / demand_range
        size = 20 + normalized_demand * 500  # Size between 20 and 520 for dramatic effect, larger overall
        point_sizes.append(size)
        
        # Color based on assigned facility (if assignments provided)
        if assignments and i in assignments:
            assigned_facility = assignments[i]
            if assigned_facility in facility_color_map:
                demand_colors.append(facility_color_map[assigned_facility])
            else:
                demand_colors.append('lightgray')
        else:
            demand_colors.append('lightblue')
    
    # Scatter plot for demand points (excluding facilities which get star markers)
    demand_only_x = [x_coords[i] for i in range(n) if i not in facility_locations]
    demand_only_y = [y_coords[i] for i in range(n) if i not in facility_locations]
    demand_only_colors = [demand_colors[i] for i in range(n) if i not in facility_locations]
    demand_only_sizes = [point_sizes[i] for i in range(n) if i not in facility_locations]
    
    if demand_only_x:  # Only plot if there are demand-only points
        scatter = plt.scatter(demand_only_x, demand_only_y, 
                             c=demand_only_colors, 
                             s=demand_only_sizes,
                             alpha=0.7, 
                             edgecolors='black',
                             linewidth=0.5,
                             zorder=2)
    
    # Highlight facilities with special markers
    facility_x = [coordinates[fac][0] for fac in facility_locations]
    facility_y = [coordinates[fac][1] for fac in facility_locations]
    facility_demands = [demands[fac] for fac in facility_locations]
    
    # Plot facilities as large stars
    for i, fac in enumerate(facility_locations):
        x, y = coordinates[fac]
        color = facility_color_map[fac]
        plt.scatter(x, y, 
                   marker='*', 
                   s=600, 
                   c=[color], 
                   edgecolors='darkred',
                   linewidth=2,
                   zorder=4,
                   label=f'Facility {fac}' if i < 5 else "")  # Only label first 5
    
    # Facility numbers removed as requested
    
    # Removed demand point labels as requested
    
    # Legend removed as requested
    
    # Statistics text box removed as requested
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = f"facility_location_{len(facility_locations)}fac_{n}loc.png"
    
    plt.savefig(save_path, dpi=80, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path


def create_facility_analysis_plot(fl_data: Dict[str, Any],
                                facility_locations: List[int],
                                assignments: Dict[int, int],
                                total_cost: float,
                                save_path: str = None) -> str:
    """
    Create detailed analysis plot with cost breakdown
    """
    coordinates = fl_data.get('coordinates')
    demands = fl_data.get('demands')
    distance_matrix = fl_data.get('distance_matrix')
    
    if not all([coordinates, demands, distance_matrix is not None]):
        raise ValueError("FL data must contain coordinates, demands, and distance_matrix")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: Facility locations with service areas
    n = fl_data['num_locations']
    x_coords = [coordinates[i][0] for i in range(n)]
    y_coords = [coordinates[i][1] for i in range(n)]
    demand_values = [demands[i] for i in range(n)]
    
    # Create Voronoi-like regions by color coding assignments
    facility_colors = plt.cm.Set3(np.linspace(0, 1, len(facility_locations)))
    facility_color_map = {fac: color for fac, color in zip(facility_locations, facility_colors)}
    
    # Plot service areas (demand points colored by assigned facility)
    demand_colors = []
    for i in range(n):
        if i in assignments:
            assigned_facility = assignments[i]
            demand_colors.append(facility_color_map.get(assigned_facility, 'gray'))
        else:
            demand_colors.append('lightgray')
    
    # Size by demand
    min_demand = min(demand_values)
    max_demand = max(demand_values)
    demand_range = max_demand - min_demand if max_demand > min_demand else 1
    point_sizes = [30 + ((d - min_demand) / demand_range) * 100 for d in demand_values]
    
    ax1.scatter(x_coords, y_coords, c=demand_colors, s=point_sizes, 
               alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Plot facilities
    for fac in facility_locations:
        x, y = coordinates[fac]
        color = facility_color_map[fac]
        ax1.scatter(x, y, marker='*', s=500, c=[color], 
                   edgecolors='darkred', linewidth=3)
        ax1.annotate(f'F{fac}', (x, y), xytext=(10, 10), 
                    textcoords='offset points', fontsize=12, 
                    fontweight='bold', color='darkred')
    
    ax1.set_title('Facility Locations & Service Areas', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Right plot: Cost analysis
    facility_costs = {}
    facility_loads = {}
    
    for demand_point, facility in assignments.items():
        demand = demands[demand_point]
        distance = distance_matrix[demand_point][facility]
        cost = demand * distance
        
        if facility not in facility_costs:
            facility_costs[facility] = 0
            facility_loads[facility] = 0
        
        facility_costs[facility] += cost
        facility_loads[facility] += demand
    
    # Bar plot of facility costs and loads
    facilities = sorted(facility_locations)
    costs = [facility_costs.get(f, 0) for f in facilities]
    loads = [facility_loads.get(f, 0) for f in facilities]
    
    x_pos = np.arange(len(facilities))
    width = 0.35
    
    # Normalize loads for second y-axis
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar(x_pos - width/2, costs, width, label='Cost', 
                   color=[facility_color_map[f] for f in facilities], alpha=0.7)
    bars2 = ax2_twin.bar(x_pos + width/2, loads, width, label='Demand Load', 
                        color='lightcoral', alpha=0.7)
    
    ax2.set_xlabel('Facility ID')
    ax2.set_ylabel('Total Cost', color='blue')
    ax2_twin.set_ylabel('Demand Load', color='red')
    ax2.set_title(f'Facility Cost & Load Analysis\\nTotal Cost: {total_cost:.2f}', 
                 fontsize=12, fontweight='bold')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'F{f}' for f in facilities])
    
    # Add value labels on bars
    for i, (cost, load) in enumerate(zip(costs, loads)):
        ax2.text(i - width/2, cost + max(costs) * 0.01, f'{cost:.1f}', 
                ha='center', va='bottom', fontsize=10)
        ax2_twin.text(i + width/2, load + max(loads) * 0.01, f'{load:.1f}', 
                     ha='center', va='bottom', fontsize=10, color='red')
    
    # Create combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = f"facility_analysis_{len(facility_locations)}fac.png"
    
    plt.savefig(save_path, dpi=80, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path


if __name__ == "__main__":
    # Example usage - this would normally be called from the main experiment
    print("Facility location visualization module loaded successfully")
    print("Use visualize_facility_location() and create_facility_analysis_plot() functions")