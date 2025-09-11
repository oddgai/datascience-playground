#!/usr/bin/env python3
"""
TSP route visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from typing import List, Tuple, Dict, Any

def visualize_tsp_route(tsp_data: Dict[str, Any], route: List[int], 
                       title: str = "TSP Route", 
                       save_path: str = None) -> str:
    """
    Visualize TSP route with coordinates and save to file
    
    Args:
        tsp_data: TSP instance data with coordinates and distance matrix
        route: Solution route as list of node indices
        title: Plot title
        save_path: Path to save the plot (optional)
    
    Returns:
        Path where the plot was saved
    """
    coordinates = tsp_data.get('coordinates')
    if coordinates is None:
        raise ValueError("TSP data must contain 'coordinates' for visualization")
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot all nodes - coordinates is a dict {0: (x, y), 1: (x, y), ...}
    x_coords = [coordinates[i][0] for i in range(len(coordinates))]
    y_coords = [coordinates[i][1] for i in range(len(coordinates))]
    
    plt.scatter(x_coords, y_coords, c='lightblue', s=100, alpha=0.7, zorder=2)
    
    # Highlight depot (node 0) in red
    plt.scatter(x_coords[0], y_coords[0], c='red', s=200, marker='s', 
                label='Depot', zorder=3)
    
    # Plot route
    if route:
        route_x = [coordinates[i][0] for i in route]
        route_y = [coordinates[i][1] for i in route]
        
        # Add depot at the end to close the route
        if route[0] != route[-1]:
            route_x.append(coordinates[route[0]][0])
            route_y.append(coordinates[route[0]][1])
        
        plt.plot(route_x, route_y, 'b-', linewidth=2, alpha=0.7, label='Route')
        
        # Add arrows to show direction
        for i in range(len(route_x) - 1):
            dx = route_x[i+1] - route_x[i]
            dy = route_y[i+1] - route_y[i]
            plt.arrow(route_x[i], route_y[i], dx*0.1, dy*0.1, 
                     head_width=2, head_length=1, fc='blue', ec='blue', alpha=0.6)
    
    # Add node labels
    for i in range(len(coordinates)):
        x, y = coordinates[i]
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save plot
    if save_path is None:
        save_path = f"tsp_route_{len(coordinates)}nodes.png"
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path

def create_route_comparison_plot(tsp_data: Dict[str, Any], 
                                ortools_route: List[int], 
                                mip_route: List[int],
                                ortools_cost: float,
                                mip_cost: float,
                                save_path: str = None) -> str:
    """
    Create side-by-side comparison of OR-Tools and MIP routes
    """
    coordinates = tsp_data.get('coordinates')
    if coordinates is None:
        raise ValueError("TSP data must contain 'coordinates' for visualization")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Common settings - coordinates is a dict {0: (x, y), 1: (x, y), ...}
    x_coords = [coordinates[i][0] for i in range(len(coordinates))]
    y_coords = [coordinates[i][1] for i in range(len(coordinates))]
    
    # OR-Tools plot
    ax1.scatter(x_coords, y_coords, c='lightblue', s=100, alpha=0.7, zorder=2)
    ax1.scatter(x_coords[0], y_coords[0], c='red', s=200, marker='s', 
                label='Depot', zorder=3)
    
    if ortools_route:
        route_x = [coordinates[i][0] for i in ortools_route]
        route_y = [coordinates[i][1] for i in ortools_route]
        
        if ortools_route[0] != ortools_route[-1]:
            route_x.append(coordinates[ortools_route[0]][0])
            route_y.append(coordinates[ortools_route[0]][1])
        
        ax1.plot(route_x, route_y, 'b-', linewidth=2, alpha=0.7, label='Route')
    
    for i in range(len(coordinates)):
        x, y = coordinates[i]
        ax1.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points',
                    fontsize=7, ha='left')
    
    ax1.set_title(f'OR-Tools Solution\nCost: {ortools_cost:.1f}', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # MIP plot
    ax2.scatter(x_coords, y_coords, c='lightcoral', s=100, alpha=0.7, zorder=2)
    ax2.scatter(x_coords[0], y_coords[0], c='darkred', s=200, marker='s', 
                label='Depot', zorder=3)
    
    if mip_route:
        route_x = [coordinates[i][0] for i in mip_route]
        route_y = [coordinates[i][1] for i in mip_route]
        
        if mip_route[0] != mip_route[-1]:
            route_x.append(coordinates[mip_route[0]][0])
            route_y.append(coordinates[mip_route[0]][1])
        
        ax2.plot(route_x, route_y, 'r-', linewidth=2, alpha=0.7, label='Route')
    
    for i in range(len(coordinates)):
        x, y = coordinates[i]
        ax2.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points',
                    fontsize=7, ha='left')
    
    ax2.set_title(f'MIP Solution\nCost: {mip_cost:.1f}', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = f"tsp_comparison_{len(coordinates)}nodes.png"
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return save_path