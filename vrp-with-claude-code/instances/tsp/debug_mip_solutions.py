#!/usr/bin/env python3
"""
Debug script to analyze MIP vs OR-Tools solution quality
Determines if crossing routes are due to plotting errors or suboptimal solutions
"""

import sys
import numpy as np
sys.path.append('src')
sys.path.append('../../src')
from tsp_utils import TSPDataExtractor
from tsp_ortools import solve_tsp_with_ortools
from tsp_mip import solve_tsp_with_mip
from tsp_visualization import visualize_tsp_route, create_route_comparison_plot

def calculate_tour_distance(tsp_data, tour):
    """Calculate total distance of a tour"""
    if not tour or len(tour) < 3:
        return float('inf')
    
    distance_matrix = tsp_data['distance_matrix']
    total_distance = 0
    
    for i in range(len(tour) - 1):
        from_node = tour[i]
        to_node = tour[i + 1]
        if from_node < len(distance_matrix) and to_node < len(distance_matrix):
            total_distance += distance_matrix[from_node][to_node]
        else:
            return float('inf')
    
    return total_distance

def validate_tour(tsp_data, tour):
    """Validate that a tour is valid (visits all nodes exactly once)"""
    n = tsp_data['dimension']
    
    if not tour:
        return False, "Empty tour"
    
    if len(tour) < 2:
        return False, f"Tour too short: {len(tour)}"
    
    # Check if tour starts and ends with depot
    if tour[0] != tour[-1]:
        return False, f"Tour doesn't start and end with depot: {tour[0]} != {tour[-1]}"
    
    # Check if all nodes are visited exactly once (excluding final depot)
    visited_nodes = set(tour[:-1])  # Exclude the final depot
    expected_nodes = set(range(n))
    
    if visited_nodes != expected_nodes:
        missing = expected_nodes - visited_nodes
        extra = visited_nodes - expected_nodes
        return False, f"Missing nodes: {missing}, Extra nodes: {extra}"
    
    return True, "Valid tour"

def detect_route_crossings(tsp_data, tour):
    """Detect if a route has crossing edges"""
    if not tour or len(tour) < 4:
        return [], "Tour too short to have crossings"
    
    coordinates = tsp_data['coordinates']
    crossings = []
    
    # Check each pair of edges for intersection
    edges = []
    for i in range(len(tour) - 1):
        from_node = tour[i]
        to_node = tour[i + 1]
        if from_node in coordinates and to_node in coordinates:
            edges.append((from_node, to_node, coordinates[from_node], coordinates[to_node]))
    
    def line_intersection(p1, q1, p2, q2):
        """Check if line segments p1q1 and p2q2 intersect"""
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0
            return 1 if val > 0 else 2
        
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
        
        # Special cases
        if (o1 == 0 and on_segment(p1, p2, q1)) or \
           (o2 == 0 and on_segment(p1, q2, q1)) or \
           (o3 == 0 and on_segment(p2, p1, q2)) or \
           (o4 == 0 and on_segment(p2, q1, q2)):
            return True
        
        return False
    
    for i in range(len(edges)):
        for j in range(i + 2, len(edges)):  # Skip adjacent edges
            if j == len(edges) - 1 and i == 0:  # Skip first-last edge pair
                continue
                
            edge1 = edges[i]
            edge2 = edges[j]
            
            if line_intersection(edge1[2], edge1[3], edge2[2], edge2[3]):
                crossings.append((edge1[:2], edge2[:2]))
    
    return crossings, f"Found {len(crossings)} crossings"

def analyze_solutions():
    """Analyze MIP vs OR-Tools solutions for quality comparison"""
    print("🔍 MIP Solution Quality Analysis")
    print("=" * 50)
    
    # Test with larger instance that shows crossing issue
    extractor = TSPDataExtractor("../tai75a/data/tai75a.vrp")
    tsp_data = extractor.extract_tsp_subset(50, include_depot=True)
    
    print(f"Testing with {tsp_data['dimension']}-node instance: {tsp_data['name']}")
    print()
    
    # Solve with OR-Tools (longer time limit for larger instance)
    print("Solving with OR-Tools...")
    ortools_result = solve_tsp_with_ortools(tsp_data, time_limit=300)  # 5 minutes
    ortools_tour = ortools_result.get('tour', [])
    ortools_cost = ortools_result.get('solution_cost', float('inf'))
    
    print(f"OR-Tools cost: {ortools_cost}")
    print(f"OR-Tools tour: {ortools_tour}")
    
    # Solve with MIP (longer time limit for larger instance)
    print("\nSolving with MIP...")
    mip_result = solve_tsp_with_mip(tsp_data, time_limit=1800)  # 30 minutes
    mip_tour = mip_result.get('tour', [])
    mip_cost = mip_result.get('solution_cost', float('inf'))
    
    print(f"MIP cost: {mip_cost}")
    print(f"MIP tour: {mip_tour}")
    
    print("\n" + "=" * 50)
    print("SOLUTION VALIDATION")
    print("=" * 50)
    
    # Validate OR-Tools tour
    ortools_valid, ortools_msg = validate_tour(tsp_data, ortools_tour)
    print(f"OR-Tools tour valid: {ortools_valid} - {ortools_msg}")
    
    if ortools_valid:
        ortools_calculated_cost = calculate_tour_distance(tsp_data, ortools_tour)
        print(f"OR-Tools calculated distance: {ortools_calculated_cost}")
        print(f"OR-Tools reported distance: {ortools_cost}")
        print(f"OR-Tools distance match: {abs(ortools_calculated_cost - ortools_cost) < 0.01}")
    
    # Validate MIP tour
    mip_valid, mip_msg = validate_tour(tsp_data, mip_tour)
    print(f"\nMIP tour valid: {mip_valid} - {mip_msg}")
    
    if mip_valid:
        mip_calculated_cost = calculate_tour_distance(tsp_data, mip_tour)
        print(f"MIP calculated distance: {mip_calculated_cost}")
        print(f"MIP reported distance: {mip_cost}")
        print(f"MIP distance match: {abs(mip_calculated_cost - mip_cost) < 0.01}")
    
    print("\n" + "=" * 50)
    print("CROSSING ANALYSIS")
    print("=" * 50)
    
    # Check for crossings
    if ortools_valid:
        ortools_crossings, ortools_cross_msg = detect_route_crossings(tsp_data, ortools_tour)
        print(f"OR-Tools: {ortools_cross_msg}")
        if ortools_crossings:
            print(f"  Crossing edges: {ortools_crossings[:3]}{'...' if len(ortools_crossings) > 3 else ''}")
    
    if mip_valid:
        mip_crossings, mip_cross_msg = detect_route_crossings(tsp_data, mip_tour)
        print(f"MIP: {mip_cross_msg}")
        if mip_crossings:
            print(f"  Crossing edges: {mip_crossings[:3]}{'...' if len(mip_crossings) > 3 else ''}")
    
    print("\n" + "=" * 50)
    print("QUALITY COMPARISON")
    print("=" * 50)
    
    if ortools_valid and mip_valid:
        if ortools_cost < mip_cost:
            gap = (mip_cost - ortools_cost) / ortools_cost * 100
            print(f"OR-Tools wins by {gap:.2f}% ({ortools_cost:.1f} vs {mip_cost:.1f})")
        elif mip_cost < ortools_cost:
            gap = (ortools_cost - mip_cost) / mip_cost * 100
            print(f"MIP wins by {gap:.2f}% ({mip_cost:.1f} vs {ortools_cost:.1f})")
        else:
            print("Tie!")
            
        # Check if MIP crossings indicate suboptimal solution
        if len(mip_crossings) > 0 and len(ortools_crossings) == 0:
            print("⚠️  MIP has crossings while OR-Tools doesn't - likely suboptimal MIP solution")
        elif len(ortools_crossings) > 0 and len(mip_crossings) == 0:
            print("⚠️  OR-Tools has crossings while MIP doesn't - unexpected")
        elif len(mip_crossings) > 0 and len(ortools_crossings) > 0:
            print("⚠️  Both solutions have crossings - check algorithm implementations")
        else:
            print("✅ Neither solution has crossings - both appear optimal")
    
    # Create visualization to verify
    if ortools_valid and mip_valid:
        print(f"\n📊 Creating comparison visualization...")
        viz_file = create_route_comparison_plot(
            tsp_data, ortools_tour, mip_tour, ortools_cost, mip_cost,
            save_path="debug_comparison.png"
        )
        print(f"Saved comparison plot: {viz_file}")
    
    print(f"\n🎯 CONCLUSION:")
    if mip_valid and len(mip_crossings) > 0:
        print("The MIP solution has crossings, indicating it's SUBOPTIMAL.")
        print("This is likely due to the MIP solver not finding the optimal solution within time limits,")
        print("or issues with the MTZ subtour elimination constraints.")
        print("The visualization is CORRECTLY showing the suboptimal MIP route.")
    elif not mip_valid:
        print("The MIP solution is INVALID - there's a bug in the solution extraction.")
        print("The crossings are due to an incorrectly constructed tour.")
    else:
        print("The MIP solution appears valid without crossings.")

if __name__ == "__main__":
    analyze_solutions()