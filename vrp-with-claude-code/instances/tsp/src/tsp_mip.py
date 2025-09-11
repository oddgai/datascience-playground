"""
TSP solver using Mixed Integer Programming (PuLP)
"""

import time
import numpy as np
from typing import List, Tuple, Dict
import pulp

try:
    from .tsp_utils import TSPDataExtractor, load_tsp_instance
except ImportError:
    from tsp_utils import TSPDataExtractor, load_tsp_instance


class MIPTSPSolver:
    """TSP solver using Mixed Integer Programming with PuLP"""
    
    def __init__(self, tsp_data: Dict, time_limit: int = 60):
        """
        Initialize MIP TSP solver
        
        Args:
            tsp_data: TSP instance data dictionary
            time_limit: Time limit in seconds
        """
        self.tsp_data = tsp_data
        self.time_limit = time_limit
        self.distance_matrix = tsp_data['distance_matrix']
        self.n = tsp_data['dimension']
        
        print(f"MIP TSP Solver initialized")
        print(f"Instance: {tsp_data['name']}")
        print(f"Nodes: {self.n}")
        print(f"Time limit: {time_limit}s")
        
    def solve(self) -> Tuple[List[int], float, float, bool, str]:
        """
        Solve TSP using MIP
        
        Returns:
            Tuple of (tour, total_distance, solve_time, is_optimal, solver_used)
        """
        print("Creating MIP TSP model...")
        
        # Create the model
        model = pulp.LpProblem("TSP", pulp.LpMinimize)
        
        # Decision variables: x[i][j] = 1 if we travel from city i to city j
        x = {}
        for i in range(self.n):
            for j in range(self.n):
                if i != j:  # No self-loops
                    x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
        
        # Variables for MTZ subtour elimination: u[i] = position of city i in tour
        u = {}
        for i in range(1, self.n):  # Exclude starting city (city 0)
            u[i] = pulp.LpVariable(f"u_{i}", lowBound=1, upBound=self.n-1, cat='Continuous')
        
        print("Adding objective and constraints...")
        
        # Objective: minimize total distance
        model += pulp.lpSum(self.distance_matrix[i][j] * x[(i, j)] 
                           for i in range(self.n) 
                           for j in range(self.n) 
                           if i != j)
        
        # Constraint 1: Each city must be left exactly once
        for i in range(self.n):
            model += pulp.lpSum(x[(i, j)] for j in range(self.n) if i != j) == 1
        
        # Constraint 2: Each city must be entered exactly once
        for j in range(self.n):
            model += pulp.lpSum(x[(i, j)] for i in range(self.n) if i != j) == 1
        
        # Constraint 3: MTZ subtour elimination constraints
        for i in range(1, self.n):
            for j in range(1, self.n):
                if i != j:
                    model += u[i] - u[j] + self.n * x[(i, j)] <= self.n - 1
        
        # Try different solvers in order of preference - HiGHS first!
        solvers_to_try = [
            ('HiGHS_CMD', pulp.HiGHS_CMD),
            ('PULP_CBC_CMD', pulp.PULP_CBC_CMD),
            ('COIN_CMD', pulp.COIN_CMD),
        ]
        
        solver_used = None
        start_time = time.time()
        
        for solver_name, solver_class in solvers_to_try:
            try:
                print(f"Trying {solver_name} solver...")
                solver = solver_class(timeLimit=self.time_limit, gapRel=0.01)
                
                # Solve the model
                model.solve(solver)
                
                if model.status == pulp.LpStatusOptimal:
                    solver_used = solver_name
                    print(f"Optimal solution found with {solver_name}!")
                    break
                elif model.status == pulp.LpStatusFeasible:
                    solver_used = solver_name
                    print(f"Feasible solution found with {solver_name}!")
                    break
                else:
                    print(f"{solver_name} failed with status: {pulp.LpStatus[model.status]}")
                    continue
                    
            except Exception as e:
                print(f"{solver_name} failed with error: {e}")
                continue
        
        solve_time = time.time() - start_time
        
        if solver_used is None:
            print("All solvers failed!")
            return [], float('inf'), solve_time, False, "NONE"
        
        # Extract solution
        total_distance = pulp.value(model.objective)
        tour = self._extract_tour(x)
        is_optimal = (model.status == pulp.LpStatusOptimal)
        
        print(f"Solution cost: {total_distance}")
        print(f"Tour: {tour}")
        print(f"Solver used: {solver_used}")
        print(f"Status: {'Optimal' if is_optimal else 'Feasible'}")
        
        return tour, total_distance, solve_time, is_optimal, solver_used
    
    def _extract_tour(self, x) -> List[int]:
        """Extract tour from solution variables with robust subtour handling"""
        # Build adjacency representation from solution
        edges = {}
        for i in range(self.n):
            for j in range(self.n):
                if i != j and (i, j) in x and pulp.value(x[(i, j)]) > 0.5:
                    edges[i] = j
        
        print(f"Found {len(edges)} edges in MIP solution")
        
        # Check for complete tour starting from node 0
        if 0 not in edges:
            print("ERROR: No outgoing edge from depot (node 0)")
            return []
        
        tour = []
        current = 0
        visited = set()
        
        # Follow the path
        for step in range(self.n):
            if current in visited:
                print(f"ERROR: Cycle detected at step {step}, current node: {current}")
                break
                
            tour.append(current)
            visited.add(current)
            
            if current not in edges:
                print(f"ERROR: No outgoing edge from node {current}")
                break
                
            current = edges[current]
        
        # Verify we have a valid complete tour
        if len(tour) != self.n:
            print(f"WARNING: Incomplete tour - expected {self.n} nodes, got {len(tour)}")
            print(f"Missing nodes: {set(range(self.n)) - visited}")
            
            # Try to construct tour from subtours
            return self._handle_subtours(edges)
        
        # Add return to depot
        tour.append(0)
        return tour
    
    def _handle_subtours(self, edges) -> List[int]:
        """Handle case with subtours by finding largest subtour containing depot"""
        print("Attempting to handle subtours...")
        
        # Find all cycles/subtours
        subtours = []
        unvisited = set(range(self.n))
        
        while unvisited:
            # Start a new subtour from any unvisited node
            start = min(unvisited)  # Use depot (0) if available
            if 0 in unvisited:
                start = 0
                
            subtour = []
            current = start
            
            # Follow the cycle
            while current is not None and current in unvisited:
                subtour.append(current)
                unvisited.remove(current)
                current = edges.get(current)
                
                if current == start:  # Completed cycle
                    break
                if current is None or current not in unvisited:
                    break
            
            if len(subtour) > 1:
                subtours.append(subtour)
        
        print(f"Found {len(subtours)} subtours: {[len(st) for st in subtours]}")
        
        # Return largest subtour containing depot, or first subtour if no depot
        depot_subtour = None
        for subtour in subtours:
            if 0 in subtour:
                depot_subtour = subtour
                break
        
        if depot_subtour:
            # Rotate to start with depot
            depot_idx = depot_subtour.index(0)
            result = depot_subtour[depot_idx:] + depot_subtour[:depot_idx]
            result.append(0)  # Return to depot
            print(f"Returning depot subtour: {result}")
            return result
        
        # No depot subtour found, return largest subtour
        if subtours:
            largest = max(subtours, key=len)
            largest.append(largest[0])  # Close the loop
            print(f"Returning largest subtour: {largest}")
            return largest
        
        print("ERROR: No valid subtours found")
        return []
    
    def print_solution(self, tour: List[int], total_distance: float):
        """Print solution in readable format"""
        print("\n" + "="*50)
        print("MIP TSP Solution")
        print("="*50)
        print(f"Tour: {' -> '.join(map(str, tour))}")
        print(f"Total distance: {total_distance}")
        print(f"Number of cities: {len(set(tour)) if tour else 0}")


def solve_tsp_with_mip(tsp_data: Dict, time_limit: int = 60) -> Dict:
    """
    Solve TSP using MIP
    
    Args:
        tsp_data: TSP instance data
        time_limit: Time limit in seconds
        
    Returns:
        Dictionary containing solution results
    """
    solver = MIPTSPSolver(tsp_data, time_limit)
    tour, total_distance, solve_time, is_optimal, solver_used = solver.solve()
    
    if tour:
        solver.print_solution(tour, total_distance)
        
        return {
            "experiment_id": f"mip_tsp_{tsp_data['name']}",
            "model_type": f"MIP TSP Solver with {solver_used}",
            "model_params": {
                "solver": solver_used,
                "time_limit_seconds": time_limit,
                "max_gap": 0.01,
                "formulation": "MTZ"
            },
            "features": [
                "MIP_formulation",
                "MTZ_subtour_elimination",
                "EUC_2D_distance",
                "TSP_formulation"
            ],
            "instance_info": {
                "name": tsp_data['name'],
                "dimension": tsp_data['dimension'],
                "problem_type": "TSP"
            },
            "solve_time_seconds": solve_time,
            "solution_cost": total_distance,
            "is_optimal": is_optimal,
            "tour": tour,
            "solution_quality": {
                "feasible": True,
                "all_cities_visited": len(set(tour[:-1])) == tsp_data['dimension'] if tour else False,
                "tour_is_cycle": tour[0] == tour[-1] if len(tour) > 1 else False
            }
        }
    else:
        return {
            "experiment_id": f"mip_tsp_{tsp_data['name']}",
            "model_type": f"MIP TSP Solver with {solver_used}",
            "error": "No solution found",
            "solve_time_seconds": solve_time
        }


if __name__ == "__main__":
    # Example usage
    from tsp_utils import TSPDataExtractor
    
    # Create TSP instance from VRP data
    vrp_file = "instances/tai75a/data/tai75a.vrp"
    extractor = TSPDataExtractor(vrp_file)
    tsp_data = extractor.extract_tsp_subset(10, include_depot=True)  # Small instance for testing
    
    # Solve with MIP
    result = solve_tsp_with_mip(tsp_data, time_limit=60)
    print("\nMIP result:", result)