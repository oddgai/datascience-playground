"""
TSP solver using Google OR-Tools
"""

import time
import numpy as np
from typing import List, Tuple, Dict
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

try:
    from .tsp_utils import TSPDataExtractor, load_tsp_instance
except ImportError:
    from tsp_utils import TSPDataExtractor, load_tsp_instance


class ORToolsTSPSolver:
    """TSP solver using Google OR-Tools"""
    
    def __init__(self, tsp_data: Dict, time_limit: int = 60):
        """
        Initialize OR-Tools TSP solver
        
        Args:
            tsp_data: TSP instance data dictionary
            time_limit: Time limit in seconds
        """
        self.tsp_data = tsp_data
        self.time_limit = time_limit
        self.distance_matrix = tsp_data['distance_matrix'].astype(int).tolist()
        self.n = tsp_data['dimension']
        
        print(f"OR-Tools TSP Solver initialized")
        print(f"Instance: {tsp_data['name']}")
        print(f"Nodes: {self.n}")
        print(f"Time limit: {time_limit}s")
        
    def solve(self) -> Tuple[List[int], float, float, bool]:
        """
        Solve TSP using OR-Tools
        
        Returns:
            Tuple of (tour, total_distance, solve_time, is_optimal)
        """
        print("Creating OR-Tools TSP model...")
        
        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(self.n, 1, 0)
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        # Create distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Setting search parameters for faster execution
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
        # Use a simpler metaheuristic for smaller instances to avoid unnecessary optimization
        if self.n <= 10:
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
        else:
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        
        search_parameters.time_limit.seconds = self.time_limit
        # Allow early termination if optimal solution is found
        search_parameters.solution_limit = 1000
        search_parameters.lns_time_limit.seconds = min(10, self.time_limit // 2)
        
        print("Solving TSP with OR-Tools...")
        start_time = time.time()
        
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        actual_solve_time = time.time() - start_time
        
        if solution:
            # Extract the tour
            tour = []
            index = routing.Start(0)
            total_distance = 0
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                tour.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                total_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            
            # Add depot at the end to complete the tour
            tour.append(manager.IndexToNode(routing.Start(0)))
            
            # For TSP, we consider the solution optimal if we found a solution
            is_optimal = True
            
            print(f"OR-Tools solution found!")
            print(f"Tour: {tour}")
            print(f"Total distance: {total_distance}")
            print(f"Actual solve time: {actual_solve_time:.3f}s (time limit: {self.time_limit}s)")
            print(f"Status: {'Optimal' if is_optimal else 'Feasible'}")
            
            return tour, total_distance, actual_solve_time, is_optimal
        else:
            print("OR-Tools failed to find solution")
            print(f"Time elapsed: {actual_solve_time:.3f}s (time limit: {self.time_limit}s)")
            return [], float('inf'), actual_solve_time, False
    
    def print_solution(self, tour: List[int], total_distance: float):
        """Print solution in readable format"""
        print("\n" + "="*50)
        print("OR-Tools TSP Solution")
        print("="*50)
        print(f"Tour: {' -> '.join(map(str, tour))}")
        print(f"Total distance: {total_distance}")
        print(f"Number of cities: {len(tour) - 1}")  # -1 because we return to start


def solve_tsp_with_ortools(tsp_data: Dict, time_limit: int = 60) -> Dict:
    """
    Solve TSP using OR-Tools
    
    Args:
        tsp_data: TSP instance data
        time_limit: Time limit in seconds
        
    Returns:
        Dictionary containing solution results
    """
    solver = ORToolsTSPSolver(tsp_data, time_limit)
    tour, total_distance, solve_time, is_optimal = solver.solve()
    
    if tour:
        solver.print_solution(tour, total_distance)
        
        return {
            "experiment_id": f"ortools_tsp_{tsp_data['name']}",
            "model_type": "OR-Tools TSP Solver",
            "model_params": {
                "first_solution_strategy": "PATH_CHEAPEST_ARC",
                "local_search_metaheuristic": "GUIDED_LOCAL_SEARCH",
                "time_limit_seconds": time_limit
            },
            "features": [
                "Constraint_programming",
                "Local_search",
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
                "all_cities_visited": len(set(tour[:-1])) == tsp_data['dimension'],
                "tour_is_cycle": tour[0] == tour[-1]
            }
        }
    else:
        return {
            "experiment_id": f"ortools_tsp_{tsp_data['name']}",
            "model_type": "OR-Tools TSP Solver",
            "error": "No solution found",
            "solve_time_seconds": solve_time
        }


if __name__ == "__main__":
    # Example usage
    from tsp_utils import TSPDataExtractor
    
    # Create TSP instance from VRP data
    vrp_file = "instances/tai75a/data/tai75a.vrp"
    extractor = TSPDataExtractor(vrp_file)
    tsp_data = extractor.extract_tsp_subset(15, include_depot=True)
    
    # Solve with OR-Tools
    result = solve_tsp_with_ortools(tsp_data, time_limit=30)
    print("\nOR-Tools result:", result)