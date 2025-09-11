"""
Facility Location Problem solver using Mixed Integer Programming (PuLP)
Solves the p-median problem: given n locations with demands, select p facilities to minimize total cost
"""

import time
import numpy as np
from typing import List, Tuple, Dict
import pulp

try:
    from .facility_utils import FacilityLocationDataExtractor, load_facility_location_data
except ImportError:
    from facility_utils import FacilityLocationDataExtractor, load_facility_location_data


class MIPFacilityLocationSolver:
    """Facility Location solver using Mixed Integer Programming with PuLP"""
    
    def __init__(self, fl_data: Dict, num_facilities: int = 10, time_limit: int = 300):
        """
        Initialize MIP Facility Location solver
        
        Args:
            fl_data: Facility location problem data dictionary
            num_facilities: Number of facilities to locate (p in p-median problem)
            time_limit: Time limit in seconds
        """
        self.fl_data = fl_data
        self.num_facilities = num_facilities
        self.time_limit = time_limit
        self.distance_matrix = fl_data['distance_matrix']
        self.demands = fl_data['demands']
        self.coordinates = fl_data['coordinates']
        self.n = fl_data['num_locations']
        
        print(f"MIP Facility Location Solver initialized")
        print(f"Instance: {fl_data['name']}")
        print(f"Locations: {self.n}")
        print(f"Facilities to locate: {num_facilities}")
        print(f"Total demand: {fl_data['total_demand']}")
        print(f"Time limit: {time_limit}s")
        
    def solve(self) -> Tuple[List[int], Dict[int, int], float, float, float, bool, str]:
        """
        Solve Facility Location using MIP
        
        Returns:
            Tuple of (facility_locations, assignments, total_cost, solve_time, gap, is_optimal, solver_used)
        """
        print("Creating MIP Facility Location model...")
        
        # Create the model
        model = pulp.LpProblem("FacilityLocation", pulp.LpMinimize)
        
        # Decision variables
        # y[j] = 1 if facility j is opened, 0 otherwise
        y = {}
        for j in range(self.n):
            y[j] = pulp.LpVariable(f"y_{j}", cat='Binary')
        
        # x[i][j] = 1 if demand point i is assigned to facility j, 0 otherwise
        x = {}
        for i in range(self.n):
            for j in range(self.n):
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
        
        print("Adding objective and constraints...")
        
        # Objective: minimize total transportation cost (demand * distance)
        model += pulp.lpSum(
            self.demands[i] * self.distance_matrix[i][j] * x[(i, j)]
            for i in range(self.n)
            for j in range(self.n)
        )
        
        # Constraint 1: Each demand point must be assigned to exactly one facility
        for i in range(self.n):
            model += pulp.lpSum(x[(i, j)] for j in range(self.n)) == 1
        
        # Constraint 2: Can only assign to open facilities
        for i in range(self.n):
            for j in range(self.n):
                model += x[(i, j)] <= y[j]
        
        # Constraint 3: Open exactly num_facilities facilities
        model += pulp.lpSum(y[j] for j in range(self.n)) == self.num_facilities
        
        # Try different solvers in order of preference
        solvers_to_try = [
            ('HiGHS_CMD', pulp.HiGHS_CMD),
            ('PULP_CBC_CMD', pulp.PULP_CBC_CMD),
            ('COIN_CMD', pulp.COIN_CMD),
        ]
        
        solver_used = None
        start_time = time.time()
        gap = None
        
        for solver_name, solver_class in solvers_to_try:
            try:
                print(f"Trying {solver_name} solver...")
                solver = solver_class(timeLimit=self.time_limit, gapRel=0.01)
                
                # Solve the model
                model.solve(solver)
                
                if model.status == pulp.LpStatusOptimal:
                    solver_used = solver_name
                    gap = 0.0
                    print(f"Optimal solution found with {solver_name}!")
                    break
                elif model.status == pulp.LpStatusFeasible:
                    solver_used = solver_name
                    # Try to get gap from solver if available
                    if hasattr(solver, 'actualGap'):
                        gap = solver.actualGap
                    else:
                        gap = None
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
            return [], {}, float('inf'), solve_time, None, False, "NONE"
        
        # Extract solution
        total_cost = pulp.value(model.objective)
        facility_locations = self._extract_facilities(y)
        assignments = self._extract_assignments(x)
        is_optimal = (model.status == pulp.LpStatusOptimal)
        
        print(f"Solution cost: {total_cost}")
        print(f"Facility locations: {facility_locations}")
        print(f"Solver used: {solver_used}")
        print(f"Status: {'Optimal' if is_optimal else 'Feasible'}")
        if gap is not None:
            print(f"Gap: {gap:.4f}")
        
        return facility_locations, assignments, total_cost, solve_time, gap, is_optimal, solver_used
    
    def _extract_facilities(self, y) -> List[int]:
        """Extract facility locations from solution variables"""
        facilities = []
        for j in range(self.n):
            if pulp.value(y[j]) > 0.5:  # Binary variable should be 0 or 1
                facilities.append(j)
        
        print(f"Found {len(facilities)} facilities: {facilities}")
        return facilities
    
    def _extract_assignments(self, x) -> Dict[int, int]:
        """Extract demand point assignments from solution variables"""
        assignments = {}
        for i in range(self.n):
            for j in range(self.n):
                if pulp.value(x[(i, j)]) > 0.5:  # Binary variable should be 0 or 1
                    assignments[i] = j
                    break  # Each demand point assigned to exactly one facility
        
        print(f"Assignment summary: {len(assignments)} demand points assigned")
        return assignments
    
    def validate_solution(self, facility_locations: List[int], assignments: Dict[int, int]) -> Tuple[bool, List[str]]:
        """Validate the solution"""
        errors = []
        
        # Check number of facilities
        if len(facility_locations) != self.num_facilities:
            errors.append(f"Expected {self.num_facilities} facilities, got {len(facility_locations)}")
        
        # Check all demand points are assigned
        if len(assignments) != self.n:
            errors.append(f"Expected {self.n} assignments, got {len(assignments)}")
        
        # Check assignments point to open facilities
        for demand_point, facility in assignments.items():
            if facility not in facility_locations:
                errors.append(f"Demand point {demand_point} assigned to closed facility {facility}")
        
        # Check assignment integrity
        for i in range(self.n):
            if i not in assignments:
                errors.append(f"Demand point {i} is not assigned to any facility")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def calculate_objective_value(self, facility_locations: List[int], assignments: Dict[int, int]) -> float:
        """Calculate the objective value from solution"""
        total_cost = 0.0
        
        for demand_point, facility in assignments.items():
            demand = self.demands[demand_point]
            distance = self.distance_matrix[demand_point][facility]
            cost = demand * distance
            total_cost += cost
        
        return total_cost
    
    def print_solution(self, facility_locations: List[int], assignments: Dict[int, int], total_cost: float):
        """Print solution in readable format"""
        print("\\n" + "="*60)
        print("MIP Facility Location Solution")
        print("="*60)
        print(f"Facility locations: {facility_locations}")
        print(f"Number of facilities: {len(facility_locations)}")
        print(f"Total cost: {total_cost:.2f}")
        
        # Print assignment summary
        print("\\nAssignment summary:")
        facility_loads = {}
        for demand_point, facility in assignments.items():
            if facility not in facility_loads:
                facility_loads[facility] = 0
            facility_loads[facility] += self.demands[demand_point]
        
        for facility in sorted(facility_locations):
            load = facility_loads.get(facility, 0)
            print(f"  Facility {facility}: serves demand {load:.1f}")


def solve_facility_location_with_mip(fl_data: Dict, num_facilities: int = 10, time_limit: int = 300) -> Dict:
    """
    Solve Facility Location using MIP
    
    Args:
        fl_data: Facility location problem data
        num_facilities: Number of facilities to locate
        time_limit: Time limit in seconds
        
    Returns:
        Dictionary containing solution results
    """
    solver = MIPFacilityLocationSolver(fl_data, num_facilities, time_limit)
    facility_locations, assignments, total_cost, solve_time, gap, is_optimal, solver_used = solver.solve()
    
    if facility_locations:
        solver.print_solution(facility_locations, assignments, total_cost)
        
        # Validate solution
        is_valid, errors = solver.validate_solution(facility_locations, assignments)
        
        # Calculate verification
        calculated_cost = solver.calculate_objective_value(facility_locations, assignments)
        cost_match = abs(calculated_cost - total_cost) < 0.01
        
        return {
            "experiment_id": f"facility_location_{fl_data['name']}_{num_facilities}fac",
            "model_type": f"MIP Facility Location Solver with {solver_used}",
            "model_params": {
                "solver": solver_used,
                "time_limit_seconds": time_limit,
                "max_gap": 0.01,
                "num_facilities": num_facilities,
                "formulation": "p-median"
            },
            "features": [
                "MIP_formulation",
                "p_median_problem", 
                "EUC_2D_distance",
                "facility_location"
            ],
            "instance_info": {
                "name": fl_data['name'],
                "num_locations": fl_data['num_locations'],
                "total_demand": fl_data['total_demand'],
                "num_facilities": num_facilities,
                "problem_type": "Facility_Location"
            },
            "solve_time_seconds": solve_time,
            "solution_cost": total_cost,
            "optimization_gap": gap,
            "is_optimal": is_optimal,
            "facility_locations": facility_locations,
            "assignments": assignments,
            "solution_quality": {
                "feasible": True,
                "valid": is_valid,
                "validation_errors": errors,
                "cost_verification": calculated_cost,
                "cost_match": cost_match,
                "num_facilities_correct": len(facility_locations) == num_facilities,
                "all_demands_assigned": len(assignments) == fl_data['num_locations']
            }
        }
    else:
        return {
            "experiment_id": f"facility_location_{fl_data['name']}_{num_facilities}fac",
            "model_type": f"MIP Facility Location Solver with {solver_used}",
            "error": "No solution found",
            "solve_time_seconds": solve_time,
            "optimization_gap": gap
        }


if __name__ == "__main__":
    # Example usage
    from facility_utils import load_facility_location_data
    
    # Create Facility Location instance from tai100a data
    vrp_file = "../tai100a/data/tai100a.vrp"
    fl_data = load_facility_location_data(vrp_file, exclude_depot=True)
    
    print("Facility Location Problem Data:")
    extractor = FacilityLocationDataExtractor(vrp_file)
    print(extractor.get_problem_summary(fl_data))
    
    # Solve with MIP
    print("\\n" + "="*60)
    result = solve_facility_location_with_mip(fl_data, num_facilities=10, time_limit=300)
    print("\\nMIP result keys:", list(result.keys()))