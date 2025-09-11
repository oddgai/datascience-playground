"""
Facility Location Problem utilities for extracting data from VRP instances
"""

import numpy as np
from typing import Dict, List, Tuple
import os
import sys

# Add the VRP utils path to sys.path
sys.path.append('../../src')
try:
    from utils import VRPDataReader
except ImportError:
    # If the above fails, try relative import
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
    from utils import VRPDataReader


class FacilityLocationDataExtractor:
    """Extract facility location problem data from VRP instances"""
    
    def __init__(self, vrp_file: str):
        """
        Initialize facility location data extractor
        
        Args:
            vrp_file: Path to VRP instance file
        """
        self.vrp_file = vrp_file
        self.vrp_reader = VRPDataReader(vrp_file)
        self.vrp_data = self.vrp_reader.parse()
        self.distance_matrix = self.vrp_reader.compute_distance_matrix()
        
    def extract_facility_location_data(self, exclude_depot: bool = True) -> Dict:
        """
        Extract facility location problem data
        
        Args:
            exclude_depot: Whether to exclude the depot from candidate locations and demand points
            
        Returns:
            Dictionary containing facility location problem data
        """
        # Extract coordinates and demands
        coordinates = {}
        demands = {}
        
        if exclude_depot:
            # Exclude depot (node 1), use only customers (nodes 2 and onwards)
            node_ids = list(range(2, self.vrp_data['dimension'] + 1))
        else:
            # Include all nodes (depot and customers)
            node_ids = list(range(1, self.vrp_data['dimension'] + 1))
        
        # Convert to 0-indexed for our problem
        for i, node_id in enumerate(node_ids):
            if node_id in self.vrp_data['node_coords']:
                coordinates[i] = self.vrp_data['node_coords'][node_id]
            
            if node_id in self.vrp_data['demands']:
                demands[i] = self.vrp_data['demands'][node_id]
            else:
                # If no demand specified, use default demand of 1
                demands[i] = 1
        
        # Compute distance matrix for our nodes
        n = len(node_ids)
        distance_matrix = np.zeros((n, n))
        
        for i, node_i in enumerate(node_ids):
            for j, node_j in enumerate(node_ids):
                # Convert from 1-indexed VRP to 0-indexed array
                vrp_i = node_i - 1
                vrp_j = node_j - 1
                distance_matrix[i][j] = self.distance_matrix[vrp_i][vrp_j]
        
        # Create facility location problem data
        fl_data = {
            'name': f"{self.vrp_data['name']}_FL{n}",
            'num_locations': n,
            'coordinates': coordinates,
            'demands': demands,
            'distance_matrix': distance_matrix,
            'original_vrp_nodes': node_ids,
            'exclude_depot': exclude_depot,
            'source_vrp_file': self.vrp_file,
            'total_demand': sum(demands.values())
        }
        
        return fl_data
    
    def get_problem_summary(self, fl_data: Dict) -> str:
        """Get a summary of the facility location problem"""
        summary = f"""
Facility Location Problem Summary
=================================
Instance: {fl_data['name']}
Source VRP file: {os.path.basename(fl_data['source_vrp_file'])}
Number of locations: {fl_data['num_locations']}
Total demand: {fl_data['total_demand']:.1f}
Exclude depot: {fl_data['exclude_depot']}
Original VRP nodes: {fl_data['original_vrp_nodes'][:5]}{'...' if len(fl_data['original_vrp_nodes']) > 5 else ''}

Distance matrix shape: {fl_data['distance_matrix'].shape}
Demand range: [{min(fl_data['demands'].values()):.1f}, {max(fl_data['demands'].values()):.1f}]

Coordinates sample:
{dict(list(fl_data['coordinates'].items())[:3])}
        """
        return summary.strip()


def load_facility_location_data(vrp_file: str, exclude_depot: bool = True) -> Dict:
    """
    Convenience function to load facility location data from VRP file
    
    Args:
        vrp_file: Path to VRP instance file
        exclude_depot: Whether to exclude the depot
        
    Returns:
        Facility location problem data
    """
    extractor = FacilityLocationDataExtractor(vrp_file)
    return extractor.extract_facility_location_data(exclude_depot=exclude_depot)


if __name__ == "__main__":
    # Example usage
    vrp_file = "../tai100a/data/tai100a.vrp"
    
    print("Loading facility location data from tai100a...")
    extractor = FacilityLocationDataExtractor(vrp_file)
    fl_data = extractor.extract_facility_location_data(exclude_depot=True)
    
    print(extractor.get_problem_summary(fl_data))
    
    print(f"\nFirst 5 demands: {dict(list(fl_data['demands'].items())[:5])}")
    print(f"Distance from location 0 to others: {fl_data['distance_matrix'][0][:5]}")