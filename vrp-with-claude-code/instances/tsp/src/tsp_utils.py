"""
TSP (Traveling Salesman Problem) utilities for extracting TSP instances from VRP data
"""

import numpy as np
from typing import Dict, List, Tuple
import os

try:
    from .utils import VRPDataReader
except ImportError:
    from utils import VRPDataReader


class TSPDataExtractor:
    """Extract TSP instances from VRP data"""
    
    def __init__(self, vrp_file: str):
        """
        Initialize TSP data extractor
        
        Args:
            vrp_file: Path to VRP instance file
        """
        self.vrp_file = vrp_file
        self.vrp_reader = VRPDataReader(vrp_file)
        self.vrp_data = self.vrp_reader.parse()
        self.distance_matrix = self.vrp_reader.compute_distance_matrix()
        
    def extract_tsp_subset(self, num_nodes: int = 15, include_depot: bool = True) -> Dict:
        """
        Extract a TSP instance with specified number of nodes
        
        Args:
            num_nodes: Number of nodes to include in TSP (including depot if include_depot=True)
            include_depot: Whether to include the depot node (node 1 in VRP file)
            
        Returns:
            Dictionary containing TSP instance data
        """
        if include_depot:
            if num_nodes > self.vrp_data['dimension']:
                num_nodes = self.vrp_data['dimension']
            
            # Include depot (node 1) and first (num_nodes-1) customers
            selected_nodes = [1] + list(range(2, num_nodes + 1))
        else:
            if num_nodes > self.vrp_data['dimension'] - 1:
                num_nodes = self.vrp_data['dimension'] - 1
            
            # Take first num_nodes customers (skip depot)
            selected_nodes = list(range(2, num_nodes + 2))
        
        # Extract coordinates
        tsp_coords = {}
        for i, node_id in enumerate(selected_nodes):
            if node_id in self.vrp_data['node_coords']:
                tsp_coords[i] = self.vrp_data['node_coords'][node_id]
        
        # Extract distance matrix
        tsp_distance_matrix = np.zeros((len(selected_nodes), len(selected_nodes)))
        for i, node_i in enumerate(selected_nodes):
            for j, node_j in enumerate(selected_nodes):
                # Convert from 1-indexed VRP to 0-indexed array
                vrp_i = node_i - 1
                vrp_j = node_j - 1
                tsp_distance_matrix[i][j] = self.distance_matrix[vrp_i][vrp_j]
        
        # Create TSP instance data
        tsp_data = {
            'name': f"{self.vrp_data['name']}_TSP{num_nodes}",
            'dimension': len(selected_nodes),
            'coordinates': tsp_coords,
            'distance_matrix': tsp_distance_matrix,
            'original_vrp_nodes': selected_nodes,
            'include_depot': include_depot,
            'source_vrp_file': self.vrp_file
        }
        
        return tsp_data
    
    def save_tsp_instance(self, tsp_data: Dict, output_file: str):
        """
        Save TSP instance in TSPLIB format
        
        Args:
            tsp_data: TSP instance data
            output_file: Output file path
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(f"NAME : {tsp_data['name']}\n")
            f.write(f"TYPE : TSP\n")
            f.write(f"COMMENT : TSP extracted from VRP instance\n")
            f.write(f"DIMENSION : {tsp_data['dimension']}\n")
            f.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
            f.write(f"NODE_COORD_SECTION\n")
            
            for i in range(tsp_data['dimension']):
                coord = tsp_data['coordinates'][i]
                f.write(f"{i+1} {coord[0]} {coord[1]}\n")
            
            f.write(f"EOF\n")
    
    def create_multiple_tsp_instances(self, sizes: List[int] = [10, 15, 20]) -> Dict[str, Dict]:
        """
        Create multiple TSP instances of different sizes
        
        Args:
            sizes: List of TSP sizes to create
            
        Returns:
            Dictionary mapping size to TSP instance data
        """
        tsp_instances = {}
        
        for size in sizes:
            if size <= self.vrp_data['dimension']:
                tsp_data = self.extract_tsp_subset(size, include_depot=True)
                tsp_instances[f"tsp{size}"] = tsp_data
            else:
                print(f"Warning: TSP size {size} exceeds VRP dimension {self.vrp_data['dimension']}")
        
        return tsp_instances


def load_tsp_instance(tsp_file: str) -> Dict:
    """
    Load TSP instance from file
    
    Args:
        tsp_file: Path to TSP instance file
        
    Returns:
        Dictionary containing TSP instance data
    """
    tsp_data = {
        'name': '',
        'dimension': 0,
        'coordinates': {},
        'distance_matrix': None
    }
    
    with open(tsp_file, 'r') as f:
        lines = f.readlines()
    
    reading_coords = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('NAME'):
            tsp_data['name'] = line.split(':')[1].strip()
        elif line.startswith('DIMENSION'):
            tsp_data['dimension'] = int(line.split(':')[1].strip())
        elif line.startswith('NODE_COORD_SECTION'):
            reading_coords = True
        elif line.startswith('EOF'):
            break
        elif reading_coords and line:
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0]) - 1  # Convert to 0-indexed
                x = float(parts[1])
                y = float(parts[2])
                tsp_data['coordinates'][node_id] = (x, y)
    
    # Compute distance matrix
    n = tsp_data['dimension']
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                coord_i = tsp_data['coordinates'][i]
                coord_j = tsp_data['coordinates'][j]
                distance = np.sqrt((coord_i[0] - coord_j[0])**2 + (coord_i[1] - coord_j[1])**2)
                distance_matrix[i][j] = round(distance)
    
    tsp_data['distance_matrix'] = distance_matrix
    
    return tsp_data


if __name__ == "__main__":
    # Example usage
    vrp_file = "instances/tai75a/data/tai75a.vrp"
    extractor = TSPDataExtractor(vrp_file)
    
    # Create TSP instances of different sizes
    tsp_instances = extractor.create_multiple_tsp_instances([10, 15, 20])
    
    for name, tsp_data in tsp_instances.items():
        print(f"Created {name}: {tsp_data['dimension']} nodes")
        print(f"  Distance matrix shape: {tsp_data['distance_matrix'].shape}")
        print(f"  Original VRP nodes: {tsp_data['original_vrp_nodes']}")
        print()