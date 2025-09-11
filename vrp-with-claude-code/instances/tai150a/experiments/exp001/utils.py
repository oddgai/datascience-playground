import numpy as np
from typing import Dict, List, Tuple
import re
from decimal import Decimal, ROUND_HALF_UP


class VRPDataReader:
    def __init__(self, vrp_file_path: str):
        self.vrp_file_path = vrp_file_path
        self.name = None
        self.comment = None
        self.dimension = None
        self.capacity = None
        self.edge_weight_type = None
        self.node_coords = {}
        self.demands = {}
        
    def parse(self) -> Dict:
        with open(self.vrp_file_path, 'r') as f:
            lines = f.readlines()
        
        section = None
        for line in lines:
            line = line.strip()
            
            if not line or line == 'EOF':
                continue
                
            if line.startswith('NAME'):
                self.name = line.split(':')[1].strip()
            elif line.startswith('COMMENT'):
                self.comment = line.split(':', 1)[1].strip()
            elif line.startswith('TYPE'):
                continue
            elif line.startswith('DIMENSION'):
                self.dimension = int(line.split(':')[1].strip())
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                self.edge_weight_type = line.split(':')[1].strip()
            elif line.startswith('CAPACITY'):
                self.capacity = int(line.split(':')[1].strip())
            elif line == 'NODE_COORD_SECTION':
                section = 'COORDS'
            elif line == 'DEMAND_SECTION':
                section = 'DEMAND'
            elif section == 'COORDS':
                parts = line.split()
                if len(parts) == 3:
                    node_id = int(parts[0])
                    x, y = float(parts[1]), float(parts[2])
                    self.node_coords[node_id] = (x, y)
            elif section == 'DEMAND':
                parts = line.split()
                if len(parts) == 2:
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    self.demands[node_id] = demand
        
        return self.get_data()
    
    def get_data(self) -> Dict:
        return {
            'name': self.name,
            'comment': self.comment,
            'dimension': self.dimension,
            'capacity': self.capacity,
            'edge_weight_type': self.edge_weight_type,
            'node_coords': self.node_coords,
            'demands': self.demands,
            'num_vehicles': self._extract_num_vehicles(),
            'optimal_value': self._extract_optimal_value()
        }
    
    def _extract_num_vehicles(self) -> int:
        if self.comment:
            match = re.search(r'No of trucks:\s*(\d+)', self.comment)
            if match:
                return int(match.group(1))
        return 4
    
    def _extract_optimal_value(self) -> float:
        if self.comment:
            match = re.search(r'Optimal value:\s*([\d.]+)', self.comment)
            if match:
                return float(match.group(1))
        return None
    
    def compute_distance_matrix(self) -> np.ndarray:
        n = self.dimension
        distance_matrix = np.zeros((n, n))
        
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i != j:
                    x1, y1 = self.node_coords[i]
                    x2, y2 = self.node_coords[j]
                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    # Decimal.quantize()による数学的な四捨五入
                    distance_matrix[i-1][j-1] = int(Decimal(str(distance)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        
        return distance_matrix


def read_solution(sol_file_path: str) -> Tuple[List[List[int]], float]:
    routes = []
    cost = None
    
    with open(sol_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('Route'):
            route_str = line.split(':')[1].strip()
            route = [int(x) for x in route_str.split()]
            routes.append(route)
        elif line.startswith('Cost'):
            cost = float(line.split()[1])
    
    return routes, cost