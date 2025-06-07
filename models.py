from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pydantic import BaseModel, validator
import math
    
class Node:
    def __init__(self, id: int, lat: float, lon: float, name: str = None):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.name = name  # optional

class Edge:
    def __init__(self, from_node: int, to_node: int, weight: float, name: str = None):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.name = name  # road name if available


class Graph:
    """Graph representation using adjacency list"""
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, List[Edge]] = {}
        
    def add_node(self, node: Node):
        self.nodes[node.id] = node
        if node.id not in self.edges:
            self.edges[node.id] = []
            
    def add_edge(self, edge: Edge):
        if edge.from_node not in self.edges:
            self.edges[edge.from_node] = []
        self.edges[edge.from_node].append(edge)
        
    def get_neighbors(self, node_id: int) -> List[Edge]:
        return self.edges.get(node_id, [])
        
    def haversine_distance(self, node1_id: int, node2_id: int) -> float:
        """Calculate Haversine distance between two nodes"""
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        
        R = 6371000  # Earth radius in meters
        lat1, lon1 = math.radians(node1.lat), math.radians(node1.lon)
        lat2, lon2 = math.radians(node2.lat), math.radians(node2.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
        
    def euclidean_distance(self, node1_id: int, node2_id: int) -> float:
        """Calculate Euclidean distance between two nodes (approximation)"""
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        
        # Convert to approximate meters (rough conversion)
        lat_diff = (node2.lat - node1.lat) * 111320
        lon_diff = (node2.lon - node1.lon) * 40075000 * math.cos(math.radians((node1.lat + node2.lat) / 2)) / 360
        
        return math.sqrt(lat_diff**2 + lon_diff**2)

class RouteRequest(BaseModel):
    """API request model for route finding"""
    src_lat: float
    src_lon: float
    dst_lat: float
    dst_lon: float
    algo: str
    
    @validator('algo')
    def validate_algorithm(cls, v):
        valid_algos = ['dijkstra', 'dial', 'astar', 'bidirectional_astar']
        if v not in valid_algos:
            raise ValueError(f'Algorithm must be one of {valid_algos}')
        return v

@dataclass
class RouteResult:
    """Result of pathfinding algorithm"""
    path: List[int]
    distance: float
    nodes_expanded: int
    runtime_ms: float
    algorithm: str

@dataclass
class BenchmarkResult:
    """Result of algorithm benchmark"""
    algorithm: str
    avg_runtime_ms: float
    std_runtime_ms: float
    avg_nodes_expanded: float
    std_nodes_expanded: float
    avg_path_length: float
    success_rate: float