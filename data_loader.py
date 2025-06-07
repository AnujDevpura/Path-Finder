import json
import random
import math
from typing import List, Tuple
from models import Graph, Node, Edge
from build_graph import build_graph_from_graphml

class DataLoader:
    """Loads and generates graph data for testing"""
    
    @staticmethod
    def generate_test_graph(num_nodes: int = 1000, city_bounds: Tuple[float, float, float, float] = None) -> Graph:
        """
        Generate a test graph with random nodes and edges
        city_bounds: (min_lat, max_lat, min_lon, max_lon)
        """
        if city_bounds is None:
            # Default bounds (roughly San Francisco area)
            city_bounds = (37.7049, 37.8049, -122.5149, -122.3849)
        
        min_lat, max_lat, min_lon, max_lon = city_bounds
        
        graph = Graph()
        
        # Generate random nodes
        for i in range(num_nodes):
            lat = random.uniform(min_lat, max_lat)
            lon = random.uniform(min_lon, max_lon)
            node = Node(i, lat, lon)
            graph.add_node(node)
        
        # Generate edges (connect each node to nearby nodes)
        for node_id in range(num_nodes):
            node = graph.nodes[node_id]
            
            # Find nearby nodes and connect to some of them
            nearby_nodes = []
            for other_id, other_node in graph.nodes.items():
                if other_id != node_id:
                    distance = graph.haversine_distance(node_id, other_id)
                    if distance < 2000:  # Within 2km
                        nearby_nodes.append((other_id, distance))
            
            # Sort by distance and connect to closest 3-8 nodes
            nearby_nodes.sort(key=lambda x: x[1])
            num_connections = min(random.randint(3, 8), len(nearby_nodes))
            
            for i in range(num_connections):
                other_id, distance = nearby_nodes[i]
                edge = Edge(node_id, other_id, distance)
                graph.add_edge(edge)
        
        return graph
    
    @staticmethod
    def load_osm_data(filename: str) -> Graph:
        """
        Load graph data from OSM JSON file
        Expected format: {"nodes": [...], "edges": [...]}
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            graph = Graph()
            
            # Load nodes
            for node_data in data.get('nodes', []):
                node = Node(
                    id=node_data['id'],
                    lat=node_data['lat'],
                    lon=node_data['lon']
                )
                graph.add_node(node)
            
            # Load edges
            for edge_data in data.get('edges', []):
                edge = Edge(
                    from_node=edge_data['from'],
                    to_node=edge_data['to'],
                    weight=edge_data['weight']
                )
                graph.add_edge(edge)
            
            return graph
            
        except FileNotFoundError:
            print(f"File {filename} not found. Generating test data...")
            return DataLoader.generate_test_graph()
        except Exception as e:
            print(f"Error loading data: {e}. Generating test data...")
            return DataLoader.generate_test_graph()
    
    @staticmethod
    def find_nearest_node(graph: Graph, lat: float, lon: float) -> int:
        """Find the nearest node to given coordinates"""
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, node in graph.nodes.items():
            # Simple Euclidean distance approximation
            lat_diff = (node.lat - lat) * 111320  # meters per degree lat
            lon_diff = (node.lon - lon) * 40075000 * math.cos(math.radians((node.lat + lat) / 2)) / 360
            distance = math.sqrt(lat_diff**2 + lon_diff**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node
    
    @staticmethod
    def save_graph(graph: Graph, filename: str):
        """Save graph to JSON file"""
        data = {
            'nodes': [
                {'id': node.id, 'lat': node.lat, 'lon': node.lon}
                for node in graph.nodes.values()
            ],
            'edges': [
                {'from': edge.from_node, 'to': edge.to_node, 'weight': edge.weight}
                for edges in graph.edges.values()
                for edge in edges
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_graph(graphml_fallback: str = "central_bengaluru.graphml") -> Graph:
        import os
        if os.path.exists(graphml_fallback):
            print(f"Loading graph from GraphML: {graphml_fallback}")
            return build_graph_from_graphml(graphml_fallback)
        else:
            print("No GraphML file found. Generating test data...")
            return DataLoader.generate_test_graph()