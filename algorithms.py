import heapq
import time
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Set
from models import Graph, RouteResult
import math

class PathFinder:
    """Base class for pathfinding algorithms"""
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.nodes_expanded = 0
        
    def find_path(self, start: int, end: int) -> RouteResult:
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError
        
    def reconstruct_path(self, came_from: Dict[int, int], current: int) -> List[int]:
        """Reconstruct path from came_from dictionary"""
        path = []
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        return path[::-1]

class DijkstraPathFinder(PathFinder):
    """Dijkstra's algorithm implementation with binary heap"""
    
    def find_path(self, start: int, end: int) -> RouteResult:
        import time
        start_time = time.time()
        self.nodes_expanded = 0
        
        distances = {node_id: float('inf') for node_id in self.graph.nodes}
        distances[start] = 0
        came_from = {}
        
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            self.nodes_expanded += 1
            
            if current == end:
                path = self.reconstruct_path(came_from, current)
                runtime = (time.time() - start_time) * 1000
                return RouteResult(path, distances[end], self.nodes_expanded, runtime, 'dijkstra')
            
            for edge in self.graph.get_neighbors(current):
                neighbor = edge.to_node
                new_dist = current_dist + edge.weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    came_from[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        runtime = (time.time() - start_time) * 1000
        return RouteResult([], float('inf'), self.nodes_expanded, runtime, 'dijkstra')

class DialPathFinder(PathFinder):
    """Dial's algorithm implementation with bucket queues"""
    
    def __init__(self, graph: Graph, bucket_size: int = 100):
        super().__init__(graph)
        self.bucket_size = bucket_size  # bucket size in meters
        
    def find_path(self, start: int, end: int) -> RouteResult:
        start_time = time.time()
        self.nodes_expanded = 0
        
        distances = {node_id: float('inf') for node_id in self.graph.nodes}
        distances[start] = 0
        came_from = {}
        
        # Initialize buckets
        max_weight = max((edge.weight for edges in self.graph.edges.values() for edge in edges), default=1000)
        num_buckets = int(max_weight / self.bucket_size) + 1
        buckets = [deque() for _ in range(num_buckets)]
        
        buckets[0].append(start)
        current_bucket = 0
        
        while current_bucket < num_buckets:
            while buckets[current_bucket]:
                current = buckets[current_bucket].popleft()
                
                if distances[current] == float('inf'):
                    continue
                    
                self.nodes_expanded += 1
                
                if current == end:
                    path = self.reconstruct_path(came_from, current)
                    runtime = (time.time() - start_time) * 1000
                    return RouteResult(path, distances[end], self.nodes_expanded, runtime, 'dial')
                
                for edge in self.graph.get_neighbors(current):
                    neighbor = edge.to_node
                    new_dist = distances[current] + edge.weight
                    
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        came_from[neighbor] = current
                        
                        bucket_idx = min(int(new_dist / self.bucket_size), num_buckets - 1)
                        buckets[bucket_idx].append(neighbor)
            
            current_bucket += 1
        
        runtime = (time.time() - start_time) * 1000
        return RouteResult([], float('inf'), self.nodes_expanded, runtime, 'dial')

class AStarPathFinder(PathFinder):
    """A* algorithm implementation with heuristic"""
    
    def __init__(self, graph: Graph, heuristic: str = 'haversine'):
        super().__init__(graph)
        self.heuristic = heuristic
        
    def heuristic_distance(self, node1: int, node2: int) -> float:
        """Calculate heuristic distance between two nodes"""
        if self.heuristic == 'haversine':
            return self.graph.haversine_distance(node1, node2)
        else:
            return self.graph.euclidean_distance(node1, node2)
    
    def find_path(self, start: int, end: int) -> RouteResult:
        start_time = time.time()
        self.nodes_expanded = 0
        
        g_score = {node_id: float('inf') for node_id in self.graph.nodes}
        g_score[start] = 0
        
        f_score = {node_id: float('inf') for node_id in self.graph.nodes}
        f_score[start] = self.heuristic_distance(start, end)
        
        came_from = {}
        open_set = [(f_score[start], start)]
        closed_set = set()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            self.nodes_expanded += 1
            
            if current == end:
                path = self.reconstruct_path(came_from, current)
                runtime = (time.time() - start_time) * 1000
                return RouteResult(path, g_score[end], self.nodes_expanded, runtime, 'astar')
            
            for edge in self.graph.get_neighbors(current):
                neighbor = edge.to_node
                
                if neighbor in closed_set:
                    continue
                    
                tentative_g = g_score[current] + edge.weight
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic_distance(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        runtime = (time.time() - start_time) * 1000
        return RouteResult([], float('inf'), self.nodes_expanded, runtime, 'astar')

class BidirectionalAStarPathFinder(PathFinder):
    """Bidirectional A* algorithm implementation"""

    _reverse_edges_cache = {}

    def __init__(self, graph: Graph, heuristic: str = 'haversine'):
        super().__init__(graph)
        self.heuristic = heuristic
        # Build reverse edges once per graph instance
        if id(graph) not in BidirectionalAStarPathFinder._reverse_edges_cache:
            reverse_edges = defaultdict(list)
            for node_id, edges in graph.edges.items():
                for edge in edges:
                    reverse_edges[edge.to_node].append((node_id, edge.weight))
            BidirectionalAStarPathFinder._reverse_edges_cache[id(graph)] = reverse_edges
        self.reverse_edges = BidirectionalAStarPathFinder._reverse_edges_cache[id(graph)]

    def heuristic_distance(self, node1: int, node2: int) -> float:
        if self.heuristic == 'haversine':
            return self.graph.haversine_distance(node1, node2)
        else:
            return self.graph.euclidean_distance(node1, node2)

    def find_path(self, start: int, end: int) -> RouteResult:
        start_time = time.time()
        self.nodes_expanded = 0

        g_forward = {node_id: float('inf') for node_id in self.graph.nodes}
        g_forward[start] = 0
        f_forward = {start: self.heuristic_distance(start, end)}
        came_from_forward = {}
        open_forward = [(f_forward[start], start)]
        closed_forward = set()

        g_backward = {node_id: float('inf') for node_id in self.graph.nodes}
        g_backward[end] = 0
        f_backward = {end: self.heuristic_distance(end, start)}
        came_from_backward = {}
        open_backward = [(f_backward[end], end)]
        closed_backward = set()

        best_path_length = float('inf')
        meeting_point = None

        expanded_nodes = set()  # To count unique expansions

        while open_forward and open_backward:
            # Forward step
            if open_forward:
                f_curr, current_forward = heapq.heappop(open_forward)
                if current_forward in closed_forward:
                    continue
                closed_forward.add(current_forward)
                expanded_nodes.add(current_forward)

                if current_forward in closed_backward:
                    total_dist = g_forward[current_forward] + g_backward[current_forward]
                    if total_dist < best_path_length:
                        best_path_length = total_dist
                        meeting_point = current_forward

                for edge in self.graph.get_neighbors(current_forward):
                    neighbor = edge.to_node
                    tentative_g = g_forward[current_forward] + edge.weight
                    if tentative_g < g_forward[neighbor]:
                        came_from_forward[neighbor] = current_forward
                        g_forward[neighbor] = tentative_g
                        f_score = tentative_g + self.heuristic_distance(neighbor, end)
                        heapq.heappush(open_forward, (f_score, neighbor))

            # Backward step
            if open_backward:
                f_curr, current_backward = heapq.heappop(open_backward)
                if current_backward in closed_backward:
                    continue
                closed_backward.add(current_backward)
                expanded_nodes.add(current_backward)

                if current_backward in closed_forward:
                    total_dist = g_forward[current_backward] + g_backward[current_backward]
                    if total_dist < best_path_length:
                        best_path_length = total_dist
                        meeting_point = current_backward

                for prev_node, weight in self.reverse_edges[current_backward]:
                    tentative_g = g_backward[current_backward] + weight
                    if tentative_g < g_backward[prev_node]:
                        came_from_backward[prev_node] = current_backward
                        g_backward[prev_node] = tentative_g
                        f_score = tentative_g + self.heuristic_distance(prev_node, start)
                        heapq.heappush(open_backward, (f_score, prev_node))

            # Improved early termination
            if open_forward and open_backward:
                min_f = min(open_forward[0][0], open_backward[0][0])
                if best_path_length <= min_f:
                    break

        runtime = (time.time() - start_time) * 1000
        self.nodes_expanded = len(expanded_nodes)

        if meeting_point:
            forward_path = self.reconstruct_path(came_from_forward, meeting_point)
            backward_path = self.reconstruct_path(came_from_backward, meeting_point)
            backward_path.reverse()
            full_path = forward_path + backward_path[1:]
            return RouteResult(full_path, best_path_length, self.nodes_expanded, runtime, 'bidirectional_astar')

        return RouteResult([], float('inf'), self.nodes_expanded, runtime, 'bidirectional_astar')