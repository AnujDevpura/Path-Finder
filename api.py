from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any

from models import RouteRequest, RouteResult, Graph
from algorithms import DijkstraPathFinder, DialPathFinder, AStarPathFinder, BidirectionalAStarPathFinder
from data_loader import DataLoader
from utils import get_largest_connected_component

app = FastAPI(title="City-Scale Path Finder", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global graph instance
graph: Graph = None

class RouteResponse(BaseModel):
    path: List[Dict[str, Any]]
    distance: float
    nodes_expanded: int
    runtime_ms: float
    algorithm: str
    success: bool

@app.on_event("startup")
async def startup_event():
    global graph
    print("Loading graph from GraphML...")
    graph = DataLoader.load_graph(graphml_fallback="central_bengaluru.graphml")
    graph = get_largest_connected_component(graph)
    print(f"Loaded largest connected component with {len(graph.nodes)} nodes and {sum(len(edges) for edges in graph.edges.values())} edges")

@app.get("/")
async def root():
    return {"message": "City-Scale Path Finder API"}

@app.get("/stats")
async def get_stats():
    """Get graph statistics"""
    if not graph:
        raise HTTPException(status_code=500, detail="Graph not initialized")
    total_edges = sum(len(edges) for edges in graph.edges.values())
    return {
        "nodes": len(graph.nodes),
        "edges": total_edges,
        "algorithms": ["dijkstra", "dial", "astar", "bidirectional_astar"]
    }

@app.post("/route", response_model=RouteResponse)
async def find_route(request: RouteRequest):
    try:
        if not graph:
            raise HTTPException(status_code=503, detail="Graph not loaded")

        # Find nearest nodes
        src_id = DataLoader.find_nearest_node(graph, request.src_lat, request.src_lon)
        dst_id = DataLoader.find_nearest_node(graph, request.dst_lat, request.dst_lon)

        # Select algorithm
        if request.algo == "dijkstra":
            finder = DijkstraPathFinder(graph)
        elif request.algo == "dial":
            finder = DialPathFinder(graph)
        elif request.algo == "astar":
            finder = AStarPathFinder(graph)
        elif request.algo == "bidirectional_astar":
            finder = BidirectionalAStarPathFinder(graph)
        else:
            raise HTTPException(status_code=400, detail="Unknown algorithm")

        result = finder.find_path(src_id, dst_id)

        # Fix: Replace inf with -1.0 and mark as unsuccessful
        if not result.path or result.distance == float('inf'):
            return RouteResponse(
                path=[],
                distance=-1.0,
                nodes_expanded=result.nodes_expanded,
                runtime_ms=result.runtime_ms,
                algorithm=request.algo,
                success=False
            )

        # Otherwise, build path details
        path_nodes = [
            {
                "id": node_id,
                "lat": graph.nodes[node_id].lat,
                "lon": graph.nodes[node_id].lon,
                "name": getattr(graph.nodes[node_id], "name", None)
            }
            for node_id in result.path
        ]

        return RouteResponse(
            path=path_nodes,
            distance=result.distance,
            nodes_expanded=result.nodes_expanded,
            runtime_ms=result.runtime_ms,
            algorithm=request.algo,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/benchmark")
async def benchmark_algorithms(num_tests: int = 10):
    """Benchmark all algorithms with random start/end points (only connected pairs)"""
    if not graph:
        raise HTTPException(status_code=500, detail="Graph not initialized")
    import random
    from statistics import mean, stdev

    algorithms = {
        "dijkstra": DijkstraPathFinder(graph),
        "dial": DialPathFinder(graph),
        "astar": AStarPathFinder(graph),
        "bidirectional_astar": BidirectionalAStarPathFinder(graph)
    }

    results = {}
    node_ids = list(graph.nodes.keys())
    test_cases = []

    # Only use pairs with a valid path for Dijkstra (guaranteed to find if exists)
    attempts = 0
    while len(test_cases) < num_tests and attempts < num_tests * 20:
        start = random.choice(node_ids)
        end = random.choice(node_ids)
        if end == start:
            continue
        # Quick check: Only add if Dijkstra finds a path
        result = DijkstraPathFinder(graph).find_path(start, end)
        if result.path and result.distance != float('inf'):
            test_cases.append((start, end))
        attempts += 1

    if not test_cases:
        return {algo: {
            "avg_runtime_ms": 0,
            "std_runtime_ms": 0,
            "avg_nodes_expanded": 0,
            "std_nodes_expanded": 0,
            "avg_path_length": 0,
            "success_rate": 0,
            "total_tests": num_tests
        } for algo in algorithms}

    for algo_name, pathfinder in algorithms.items():
        runtimes = []
        nodes_expanded = []
        path_lengths = []
        successes = 0

        for start, end in test_cases:
            result = pathfinder.find_path(start, end)
            if result.path and result.distance != float('inf'):
                successes += 1
                runtimes.append(result.runtime_ms)
                nodes_expanded.append(result.nodes_expanded)
                path_lengths.append(result.distance)

        if runtimes:
            results[algo_name] = {
                "avg_runtime_ms": mean(runtimes),
                "std_runtime_ms": stdev(runtimes) if len(runtimes) > 1 else 0,
                "avg_nodes_expanded": mean(nodes_expanded),
                "std_nodes_expanded": stdev(nodes_expanded) if len(nodes_expanded) > 1 else 0,
                "avg_path_length": mean(path_lengths),
                "success_rate": successes / num_tests,
                "total_tests": num_tests
            }
        else:
            results[algo_name] = {
                "avg_runtime_ms": 0,
                "std_runtime_ms": 0,
                "avg_nodes_expanded": 0,
                "std_nodes_expanded": 0,
                "avg_path_length": 0,
                "success_rate": 0,
                "total_tests": num_tests
            }

    return results

@app.get("/nodes")
async def get_nodes():
    """Return all node IDs and their coordinates."""
    if not graph:
        raise HTTPException(status_code=500, detail="Graph not initialized")
    return [
        {"id": node.id, "lat": node.lat, "lon": node.lon, "name": getattr(node, "name", None)}
        for node in graph.nodes.values()
    ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)