#!/usr/bin/env python3
"""
CLI Benchmark Tool for City-Scale Path Finder
Run headless benchmarks and generate performance reports
"""

import argparse
import json
import time
import random
import statistics
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd

from models import Graph, BenchmarkResult
from algorithms import DijkstraPathFinder, DialPathFinder, AStarPathFinder, BidirectionalAStarPathFinder
from data_loader import DataLoader

class BenchmarkRunner:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.algorithms = {
            'dijkstra': DijkstraPathFinder(graph),
            'dial': DialPathFinder(graph),
            'astar': AStarPathFinder(graph),
            'bidirectional_astar': BidirectionalAStarPathFinder(graph)
        }
    
    def run_benchmark(self, num_tests: int = 50, distance_ranges: List[Tuple[float, float]] = None) -> Dict[str, BenchmarkResult]:
        """Run comprehensive benchmark across all algorithms"""
        
        if distance_ranges is None:
            distance_ranges = [(0, 1000), (1000, 5000), (5000, float('inf'))]  # Short, medium, long distances
        
        print(f"Running benchmark with {num_tests} tests per algorithm...")
        print(f"Graph size: {len(self.graph.nodes)} nodes, {sum(len(edges) for edges in self.graph.edges.values())} edges")
        
        results = {}
        node_ids = list(self.graph.nodes.keys())
        
        for algo_name, pathfinder in self.algorithms.items():
            print(f"\nTesting {algo_name}...")
            
            runtimes = []
            nodes_expanded = []
            path_lengths = []
            successes = 0
            
            for i in range(num_tests):
                # Generate random test case
                start = random.choice(node_ids)
                end = random.choice(node_ids)
                while end == start:
                    end = random.choice(node_ids)
                
                # Run algorithm
                result = pathfinder.find_path(start, end)
                
                if len(result.path) > 0:
                    successes += 1
                    runtimes.append(result.runtime_ms)
                    nodes_expanded.append(result.nodes_expanded)
                    path_lengths.append(result.distance)
                
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{num_tests} tests")
            
            # Calculate statistics
            if runtimes:
                results[algo_name] = BenchmarkResult(
                    algorithm=algo_name,
                    avg_runtime_ms=statistics.mean(runtimes),
                    std_runtime_ms=statistics.stdev(runtimes) if len(runtimes) > 1 else 0,
                    avg_nodes_expanded=statistics.mean(nodes_expanded),
                    std_nodes_expanded=statistics.stdev(nodes_expanded) if len(nodes_expanded) > 1 else 0,
                    avg_path_length=statistics.mean(path_lengths),
                    success_rate=successes / num_tests
                )
            else:
                results[algo_name] = BenchmarkResult(
                    algorithm=algo_name,
                    avg_runtime_ms=0, std_runtime_ms=0,
                    avg_nodes_expanded=0, std_nodes_expanded=0,
                    avg_path_length=0, success_rate=0
                )
        
        return results
    
    def run_scalability_test(self, graph_sizes: List[int] = None) -> Dict[str, List[float]]:
        """Test algorithm performance across different graph sizes"""
        
        if graph_sizes is None:
            graph_sizes = [100, 250, 500, 1000, 2000]
        
        print(f"\nRunning scalability test with graph sizes: {graph_sizes}")
        
        scalability_results = {algo: [] for algo in self.algorithms.keys()}
        
        for size in graph_sizes:
            print(f"\nTesting with {size} nodes...")
            
            # Generate test graph of specific size
            test_graph = DataLoader.generate_test_graph(size)
            
            # Create pathfinders for test graph
            test_algorithms = {
                'dijkstra': DijkstraPathFinder(test_graph),
                'dial': DialPathFinder(test_graph),
                'astar': AStarPathFinder(test_graph),
                'bidirectional_astar': BidirectionalAStarPathFinder(test_graph)
            }
            
            # Run 10 tests per size
            for algo_name, pathfinder in test_algorithms.items():
                runtimes = []
                node_ids = list(test_graph.nodes.keys())
                
                for _ in range(10):
                    start = random.choice(node_ids)
                    end = random.choice(node_ids)
                    while end == start:
                        end = random.choice(node_ids)
                    
                    result = pathfinder.find_path(start, end)
                    if len(result.path) > 0:
                        runtimes.append(result.runtime_ms)
                
                avg_runtime = statistics.mean(runtimes) if runtimes else 0
                scalability_results[algo_name].append(avg_runtime)
        
        return scalability_results, graph_sizes
    
    def generate_report(self, results: Dict[str, BenchmarkResult], output_file: str = "benchmark_report.json"):
        """Generate detailed benchmark report"""
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "graph_stats": {
                "nodes": len(self.graph.nodes),
                "edges": sum(len(edges) for edges in self.graph.edges.values())
            },
            "results": {}
        }
        
        for algo_name, result in results.items():
            report["results"][algo_name] = {
                "avg_runtime_ms": result.avg_runtime_ms,
                "std_runtime_ms": result.std_runtime_ms,
                "avg_nodes_expanded": result.avg_nodes_expanded,
                "std_nodes_expanded": result.std_nodes_expanded,
                "avg_path_length": result.avg_path_length,
                "success_rate": result.success_rate
            }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nBenchmark report saved to {output_file}")
        
        # Print summary table
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        print(f"{'Algorithm':<20} {'Runtime (ms)':<15} {'Nodes Exp.':<12} {'Success %':<10} {'Path Len (m)':<12}")
        print("-"*80)
        
        for algo_name, result in results.items():
            print(f"{algo_name:<20} {result.avg_runtime_ms:<15.2f} {result.avg_nodes_expanded:<12.0f} {result.success_rate*100:<10.1f} {result.avg_path_length:<12.0f}")
    
    def generate_plots(self, results: Dict[str, BenchmarkResult], scalability_data=None):
        """Generate performance plots"""
        
        # Performance comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        algorithms = list(results.keys())
        runtimes = [results[algo].avg_runtime_ms for algo in algorithms]
        runtime_stds = [results[algo].std_runtime_ms for algo in algorithms]
        nodes_expanded = [results[algo].avg_nodes_expanded for algo in algorithms]
        success_rates = [results[algo].success_rate * 100 for algo in algorithms]
        
        # Runtime comparison
        ax1.bar(algorithms, runtimes, yerr=runtime_stds, capsize=5)
        ax1.set_title('Average Runtime Comparison')
        ax1.set_ylabel('Runtime (ms)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Nodes expanded comparison
        ax2.bar(algorithms, nodes_expanded)
        ax2.set_title('Average Nodes Expanded')
        ax2.set_ylabel('Nodes Expanded')
        ax2.tick_params(axis='x', rotation=45)
        
        # Success rate comparison
        ax3.bar(algorithms, success_rates)
        ax3.set_title('Success Rate')
        ax3.set_ylabel('Success Rate (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Scalability plot
        if scalability_data:
            scalability_results, graph_sizes = scalability_data
            for algo, times in scalability_results.items():
                ax4.plot(graph_sizes, times, marker='o', label=algo)
            ax4.set_title('Scalability Test')
            ax4.set_xlabel('Graph Size (nodes)')
            ax4.set_ylabel('Runtime (ms)')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Scalability data not available', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        print("Performance plots saved to benchmark_results.png")

def main():
    parser = argparse.ArgumentParser(description='CLI Benchmark Tool for City-Scale Path Finder')
    parser.add_argument('--tests', type=int, default=50, help='Number of test cases per algorithm')
    parser.add_argument('--data', type=str, default='city_data.json', help='Path to graph data file')
    parser.add_argument('--output', type=str, default='benchmark_report.json', help='Output report file')
    parser.add_argument('--scalability', action='store_true', help='Run scalability tests')
    parser.add_argument('--generate-plots', action='store_true', help='Generate performance plots')
    
    args = parser.parse_args()
    
    print("City-Scale Path Finder - CLI Benchmark Tool")
    print("="*50)
    
    # Load graph data
    print(f"Loading graph data from {args.data}...")
    graph = DataLoader.load_osm_data(args.data)
    print(f"Loaded graph with {len(graph.nodes)} nodes")
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(graph)
    
    # Run main benchmark
    results = runner.run_benchmark(num_tests=args.tests)
    
    # Run scalability test if requested
    scalability_data = None
    if args.scalability:
        scalability_data = runner.run_scalability_test()
    
    # Generate report
    runner.generate_report(results, args.output)
    
    # Generate plots if requested
    if args.generate_plots:
        runner.generate_plots(results, scalability_data)

if __name__ == "__main__":
    main()