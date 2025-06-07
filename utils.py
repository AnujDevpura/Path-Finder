from models import RouteRequest, RouteResult, Graph

def get_largest_connected_component(graph):
    from collections import deque

    visited = set()
    largest_cc = set()

    for node_id in graph.nodes:
        if node_id in visited:
            continue
        # BFS to find all nodes in this component
        queue = deque([node_id])
        component = set()
        while queue:
            n = queue.popleft()
            if n in component:
                continue
            component.add(n)
            for edge in graph.edges.get(n, []):
                if edge.to_node not in component:
                    queue.append(edge.to_node)
        if len(component) > len(largest_cc):
            largest_cc = component
        visited.update(component)
    # Build new graph with only largest_cc nodes and edges
    new_graph = Graph()
    for nid in largest_cc:
        new_graph.nodes[nid] = graph.nodes[nid]
        new_graph.edges[nid] = [e for e in graph.edges[nid] if e.to_node in largest_cc]
    return new_graph