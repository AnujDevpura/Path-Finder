import xml.etree.ElementTree as ET
from models import Graph, Node, Edge

def build_graph_from_graphml(filename: str) -> Graph:
    """
    Parses a GraphML file and returns a Graph object.
    Assumes nodes have 'id', 'lat', 'lon' (or similar) and edges have 'source', 'target', and 'weight'.
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}

    graph = Graph()
    node_map = {}

    # Parse nodes
    for node in root.findall('.//graphml:node', ns):
        node_id = node.attrib['id']
        lat = None
        lon = None
        name = None
        for data in node.findall('graphml:data', ns):
            key = data.attrib.get('key', '')
            if key.lower() in ['lat', 'latitude', 'd4']:
                lat = float(data.text)
            elif key.lower() in ['lon', 'longitude', 'd5']:
                lon = float(data.text)
            elif key.lower() in ['name', 'label', 'd12']:
                name = data.text
        if lat is not None and lon is not None:
            n = Node(int(node_id), lat, lon, name)
            graph.add_node(n)
            node_map[node_id] = n

    # Parse edges
    for edge in root.findall('.//graphml:edge', ns):
        from_node = edge.attrib['source']
        to_node = edge.attrib['target']
        weight = None
        name = None
        for data in edge.findall('graphml:data', ns):
            key = data.attrib.get('key', '')
            if key.lower() in ['weight', 'length', 'd15']:
                try:
                    weight = float(data.text)
                except (TypeError, ValueError):
                    weight = 1.0
            elif key.lower() in ['name', 'label', 'd12']:
                name = data.text
        if weight is None:
            weight = 1.0  # Default weight if not specified
        graph.add_edge(Edge(int(from_node), int(to_node), weight, name))
        # If undirected, also add reverse edge:
        # graph.add_edge(Edge(int(to_node), int(from_node), weight, name))
    return graph