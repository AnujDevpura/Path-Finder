import osmnx as ox  # type: ignore
import networkx as nx
import matplotlib.pyplot as plt

print("Downloading road network...")

bbox = (77.58, 12.95, 77.615, 12.985)  # west, south, east, north


G = ox.graph_from_bbox(bbox, network_type='drive')

# G = ox.simplify_graph(G)

# Keep largest strongly connected component
G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()

ox.save_graphml(G, "central_bengaluru.graphml")
ox.save_graph_geopackage(G, "central_bengaluru.gpkg")

ox.plot_graph(G, node_size=5, edge_linewidth=0.5)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())