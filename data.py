import pandas as pd
from pyrosm import OSM

# Load the Bengaluru OSM data
osm = OSM("bengaluru.osm.pbf")

# Get drivable roads
roads = osm.get_network(network_type="driving")

# Drop missing geometries or lengths
roads = roads.dropna(subset=["geometry", "length"]).copy()

# Function to extract start and end coordinates
def extract_coords(geom):
    try:
        if geom.geom_type == "LineString":
            return geom.coords[0], geom.coords[-1]
        elif geom.geom_type == "MultiLineString":
            first_line = list(geom.geoms)[0]
            return first_line.coords[0], first_line.coords[-1]
    except:
        return None, None

# Apply the function to get coordinates
roads["start"], roads["end"] = zip(*roads["geometry"].map(extract_coords))
roads = roads.dropna(subset=["start", "end"])

# Optional: Clean and convert coordinates to readable format
roads["start"] = roads["start"].apply(lambda x: f"{x[0]},{x[1]}")
roads["end"] = roads["end"].apply(lambda x: f"{x[0]},{x[1]}")

# Save only useful columns
roads[["start", "end", "length", "name", "highway"]].to_csv("bengaluru_roads.csv", index=False)
