import osmnx as ox
from pathlib import Path

output_path = Path("data/processed/road_segments.geojson")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Guangzhou metro area (smaller bbox to avoid Overpass timeout)
north = 23.4
south = 22.9
west = 113.0
east = 113.5

print(f"Region: {south}-{north}°N, {west}-{east}°E")
print("Downloading road network from OSM (this may take a few minutes)...")

# osmnx v2.x expects bbox as (west, south, east, north) = (left, bottom, right, top)
bbox = (west, south, east, north)

G = ox.graph_from_bbox(bbox=bbox, network_type='drive')

print(f"Graph downloaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

edges = ox.graph_to_gdfs(G, nodes=False)

# Keep useful columns (filter only those that exist)
keep_cols = ['highway', 'name', 'maxspeed', 'length', 'geometry']
keep_cols = [c for c in keep_cols if c in edges.columns]
edges = edges[keep_cols].copy()

# Convert list-type columns to strings (OGR can't serialize lists to GeoJSON)
for col in ['highway', 'name', 'maxspeed']:
    if col in edges.columns:
        edges[col] = edges[col].apply(
            lambda x: x[0] if isinstance(x, list) else (str(x) if x is not None else None)
        )

edges.to_file(output_path, driver="GeoJSON")

print(f"Saved {len(edges)} road segments to {output_path}")
print(f"Columns: {edges.columns.tolist()}")
