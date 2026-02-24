"""
Phase 4A: Road Network Risk Model
-----------------------------------
Pipeline:
  1. Load road segments (GeoJSON from download_roads.py)
  2. Convert rainfall grid → GeoDataFrame
  3. Spatial join (nearest neighbor) — assigns hazard to each road
  4. Build vulnerability index (road type + betweenness centrality)
  5. Compute Risk = Hazard × Vulnerability → export GeoJSON
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import Point
from pathlib import Path

# ================================================================
# CONFIGURATION
# ================================================================

ROAD_SEGMENTS_PATH = "data/processed/road_segments.geojson"
ENSEMBLE_PREDICTIONS_PATH = "data/processed/monsoon_ensemble_predictions_10mm.csv"
OUTPUT_PATH = "data/processed/road_risk_scores.geojson"

# Road bounding box (must match download_roads.py)
NORTH, SOUTH, WEST, EAST = 23.4, 22.9, 113.0, 113.5

# Vulnerability weights by road type (0 = low vulnerability, 1 = high)
ROAD_TYPE_VULNERABILITY = {
    "motorway":       0.10,
    "motorway_link":  0.15,
    "trunk":          0.20,
    "trunk_link":     0.25,
    "primary":        0.30,
    "primary_link":   0.35,
    "secondary":      0.40,
    "secondary_link": 0.45,
    "tertiary":       0.50,
    "tertiary_link":  0.55,
    "unclassified":   0.65,
    "residential":    0.75,
    "living_street":  0.80,
    "service":        0.70,
    "track":          0.85,
    "road":           0.60,
}
DEFAULT_VULNERABILITY = 0.60  # Fallback for unknown road types

# Weighting between road-type and centrality vulnerability
W_TYPE = 0.5
W_CENTRALITY = 0.5


# ================================================================
# STEP 1: LOAD ROAD SEGMENTS
# ================================================================

def load_road_segments(path):
    """Load road segments GeoJSON and ensure CRS is WGS84."""
    print("\n[Step 1] Loading road segments...")

    roads = gpd.read_file(path)
    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")
    elif roads.crs.to_epsg() != 4326:
        roads = roads.to_crs("EPSG:4326")

    print(f"  Loaded {len(roads)} road segments")
    print(f"  Columns: {roads.columns.tolist()}")
    return roads


# ================================================================
# STEP 2: RAINFALL GRID → GEODATAFRAME
# ================================================================

def load_and_aggregate_rainfall(path):
    """
    Load ensemble predictions, aggregate per grid cell to get:
      - hazard_probability: mean of mean_probability per (lat, lon)
      - hazard_uncertainty: mean of uncertainty_std per (lat, lon)
    Returns a GeoDataFrame with Point geometry.
    """
    print("\n[Step 2] Loading and aggregating rainfall predictions...")

    df = pd.read_csv(path)
    print(f"  Raw predictions: {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")

    # Check for required columns
    required = ["latitude", "longitude", "mean_probability", "uncertainty_std"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in ensemble predictions: {missing}\n"
            f"Please re-run train_bootstrap_ensemble_10mm.py to generate "
            f"predictions with lat/lon columns."
        )

    # Aggregate per grid cell (average across all time steps)
    grid_agg = df.groupby(["latitude", "longitude"]).agg(
        hazard_probability=("mean_probability", "mean"),
        hazard_uncertainty=("uncertainty_std", "mean"),
        sample_count=("mean_probability", "count")
    ).reset_index()

    print(f"  Aggregated to {len(grid_agg)} unique grid cells")

    # Convert to GeoDataFrame
    geometry = [Point(lon, lat) for lat, lon in zip(grid_agg["latitude"], grid_agg["longitude"])]
    rainfall_gdf = gpd.GeoDataFrame(grid_agg, geometry=geometry, crs="EPSG:4326")

    print(f"  Hazard probability range: [{grid_agg['hazard_probability'].min():.6f}, "
          f"{grid_agg['hazard_probability'].max():.6f}]")

    return rainfall_gdf


# ================================================================
# STEP 3: SPATIAL JOIN (NEAREST NEIGHBOR)
# ================================================================

def spatial_join_roads_rainfall(roads, rainfall_gdf):
    """
    For each road segment, find the nearest rainfall grid cell
    and assign its hazard probability + uncertainty.
    """
    print("\n[Step 3] Performing spatial join (nearest neighbor)...")

    # Compute centroids of road segments for matching
    roads_with_centroid = roads.copy()
    roads_with_centroid["centroid"] = roads_with_centroid.geometry.centroid
    roads_centroids = roads_with_centroid.set_geometry("centroid")

    # Nearest spatial join
    joined = gpd.sjoin_nearest(
        roads_centroids[["centroid"]],
        rainfall_gdf[["geometry", "hazard_probability", "hazard_uncertainty"]],
        how="left",
        distance_col="join_distance"
    )

    # Remove duplicates (if a road centroid matches multiple equidistant cells)
    joined = joined[~joined.index.duplicated(keep="first")]

    # Transfer hazard values back to original roads
    roads["hazard_probability"] = joined["hazard_probability"].values
    roads["hazard_uncertainty"] = joined["hazard_uncertainty"].values
    roads["join_distance_deg"] = joined["join_distance"].values

    # Fill NaN (roads too far from any grid cell)
    roads["hazard_probability"] = roads["hazard_probability"].fillna(0.0)
    roads["hazard_uncertainty"] = roads["hazard_uncertainty"].fillna(0.0)

    print(f"  Joined {len(roads)} road segments to nearest rainfall cell")
    print(f"  Mean join distance: {roads['join_distance_deg'].mean():.4f}°")

    return roads


# ================================================================
# STEP 4: VULNERABILITY INDEX
# ================================================================

def get_road_type_vulnerability(highway_val):
    """Map road type to vulnerability score [0, 1]."""
    if isinstance(highway_val, list):
        # Some OSM roads have multiple type tags — take the first
        highway_val = highway_val[0]

    if isinstance(highway_val, str):
        return ROAD_TYPE_VULNERABILITY.get(highway_val, DEFAULT_VULNERABILITY)

    return DEFAULT_VULNERABILITY


def compute_vulnerability(roads):
    """
    Build vulnerability index combining:
      - Road type vulnerability (0-1)
      - Betweenness centrality from road network graph (0-1)
    """
    print("\n[Step 4] Computing vulnerability index...")

    # --- 4a: Road type vulnerability ---
    print("  Computing road type vulnerability...")
    roads["vuln_type"] = roads["highway"].apply(get_road_type_vulnerability)

    # --- 4b: Network centrality ---
    print("  Downloading road graph for centrality analysis...")
    bbox = (WEST, SOUTH, EAST, NORTH)

    try:
        G = ox.graph_from_bbox(bbox=bbox, network_type="drive")
        print(f"  Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

        print("  Computing betweenness centrality (this may take a moment)...")
        # Use edge betweenness centrality (weight by length)
        edge_centrality = nx.edge_betweenness_centrality(G, weight="length", k=min(500, len(G.nodes)))

        # Convert to DataFrame for matching
        centrality_data = []
        for (u, v, k), cent_val in edge_centrality.items():
            centrality_data.append({"u": u, "v": v, "key": k, "centrality": cent_val})

        cent_df = pd.DataFrame(centrality_data)

        # Normalize centrality to [0, 1]
        if cent_df["centrality"].max() > 0:
            cent_df["centrality_norm"] = cent_df["centrality"] / cent_df["centrality"].max()
        else:
            cent_df["centrality_norm"] = 0.0

        # Convert graph edges to GeoDataFrame for spatial matching
        edges_gdf = ox.graph_to_gdfs(G, nodes=False)
        edges_gdf["centrality_norm"] = cent_df["centrality_norm"].values[:len(edges_gdf)]

        # Match road segments to graph edges by nearest centroid
        road_centroids = roads.copy()
        road_centroids["centroid"] = road_centroids.geometry.centroid
        road_centroids = road_centroids.set_geometry("centroid")

        edge_centroids = edges_gdf.copy()
        edge_centroids["edge_centroid"] = edge_centroids.geometry.centroid
        edge_centroids = edge_centroids.set_geometry("edge_centroid")

        cent_join = gpd.sjoin_nearest(
            road_centroids[["centroid"]],
            edge_centroids[["edge_centroid", "centrality_norm"]],
            how="left"
        )
        cent_join = cent_join[~cent_join.index.duplicated(keep="first")]

        roads["vuln_centrality"] = cent_join["centrality_norm"].fillna(0.0).values

    except Exception as e:
        print(f"  Warning: Centrality computation failed ({e})")
        print("  Using road type vulnerability only.")
        roads["vuln_centrality"] = 0.0

    # --- 4c: Combined vulnerability ---
    roads["vulnerability"] = (
        W_TYPE * roads["vuln_type"] +
        W_CENTRALITY * roads["vuln_centrality"]
    )

    # Clip to [0, 1]
    roads["vulnerability"] = roads["vulnerability"].clip(0, 1)

    print(f"  Vulnerability range: [{roads['vulnerability'].min():.4f}, "
          f"{roads['vulnerability'].max():.4f}]")

    return roads


# ================================================================
# STEP 5: COMPUTE RISK & EXPORT
# ================================================================

def compute_risk(roads):
    """
    Risk = Hazard_Probability × Vulnerability
    Functionality = 1 − Risk
    Risk_uncertainty = Hazard_Uncertainty × Vulnerability
    """
    print("\n[Step 5] Computing risk scores...")

    roads["risk_score"] = roads["hazard_probability"] * roads["vulnerability"]
    roads["functionality"] = 1.0 - roads["risk_score"]
    roads["risk_uncertainty"] = roads["hazard_uncertainty"] * roads["vulnerability"]

    # Clip all scores to [0, 1]
    for col in ["risk_score", "functionality", "risk_uncertainty"]:
        roads[col] = roads[col].clip(0, 1)

    print(f"  Risk score range:    [{roads['risk_score'].min():.6f}, "
          f"{roads['risk_score'].max():.6f}]")
    print(f"  Functionality range: [{roads['functionality'].min():.6f}, "
          f"{roads['functionality'].max():.6f}]")
    print(f"  Mean risk:           {roads['risk_score'].mean():.6f}")
    print(f"  Mean functionality:  {roads['functionality'].mean():.6f}")

    return roads


def export_results(roads, output_path):
    """Export risk-scored roads as GeoJSON."""
    print(f"\n  Exporting to {output_path}...")

    # Keep only relevant columns for output
    output_cols = [
        "geometry", "highway", "length",
        "hazard_probability", "hazard_uncertainty",
        "vulnerability", "vuln_type", "vuln_centrality",
        "risk_score", "functionality", "risk_uncertainty"
    ]
    # Filter to only columns that exist
    output_cols = [c for c in output_cols if c in roads.columns]
    roads_out = roads[output_cols]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    roads_out.to_file(output_path, driver="GeoJSON")

    print(f"  Saved {len(roads_out)} risk-scored road segments")


# ================================================================
# SUMMARY
# ================================================================

def print_summary(roads):
    """Print a summary of the risk analysis."""
    print("\n" + "=" * 60)
    print("PHASE 4A — RISK ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nTotal road segments:           {len(roads)}")
    print(f"Mean hazard probability:       {roads['hazard_probability'].mean():.6f}")
    print(f"Mean vulnerability:            {roads['vulnerability'].mean():.4f}")
    print(f"Mean risk score:               {roads['risk_score'].mean():.6f}")
    print(f"Mean functionality:            {roads['functionality'].mean():.6f}")

    # Risk categories
    high_risk = (roads["risk_score"] > 0.05).sum()
    med_risk = ((roads["risk_score"] > 0.01) & (roads["risk_score"] <= 0.05)).sum()
    low_risk = (roads["risk_score"] <= 0.01).sum()

    print(f"\nRisk distribution:")
    print(f"  High risk (>0.05):  {high_risk} segments ({100*high_risk/len(roads):.1f}%)")
    print(f"  Medium (0.01-0.05): {med_risk} segments ({100*med_risk/len(roads):.1f}%)")
    print(f"  Low risk (≤0.01):   {low_risk} segments ({100*low_risk/len(roads):.1f}%)")

    print(f"\nOutput: {OUTPUT_PATH}")
    print("=" * 60)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    # Step 1: Load roads
    roads = load_road_segments(ROAD_SEGMENTS_PATH)

    # Step 2: Load + aggregate rainfall hazard
    rainfall_gdf = load_and_aggregate_rainfall(ENSEMBLE_PREDICTIONS_PATH)

    # Step 3: Spatial join
    roads = spatial_join_roads_rainfall(roads, rainfall_gdf)

    # Step 4: Vulnerability
    roads = compute_vulnerability(roads)

    # Step 5: Risk + export
    roads = compute_risk(roads)
    export_results(roads, OUTPUT_PATH)

    # Summary
    print_summary(roads)
