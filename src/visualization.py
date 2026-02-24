"""
Visualization helpers for Rainfall → Road Risk Pipeline
---------------------------------------------------------
Provides plotting functions for:
  - Rainfall probability maps (Folium)
  - Road risk heatmaps (Folium)
  - ROC & PR curves (Plotly)
  - Feature importance (Plotly)
  - Risk distribution (Plotly)
  - Uncertainty visualization (Plotly)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
import branca.colormap as cm


# ================================================================
# FOLIUM MAP HELPERS
# ================================================================

def create_base_map(center_lat=22.0, center_lon=112.0, zoom=9):
    """Create a base Folium map centered on the study area."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB dark_matter"
    )
    return m


def add_rainfall_heatmap(folium_map, rainfall_gdf):
    """
    Add rainfall probability heatmap layer to a Folium map.

    Parameters:
        folium_map: Folium Map object
        rainfall_gdf: GeoDataFrame with latitude, longitude, hazard_probability
    """
    heat_data = [
        [row["latitude"], row["longitude"], row["hazard_probability"]]
        for _, row in rainfall_gdf.iterrows()
        if row["hazard_probability"] > 0
    ]

    if heat_data:
        HeatMap(
            heat_data,
            name="Rainfall Probability",
            radius=15,
            blur=10,
            max_zoom=12,
            gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"}
        ).add_to(folium_map)

    return folium_map


def add_road_risk_layer(folium_map, roads_gdf, column="risk_score"):
    """
    Add color-coded road segments to a Folium map.

    Parameters:
        folium_map: Folium Map object
        roads_gdf: GeoDataFrame with geometry and risk columns
        column: Column to color-code (default: risk_score)
    """
    # Create colormap
    vmin = roads_gdf[column].min()
    vmax = max(roads_gdf[column].max(), 0.01)  # Avoid zero range

    colormap = cm.LinearColormap(
        colors=["green", "yellow", "orange", "red"],
        vmin=vmin,
        vmax=vmax,
        caption=f"Road {column.replace('_', ' ').title()}"
    )
    colormap.add_to(folium_map)

    # Add road segments
    road_layer = folium.FeatureGroup(name="Road Risk")

    for _, row in roads_gdf.iterrows():
        if row.geometry is None:
            continue

        color = colormap(row[column])

        # Create popup info
        popup_html = f"""
        <b>Road Type:</b> {row.get('highway', 'N/A')}<br>
        <b>Risk Score:</b> {row.get('risk_score', 0):.4f}<br>
        <b>Functionality:</b> {row.get('functionality', 0):.4f}<br>
        <b>Vulnerability:</b> {row.get('vulnerability', 0):.4f}<br>
        <b>Hazard Prob:</b> {row.get('hazard_probability', 0):.6f}
        """

        # Handle different geometry types
        if row.geometry.geom_type == "LineString":
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            folium.PolyLine(
                coords,
                color=color,
                weight=3,
                opacity=0.8,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(road_layer)
        elif row.geometry.geom_type == "MultiLineString":
            for line in row.geometry.geoms:
                coords = [(lat, lon) for lon, lat in line.coords]
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=folium.Popup(popup_html, max_width=250)
                ).add_to(road_layer)

    road_layer.add_to(folium_map)
    folium.LayerControl().add_to(folium_map)
    return folium_map


# ================================================================
# PLOTLY CHART HELPERS
# ================================================================

def plot_risk_distribution(roads_gdf):
    """Create a histogram of road risk scores."""
    fig = px.histogram(
        roads_gdf,
        x="risk_score",
        nbins=50,
        title="Distribution of Road Risk Scores",
        labels={"risk_score": "Risk Score"},
        color_discrete_sequence=["#FF6B6B"]
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Risk Score",
        yaxis_title="Number of Road Segments"
    )
    return fig


def plot_functionality_distribution(roads_gdf):
    """Create a histogram of road functionality scores."""
    fig = px.histogram(
        roads_gdf,
        x="functionality",
        nbins=50,
        title="Distribution of Road Functionality",
        labels={"functionality": "Functionality (1 - Risk)"},
        color_discrete_sequence=["#4ECDC4"]
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Functionality Score",
        yaxis_title="Number of Road Segments"
    )
    return fig


def plot_vulnerability_by_road_type(roads_gdf):
    """Bar chart of average vulnerability by road type."""
    if "highway" not in roads_gdf.columns:
        return None

    # Flatten list-type highway values
    df = roads_gdf.copy()
    df["highway_str"] = df["highway"].apply(
        lambda x: x[0] if isinstance(x, list) else (x if isinstance(x, str) else "unknown")
    )

    avg_vuln = df.groupby("highway_str")["vulnerability"].mean().sort_values(ascending=False)

    fig = px.bar(
        x=avg_vuln.index,
        y=avg_vuln.values,
        title="Average Vulnerability by Road Type",
        labels={"x": "Road Type", "y": "Vulnerability"},
        color=avg_vuln.values,
        color_continuous_scale="RdYlGn_r"
    )
    fig.update_layout(template="plotly_dark")
    return fig


def plot_risk_vs_uncertainty(roads_gdf):
    """Scatter plot of risk score vs uncertainty."""
    if "risk_uncertainty" not in roads_gdf.columns:
        return None

    # Sample if too many points
    df = roads_gdf if len(roads_gdf) < 5000 else roads_gdf.sample(5000, random_state=42)

    fig = px.scatter(
        df,
        x="risk_score",
        y="risk_uncertainty",
        title="Risk Score vs. Prediction Uncertainty",
        labels={"risk_score": "Risk Score", "risk_uncertainty": "Risk Uncertainty (σ)"},
        opacity=0.5,
        color_discrete_sequence=["#FFE66D"]
    )
    fig.update_layout(template="plotly_dark")
    return fig


def plot_roc_curve(fpr, tpr, auc_score=None):
    """Plot ROC curve using Plotly."""
    title = "ROC Curve"
    if auc_score is not None:
        title += f" (AUC = {auc_score:.4f})"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color="#FF6B6B", width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                              line=dict(color="gray", dash="dash")))
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark"
    )
    return fig


def plot_feature_importance(importances, feature_names):
    """Plot feature importance as horizontal bar chart."""
    sorted_idx = np.argsort(importances)
    fig = px.bar(
        x=importances[sorted_idx],
        y=np.array(feature_names)[sorted_idx],
        orientation="h",
        title="Feature Importance (Random Forest)",
        labels={"x": "Importance", "y": "Feature"},
        color=importances[sorted_idx],
        color_continuous_scale="Viridis"
    )
    fig.update_layout(template="plotly_dark")
    return fig


def plot_metrics_gauges(metrics_dict):
    """Create gauge indicators for key metrics."""
    fig = go.Figure()

    gauge_data = [
        ("Avg Functionality", metrics_dict.get("avg_functionality", 0), "green"),
        ("Avg Risk", metrics_dict.get("avg_risk", 0), "red"),
        ("High Risk %", metrics_dict.get("pct_high_risk", 0) / 100, "orange"),
    ]

    for i, (name, value, color) in enumerate(gauge_data):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value * 100 if "%" not in name else value * 100,
            title={"text": name},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
            domain={"row": 0, "column": i}
        ))

    fig.update_layout(
        grid={"rows": 1, "columns": 3, "pattern": "independent"},
        template="plotly_dark",
        height=250
    )
    return fig
