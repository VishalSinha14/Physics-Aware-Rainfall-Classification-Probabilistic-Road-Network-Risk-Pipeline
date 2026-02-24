"""
🌧️ Rainfall → Road Risk Dashboard
------------------------------------
Interactive Streamlit dashboard with:
  1. Rainfall probability map
  2. Road risk heatmap
  3. Evaluation metrics
  4. Uncertainty visualization
  5. Risk distribution charts

Run: streamlit run app/dashboard.py
"""

import sys
import os
import json

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

from src.evaluation_metrics import (
    compute_infrastructure_metrics,
    compute_uncertainty_metrics,
    compute_meteorological_metrics
)
from src.visualization import (
    create_base_map,
    add_road_risk_layer,
    add_rainfall_heatmap,
    plot_risk_distribution,
    plot_functionality_distribution,
    plot_vulnerability_by_road_type,
    plot_risk_vs_uncertainty,
    plot_metrics_gauges
)

# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="🌧️ Rainfall → Road Risk Dashboard",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# DATA LOADING (cached)
# ================================================================

@st.cache_data
def load_road_risk():
    """Load risk-scored road network."""
    path = "data/processed/road_risk_scores.geojson"
    if not os.path.exists(path):
        return None
    return gpd.read_file(path)


@st.cache_data
def load_ensemble_predictions():
    """Load ensemble predictions with spatial columns."""
    path = "data/processed/monsoon_ensemble_predictions_10mm.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_rainfall_grid():
    """Aggregate ensemble predictions to per-grid-cell hazard."""
    df = load_ensemble_predictions()
    if df is None:
        return None

    grid = df.groupby(["latitude", "longitude"]).agg(
        hazard_probability=("mean_probability", "mean"),
        hazard_uncertainty=("uncertainty_std", "mean"),
    ).reset_index()
    return grid


# ================================================================
# SIDEBAR
# ================================================================

def render_sidebar():
    """Render sidebar controls."""
    st.sidebar.title("🌧️ Rainfall → Road Risk")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview", "🗺️ Risk Map", "📈 Metrics", "🔬 Uncertainty",
         "🧪 Phase 6 Validation", "🆚 Model Comparison", "⏱️ Lead-Time",
         "📋 Data Explorer"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Study Area")
    st.sidebar.markdown("**Region:** South China / Guangdong")
    st.sidebar.markdown("**Period:** Jun-Aug 2022 (Monsoon)")
    st.sidebar.markdown("**Hazard Threshold:** ≥10 mm/hr")
    st.sidebar.markdown("**Ensemble:** 30 RF models")

    return page


# ================================================================
# OVERVIEW PAGE
# ================================================================

def render_overview(roads_gdf, predictions_df):
    """Render the overview dashboard."""
    st.title("🌧️ Physics-Aware Rainfall → Road Risk Pipeline")
    st.markdown("### End-to-End Hazard → Infrastructure Risk Forecasting")
    st.markdown("---")

    if roads_gdf is None:
        st.error("⚠️ Road risk data not found. Run `python src/risk_model.py` first.")
        return

    # Key metrics row
    infra_metrics = compute_infrastructure_metrics(roads_gdf)
    uncert_metrics = compute_uncertainty_metrics(roads_gdf)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🛤️ Total Road Segments",
            f"{len(roads_gdf):,}"
        )
    with col2:
        st.metric(
            "✅ Avg Functionality",
            f"{infra_metrics['avg_functionality']:.4f}",
            delta=f"{(infra_metrics['avg_functionality'] - 1) * 100:.2f}%"
        )
    with col3:
        st.metric(
            "⚠️ High Risk Roads",
            f"{infra_metrics['pct_high_risk']:.1f}%"
        )
    with col4:
        st.metric(
            "🔬 Mean Uncertainty",
            f"{uncert_metrics.get('mean_risk_uncertainty', 0):.6f}"
        )

    st.markdown("---")

    # Charts row
    col_a, col_b = st.columns(2)

    with col_a:
        fig = plot_risk_distribution(roads_gdf)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = plot_functionality_distribution(roads_gdf)
        st.plotly_chart(fig, use_container_width=True)

    # Vulnerability by road type
    fig = plot_vulnerability_by_road_type(roads_gdf)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Pipeline flowchart
    st.markdown("---")
    st.markdown("### Pipeline Architecture")
    st.markdown("""
    ```
    ERA5 Rainfall (NetCDF) → Temporal Features → Random Forest (10mm) → Bootstrap Ensemble (30 models)
        → Mean Probability + Uncertainty → Spatial Join with OSM Roads → Risk = Hazard × Vulnerability
        → Road Functionality Map → This Dashboard
    ```
    """)


# ================================================================
# RISK MAP PAGE
# ================================================================

def render_risk_map(roads_gdf, rainfall_grid):
    """Render the interactive risk map."""
    st.title("🗺️ Road Risk Map")
    st.markdown("Interactive map showing road risk scores and rainfall probability")

    if roads_gdf is None:
        st.error("⚠️ Road risk data not found. Run `python src/risk_model.py` first.")
        return

    # Map controls
    col1, col2 = st.columns([1, 3])
    with col1:
        show_rainfall = st.checkbox("Show Rainfall Heatmap", value=True)
        show_roads = st.checkbox("Show Road Risk", value=True)
        color_by = st.selectbox("Color roads by", ["risk_score", "functionality", "vulnerability"])

        # Limit to top-N risky roads for performance
        max_roads = st.slider("Max roads to display", 500, 10000, 3000, step=500)

    with col2:
        # Build map
        m = create_base_map(center_lat=22.0, center_lon=112.0, zoom=9)

        if show_rainfall and rainfall_grid is not None:
            m = add_rainfall_heatmap(m, rainfall_grid)

        if show_roads:
            # Sort by risk and take top N for performance
            display_roads = roads_gdf.nlargest(max_roads, "risk_score")
            m = add_road_risk_layer(m, display_roads, column=color_by)

        st_folium(m, width=900, height=600)


# ================================================================
# METRICS PAGE
# ================================================================

def render_metrics(roads_gdf, predictions_df):
    """Render evaluation metrics."""
    st.title("📈 Evaluation Metrics")

    if roads_gdf is None:
        st.error("⚠️ Road risk data not found.")
        return

    # Infrastructure metrics
    st.markdown("### 🏗️ Infrastructure Metrics")
    infra = compute_infrastructure_metrics(roads_gdf)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Network Functionality", f"{infra['avg_functionality']:.4f}")
        st.metric("Avg Risk Score", f"{infra['avg_risk']:.6f}")
    with col2:
        st.metric("High Risk Roads (>5%)", f"{infra['pct_high_risk']:.1f}%")
        st.metric("Medium Risk Roads (1-5%)", f"{infra['pct_medium_risk']:.1f}%")
    with col3:
        st.metric("Low Risk Roads (<1%)", f"{infra['pct_low_risk']:.1f}%")
        if "critical_roads_count" in infra:
            st.metric("Critical Roads", infra["critical_roads_count"])

    # Meteorological metrics (from ensemble predictions)
    if predictions_df is not None and "true_label" in predictions_df.columns:
        st.markdown("---")
        st.markdown("### 🌧️ Meteorological Metrics")

        y_true = predictions_df["true_label"]
        y_prob = predictions_df["mean_probability"]
        threshold = st.slider("Classification threshold", 0.0, 1.0, 0.5, 0.01)
        y_pred = (y_prob >= threshold).astype(int)

        met_metrics = compute_meteorological_metrics(y_true, y_pred, y_prob)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("POD (Recall)", f"{met_metrics.get('POD', 0):.4f}")
        with col2:
            st.metric("FAR", f"{met_metrics.get('FAR', 0):.4f}")
        with col3:
            st.metric("CSI", f"{met_metrics.get('CSI', 0):.4f}")
        with col4:
            st.metric("ROC-AUC", f"{met_metrics.get('ROC_AUC', 0):.4f}")

        # Confusion matrix
        st.markdown("#### Confusion Matrix")
        cm_data = pd.DataFrame(
            [[met_metrics.get("TN", 0), met_metrics.get("FP", 0)],
             [met_metrics.get("FN", 0), met_metrics.get("TP", 0)]],
            index=["Actual: No Rain", "Actual: Heavy Rain"],
            columns=["Predicted: No Rain", "Predicted: Heavy Rain"]
        )
        st.dataframe(cm_data, use_container_width=True)


# ================================================================
# UNCERTAINTY PAGE
# ================================================================

def render_uncertainty(roads_gdf, predictions_df):
    """Render uncertainty analysis."""
    st.title("🔬 Uncertainty Analysis")
    st.markdown("Epistemic uncertainty from 30-model bootstrap ensemble")

    if roads_gdf is None:
        st.error("⚠️ Road risk data not found.")
        return

    uncert = compute_uncertainty_metrics(roads_gdf)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Risk Uncertainty", f"{uncert.get('mean_risk_uncertainty', 0):.6f}")
    with col2:
        st.metric("Max Risk Uncertainty", f"{uncert.get('max_risk_uncertainty', 0):.6f}")
    with col3:
        st.metric("Std Risk Uncertainty", f"{uncert.get('std_risk_uncertainty', 0):.6f}")

    # Risk vs Uncertainty scatter
    fig = plot_risk_vs_uncertainty(roads_gdf)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Uncertainty distribution
    if "risk_uncertainty" in roads_gdf.columns:
        fig = px.histogram(
            roads_gdf,
            x="risk_uncertainty",
            nbins=50,
            title="Distribution of Risk Uncertainty",
            color_discrete_sequence=["#FFE66D"]
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # High-uncertainty road segments
    st.markdown("### Top 20 Most Uncertain Road Segments")
    if "risk_uncertainty" in roads_gdf.columns:
        top_uncertain = roads_gdf.nlargest(20, "risk_uncertainty")[
            ["highway", "risk_score", "functionality", "vulnerability",
             "hazard_probability", "risk_uncertainty"]
        ].reset_index(drop=True)
        st.dataframe(top_uncertain, use_container_width=True)


# ================================================================
# DATA EXPLORER PAGE
# ================================================================

def render_data_explorer(roads_gdf, predictions_df):
    """Render raw data explorer."""
    st.title("📋 Data Explorer")

    tab1, tab2 = st.tabs(["Road Risk Data", "Ensemble Predictions"])

    with tab1:
        if roads_gdf is not None:
            st.markdown(f"**{len(roads_gdf):,} road segments** with risk scores")

            # Filter controls
            cols_to_show = [c for c in roads_gdf.columns if c != "geometry"]
            selected_cols = st.multiselect("Columns", cols_to_show, default=cols_to_show[:8])

            if selected_cols:
                st.dataframe(roads_gdf[selected_cols].head(500), use_container_width=True)

            # Download button
            csv = roads_gdf.drop(columns=["geometry"]).to_csv(index=False)
            st.download_button("📥 Download as CSV", csv, "road_risk_scores.csv", "text/csv")
        else:
            st.warning("No road risk data available.")

    with tab2:
        if predictions_df is not None:
            st.markdown(f"**{len(predictions_df):,} predictions** from bootstrap ensemble")
            st.dataframe(predictions_df.head(500), use_container_width=True)
        else:
            st.warning("No ensemble prediction data available.")


# ================================================================
# PHASE 6 VALIDATION PAGE
# ================================================================

PHASE6_DIR = "results/phase6"

@st.cache_data
def load_phase6_json(filename):
    """Load a Phase 6 JSON result file."""
    path = os.path.join(PHASE6_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def show_phase6_section(title, png_file, json_file, description=""):
    """Helper to display a Phase 6 plot + metrics."""
    st.markdown(f"### {title}")
    if description:
        st.markdown(description)

    png_path = os.path.join(PHASE6_DIR, png_file)
    if os.path.exists(png_path):
        img = Image.open(png_path)
        st.image(img, use_container_width=True)
    else:
        st.warning(f"Plot not found: {png_file}")

    data = load_phase6_json(json_file)
    if data:
        with st.expander("📊 View Metrics (JSON)"):
            st.json(data)

    st.markdown("---")


def render_phase6():
    """Render Phase 6 research-grade validation results."""
    st.title("🧪 Phase 6: Research-Grade Validation")
    st.markdown("Scientific validation of the hazard model and risk pipeline.")

    if not os.path.exists(PHASE6_DIR):
        st.error("⚠️ Phase 6 results not found. Run `python src/phase6_validation.py` first.")
        return

    # Summary metrics at top
    ts = load_phase6_json("1_time_split_metrics.json")
    cal = load_phase6_json("2_calibration_metrics.json")
    typh = load_phase6_json("3_typhoon_results.json")
    add = load_phase6_json("10_additional_metrics.json")

    if ts and ts.get("test"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ROC-AUC (Time Split)", f"{ts['test']['ROC_AUC']:.4f}")
        with col2:
            st.metric("CSI (Time Split)", f"{ts['test']['CSI']:.4f}")
        with col3:
            brier = cal.get("brier_before", 0) if cal else 0
            st.metric("Brier Score", f"{brier:.6f}")
        with col4:
            amp = typh.get("amplification_factor", 0) if typh else 0
            st.metric("Typhoon Amplification", f"{amp:.0f}x")

    st.markdown("---")

    # All 10 sections

    # Section 1: Time-based split (no PNG, show metrics as table)
    st.markdown("### 1️⃣ Time-Based Train/Test Split")
    st.markdown("Train on Jun–Jul, test on Aug 16–31 — respects temporal causality.")
    ts_data = load_phase6_json("1_time_split_metrics.json")
    if ts_data:
        for split_name, metrics in ts_data.items():
            st.markdown(f"**{split_name.title()} Set**")
            metric_cols = st.columns(min(len(metrics), 4))
            for i, (k, v) in enumerate(metrics.items()):
                with metric_cols[i % len(metric_cols)]:
                    if isinstance(v, float):
                        st.metric(k, f"{v:.4f}")
                    else:
                        st.metric(k, f"{v:,}")
    st.markdown("---")

    show_phase6_section(
        "2️⃣ Reliability Diagram & Probability Calibration",
        "2_reliability_diagram.png", "2_calibration_metrics.json",
        "Checks if predicted probabilities match observed frequencies. Isotonic regression applied."
    )

    show_phase6_section(
        "3️⃣ Typhoon Stress Test",
        "3_typhoon_stress_test.png", "3_typhoon_results.json",
        "Injects extreme typhoon-level rainfall to stress-test the hazard model."
    )

    show_phase6_section(
        "4️⃣ Dynamic Hazard Scaling",
        "4_dynamic_scaling.png", "4_scaling_results.json",
        "Tests risk engine stability under 1x, 1.5x, 2x and 3x rainfall amplification."
    )

    show_phase6_section(
        "5️⃣ Threshold Sensitivity (5mm / 10mm / 20mm)",
        "5_threshold_sensitivity.png", "5_threshold_results.json",
        "Compares three rainfall thresholds to understand threshold-induced bias."
    )

    show_phase6_section(
        "6️⃣ Multi-Threshold Fusion",
        "6_multi_threshold_fusion.png", "6_fusion_results.json",
        "Graded severity risk from minor (5mm), moderate (10mm) and major (20mm) disruption."
    )

    show_phase6_section(
        "7️⃣ Spatial Cross-Validation",
        "7_spatial_cv.png", "7_spatial_cv_results.json",
        "Leave-one-region-out validation (16 spatial blocks) to test generalization."
    )

    show_phase6_section(
        "8️⃣ Monte Carlo Vulnerability Perturbation",
        "8_monte_carlo_validation.png", "8_monte_carlo_results.json",
        "100 simulations with ±20% vulnerability weight noise to assess robustness."
    )

    show_phase6_section(
        "9️⃣ Ensemble Diversity Comparison",
        "9_ensemble_diversity.png", "9_ensemble_diversity_results.json",
        "Compares 4 ensemble approaches: Bootstrap RF, Random Features, Varying HP, Mixed RF+GBM."
    )

    show_phase6_section(
        "📊 Additional Research Metrics (CRPS, Brier Decomposition)",
        "10_additional_metrics.png", "10_additional_metrics.json",
        "Brier Score decomposition (Reliability–Resolution–Uncertainty), CRPS, risk exceedance."
    )


# ================================================================
# MAIN
# ================================================================

def main():
    page = render_sidebar()

    # Load data
    roads_gdf = load_road_risk()
    predictions_df = load_ensemble_predictions()
    rainfall_grid = load_rainfall_grid()

    if page == "📊 Overview":
        render_overview(roads_gdf, predictions_df)
    elif page == "🗺️ Risk Map":
        render_risk_map(roads_gdf, rainfall_grid)
    elif page == "📈 Metrics":
        render_metrics(roads_gdf, predictions_df)
    elif page == "🔬 Uncertainty":
        render_uncertainty(roads_gdf, predictions_df)
    elif page == "🧪 Phase 6 Validation":
        render_phase6()
    elif page == "🆚 Model Comparison":
        render_model_comparison()
    elif page == "⏱️ Lead-Time":
        render_leadtime()
    elif page == "📋 Data Explorer":
        render_data_explorer(roads_gdf, predictions_df)


# ================================================================
# PAGE: MODEL COMPARISON
# ================================================================

def render_model_comparison():
    """Logistic Regression vs Ensemble RF comparison."""
    st.title("🆚 Model Comparison")
    st.markdown(
        "**Why this matters:** Comparing Logistic Regression (simple baseline) against "
        "Ensemble Random Forest proves that the non-linear model adds real scientific value. "
        "If the baseline were equally good, complex modeling would be unjustified."
    )
    st.markdown("---")

    # Load JSON results
    lr_path  = "results/baselines/lr_results.json"
    cmp_path = "results/baselines/model_comparison.json"

    if not os.path.exists(cmp_path):
        st.warning("Run `python src/baselines/train_logistic_regression.py` then `python src/baselines/model_comparison.py` first.")
        return

    with open(cmp_path) as f:
        cmp_data = json.load(f)

    # KPI row
    if os.path.exists(lr_path):
        with open(lr_path) as f:
            lr_data = json.load(f)
        m = lr_data.get("metrics", {})
        impr = cmp_data.get("improvements_pct", {})
        st.subheader("Logistic Regression Baseline Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("POD",      f"{m.get('POD', 0):.4f}")
        col2.metric("CSI",      f"{m.get('CSI', 0):.4f}")
        col3.metric("ROC-AUC",  f"{m.get('ROC_AUC', 0):.4f}")
        col4.metric("Brier Score", f"{m.get('Brier', 0):.6f}")

        # McNemar test result
        mc = lr_data.get("mcnemar_vs_ensemble")
        if mc:
            st.markdown("---")
            sig = "✅ YES" if mc.get("significant") else "❌ NO"
            st.info(f"**McNemar Test** (Ensemble RF vs LR): χ²={mc['statistic']:.4f}, "
                    f"p={mc['p_value']:.6f} — Statistically significant difference: **{sig}**")

    st.markdown("---")

    # Comparison table from JSON
    st.subheader("Side-by-Side Metric Comparison")
    rows = cmp_data.get("comparison", [])
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df.style.highlight_max(
            subset=[c for c in df.columns if c in ["POD","CSI","ROC-AUC","PR-AUC"]],
            color="#D5F5E3"
        ).highlight_min(
            subset=[c for c in df.columns if c in ["FAR","Brier"]],
            color="#D5F5E3"
        ), use_container_width=True)

    st.markdown("---")

    # Comparison plot image
    st.subheader("Comparison Chart")
    plot_path = "results/baselines/model_comparison_table.png"
    if os.path.exists(plot_path):
        img = Image.open(plot_path)
        st.image(img, use_container_width=True,
                 caption="Left: Metric table | Right: Key metrics bar chart (green=best)")

    # LR analysis image
    lr_plot = "results/baselines/lr_analysis.png"
    if os.path.exists(lr_plot):
        st.subheader("Logistic Regression Analysis")
        img = Image.open(lr_plot)
        st.image(img, use_container_width=True,
                 caption="Left: Reliability diagram | Right: ROC curve")

    # PR curves image
    pr_plot = "results/scientific/pr_curves_comparison.png"
    if os.path.exists(pr_plot):
        st.subheader("Precision-Recall Curves (Better than ROC for Imbalanced Data)")
        img = Image.open(pr_plot)
        st.image(img, use_container_width=True)

    # Imbalance analysis
    imb_path = "results/scientific/imbalance_analysis.png"
    if os.path.exists(imb_path):
        st.subheader("Class Imbalance Analysis")
        img = Image.open(imb_path)
        st.image(img, use_container_width=True,
                 caption="Why accuracy is misleading — and what metrics to use instead")

        imb_json = "results/scientific/imbalance_report.json"
        if os.path.exists(imb_json):
            with open(imb_json) as f:
                imb = json.load(f)
            with st.expander("📄 Imbalance Report JSON"):
                st.json(imb)


# ================================================================
# PAGE: LEAD-TIME FORECASTING
# ================================================================

def render_leadtime():
    """Lead-time forecasting degradation page."""
    st.title("⏱️ Lead-Time Forecasting")
    st.markdown(
        "This page shows how model performance **degrades as we predict further into the future**. "
        "A nowcast (0h) uses recent rainfall — a 30-minute or 60-minute forecast only uses "
        "atmospheric variables from the current time. The smooth degradation proves the model "
        "has **genuine predictive skill** (not just pattern-matching)."
    )
    st.markdown("---")

    metrics_path = "results/leadtime/leadtime_metrics.json"
    if not os.path.exists(metrics_path):
        st.warning("Run `python src/leadtime/train_leadtime_models.py` first.")
        return

    with open(metrics_path) as f:
        lead_metrics = json.load(f)

    # KPI table
    st.subheader("Performance at Each Lead Time")
    lead_labels = {"0h": "0h (Nowcast)", "30m": "+30 min", "60m": "+60 min"}
    cols = st.columns(len(lead_metrics))
    for i, (lead, m) in enumerate(lead_metrics.items()):
        with cols[i]:
            st.markdown(f"**{lead_labels.get(lead, lead)}**")
            st.metric("ROC-AUC", f"{m.get('ROC_AUC', 0):.4f}")
            st.metric("CSI",     f"{m.get('CSI', 0):.4f}")
            st.metric("POD",     f"{m.get('POD', 0):.4f}")
            st.metric("Brier",   f"{m.get('Brier', 0):.5f}")

    st.markdown("---")

    # Degradation curve plot
    st.subheader("Degradation Curve")
    deg_path = "results/leadtime/degradation_curve.png"
    if os.path.exists(deg_path):
        img = Image.open(deg_path)
        st.image(img, use_container_width=True,
                 caption="Performance drops as lead time increases — this is expected and validates the model")
    else:
        st.info("Run `python src/leadtime/evaluate_leadtime.py`")

    # Degradation summary
    deg_sum_path = "results/leadtime/degradation_summary.json"
    if os.path.exists(deg_sum_path):
        with open(deg_sum_path) as f:
            deg_sum = json.load(f)
        st.markdown("**Key degradation rates (0h → 60m):**")
        metrics_info = deg_sum.get("metrics", {})
        cols2 = st.columns(min(len(metrics_info), 5))
        for i, (mkey, info) in enumerate(metrics_info.items()):
            vals = info.get("values_by_lead", {})
            ch   = info.get("total_change_pct", 0)
            higher_better = info.get("higher_better", True)
            direction = "📉" if ch > 0 and higher_better else "📈"
            with cols2[i % 5]:
                st.metric(mkey, f"{ch:.1f}%", delta=f"{direction} change",
                          delta_color="inverse" if higher_better else "normal")

    # Reliability comparison for lead times
    rel_path = "results/scientific/reliability_all_models.png"
    if os.path.exists(rel_path):
        st.markdown("---")
        st.subheader("Reliability Diagrams — All Lead Times")
        img = Image.open(rel_path)
        st.image(img, use_container_width=True,
                 caption="Each panel: predicted probability vs observed frequency. Closer to diagonal = better calibration.")

    # Brier comparison
    brier_path = "results/scientific/brier_comparison.png"
    if os.path.exists(brier_path):
        st.markdown("---")
        st.subheader("Brier Score Comparison")
        img = Image.open(brier_path)
        st.image(img, use_container_width=True,
                 caption="Brier Score: lower = better probabilistic calibration")

    # Full metrics JSON
    with st.expander("📄 Full Lead-Time Metrics JSON"):
        st.json(lead_metrics)


if __name__ == "__main__":
    main()
