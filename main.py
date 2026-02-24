"""
🌧️ Rainfall → Road Risk Pipeline — Main Entry Point
------------------------------------------------------
Orchestrates the full pipeline:
  Phase 1-3: Data loading, feature engineering, model training (pre-run)
  Phase 4:   Road network risk modeling
  Phase 5:   Launch interactive dashboard

Usage:
  python main.py --phase 4    # Run risk model only
  python main.py --phase 5    # Launch dashboard only
  python main.py --all        # Run risk model + launch dashboard
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def check_prerequisites():
    """Check that required data files exist."""
    required_files = {
        "Ensemble predictions": "data/processed/monsoon_ensemble_predictions_10mm.csv",
        "Road segments": "data/processed/road_segments.geojson",
    }

    all_ok = True
    for name, path in required_files.items():
        if Path(path).exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} — MISSING")
            all_ok = False

    return all_ok


def run_phase4():
    """Run Phase 4: Road Network Risk Model."""
    print("\n" + "=" * 60)
    print("PHASE 4: Road Network Risk Modeling")
    print("=" * 60)

    # Check if road segments exist
    if not Path("data/processed/road_segments.geojson").exists():
        print("\nRoad segments not found. Downloading from OSM...")
        result = subprocess.run(
            [sys.executable, "src/download_roads.py"],
            cwd=os.getcwd()
        )
        if result.returncode != 0:
            print("❌ Road download failed!")
            return False

    # Check if ensemble predictions exist
    if not Path("data/processed/monsoon_ensemble_predictions_10mm.csv").exists():
        print("❌ Ensemble predictions not found!")
        print("   Run: python src/train_bootstrap_ensemble_10mm.py")
        return False

    # Run risk model
    print("\nRunning risk model pipeline...")
    result = subprocess.run(
        [sys.executable, "src/risk_model.py"],
        cwd=os.getcwd()
    )

    if result.returncode != 0:
        print("❌ Risk model failed!")
        return False

    print("✅ Phase 4 completed successfully!")
    return True


def run_phase5():
    """Run Phase 5: Launch Streamlit Dashboard."""
    print("\n" + "=" * 60)
    print("PHASE 5: Launching Dashboard")
    print("=" * 60)

    if not Path("data/processed/road_risk_scores.geojson").exists():
        print("⚠️  Risk scores not found. Run Phase 4 first (python main.py --phase 4)")
        return False

    print("\nStarting Streamlit dashboard...")
    print("   URL: http://localhost:8501")
    print("   Press Ctrl+C to stop\n")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/dashboard.py"],
        cwd=os.getcwd()
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="🌧️ Rainfall → Road Risk Pipeline"
    )
    parser.add_argument(
        "--phase", type=int, choices=[4, 5],
        help="Run a specific phase (4=Risk Model, 5=Dashboard)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run risk model then launch dashboard"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check prerequisites only"
    )

    args = parser.parse_args()

    print("🌧️ Rainfall → Road Risk Pipeline")
    print("=" * 60)

    if args.check:
        print("\nChecking prerequisites...")
        check_prerequisites()
        return

    if args.phase == 4:
        run_phase4()
    elif args.phase == 5:
        run_phase5()
    elif args.all:
        if run_phase4():
            run_phase5()
    else:
        # Default: show status and usage
        print("\nPrerequisites:")
        all_ok = check_prerequisites()

        if Path("data/processed/road_risk_scores.geojson").exists():
            print(f"\n  ✅ Risk scores: data/processed/road_risk_scores.geojson")
        else:
            print(f"\n  ❌ Risk scores not generated yet")

        print("\nUsage:")
        print("  python main.py --phase 4    # Run risk model")
        print("  python main.py --phase 5    # Launch dashboard")
        print("  python main.py --all        # Run risk model + dashboard")
        print("  python main.py --check      # Check prerequisites")


if __name__ == "__main__":
    main()
