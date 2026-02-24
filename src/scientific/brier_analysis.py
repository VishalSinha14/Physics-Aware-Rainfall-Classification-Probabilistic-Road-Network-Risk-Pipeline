"""
Brier Score Analysis
---------------------
Computes Brier Score for all models (LR, Ensemble RF at 0h, 30m, 60m)
and produces a comparison bar chart.

Brier Score = mean((p - y)^2), lower is better.

Outputs → results/scientific/brier_comparison.png
          results/scientific/brier_scores.json
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/scientific", exist_ok=True)


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    print("=" * 60)
    print("Brier Score Comparison")
    print("=" * 60)

    sources = {
        "Logistic\nRegression":    ("results/baselines/lr_results.json",         "metrics.Brier"),
        "Ensemble RF\n(0h)":       ("results/phase6/2_calibration_metrics.json", "brier_before"),
        "Ensemble RF\n(30m)":      ("results/leadtime/leadtime_metrics.json",     "30m.Brier"),
        "Ensemble RF\n(60m)":      ("results/leadtime/leadtime_metrics.json",     "60m.Brier"),
    }

    brier_scores = {}
    for model_name, (path, key_path) in sources.items():
        data = load_json(path)
        if data is None:
            print(f"  ⚠️  {model_name}: file not found ({path})")
            continue
        # Drill into nested key path
        val = data
        for key in key_path.split("."):
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                val = None
                break
        if val is not None:
            brier_scores[model_name] = float(val)
            print(f"  {model_name.replace(chr(10),' ')}: {val:.6f}")
        else:
            print(f"  ⚠️  {model_name}: key '{key_path}' not found in {path}")

    if not brier_scores:
        print("No Brier scores loaded — run prerequisite scripts first")
        return

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    models  = list(brier_scores.keys())
    scores  = list(brier_scores.values())

    colors = ["#E74C3C", "#2ECC71", "#3498DB", "#9B59B6"][:len(models)]
    bars = ax.bar(range(len(models)), scores, color=colors, alpha=0.85, edgecolor="white", lw=1.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(scores) * 0.01,
                f"{score:.6f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_title("Brier Score Comparison Across Models", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(scores) * 1.2)

    # Add "lower is better" annotation
    ax.annotate("↓ Lower is better",
                xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", fontsize=10, color="gray",
                style="italic")

    # Find best model
    best_model = min(brier_scores, key=brier_scores.get)
    ax.annotate("★ Best", (list(brier_scores.keys()).index(best_model),
                            brier_scores[best_model]),
                xytext=(0, 25), textcoords="offset points",
                ha="center", fontsize=10, color="green",
                arrowprops=dict(arrowstyle="->", color="green"))

    plt.tight_layout()
    plt.savefig("results/scientific/brier_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()

    out = {
        "brier_scores": brier_scores,
        "best_model": best_model,
        "best_score": float(brier_scores[best_model]),
        "note": "Lower Brier Score = better probabilistic calibration"
    }
    with open("results/scientific/brier_scores.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n✅ Best: {best_model.replace(chr(10),' ')} = {brier_scores[best_model]:.6f}")
    print("✅ Saved → results/scientific/brier_comparison.png")
    print("✅ Saved → results/scientific/brier_scores.json")


if __name__ == "__main__":
    main()
