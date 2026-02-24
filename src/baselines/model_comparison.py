"""
Model Comparison Table
-----------------------
Loads results from:
  - Logistic Regression baseline    → results/baselines/lr_results.json
  - Phase 6 Ensemble RF             → results/phase6/1_time_split_metrics.json

Combines them into a publication-quality comparison table image and JSON.

Outputs → results/baselines/model_comparison_table.png
          results/baselines/model_comparison.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("results/baselines", exist_ok=True)
os.makedirs("results/scientific", exist_ok=True)


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    print("=" * 60)
    print("Model Comparison Table")
    print("=" * 60)

    # ── Load Logistic Regression results ─────────────────────────
    lr_data = load_json("results/baselines/lr_results.json")
    if lr_data is None:
        print("❌ LR results not found — run train_logistic_regression.py first")
        return

    lr_m = lr_data["metrics"]

    # ── Load Ensemble RF results from Phase 6 ────────────────────
    # Phase 6 split metrics: test set
    ts_data = load_json("results/phase6/1_time_split_metrics.json")
    ens_m = {}
    if ts_data and "test" in ts_data:
        ens_m = ts_data["test"]

    # ── Build comparison rows ─────────────────────────────────────
    rows = [
        {
            "Model": "Logistic Regression",
            "POD":    lr_m.get("POD", np.nan),
            "FAR":    lr_m.get("FAR", np.nan),
            "CSI":    lr_m.get("CSI", np.nan),
            "ROC-AUC": lr_m.get("ROC_AUC", np.nan),
            "PR-AUC": lr_m.get("PR_AUC", np.nan),
            "Brier":  lr_m.get("Brier", np.nan),
        },
        {
            "Model": "Ensemble RF (30-model)",
            "POD":    ens_m.get("POD", np.nan),
            "FAR":    ens_m.get("FAR", np.nan),
            "CSI":    ens_m.get("CSI", np.nan),
            "ROC-AUC": ens_m.get("roc_auc", np.nan),
            "PR-AUC": ens_m.get("pr_auc", np.nan),
            "Brier":  ens_m.get("brier", np.nan),
        },
    ]

    cmp_df = pd.DataFrame(rows)
    print("\nComparison Table:")
    print(cmp_df.to_string(index=False, float_format="%.4f"))

    # ── Compute improvement ───────────────────────────────────────
    if len(cmp_df) >= 2:
        improvements = {}
        for col in ["POD", "CSI", "ROC-AUC", "PR-AUC"]:
            lr_v  = cmp_df.loc[0, col]
            ens_v = cmp_df.loc[1, col]
            if pd.notna(lr_v) and pd.notna(ens_v) and lr_v != 0:
                improvements[col] = float((ens_v - lr_v) / abs(lr_v) * 100)
        # For Brier lower is better
        lr_b  = cmp_df.loc[0, "Brier"]
        ens_b = cmp_df.loc[1, "Brier"]
        if pd.notna(lr_b) and lr_b != 0:
            improvements["Brier_reduction_pct"] = float((lr_b - ens_b) / lr_b * 100)

        print(f"\nEnsemble RF improvement over LR baseline:")
        for k, v in improvements.items():
            print(f"  {k}: +{v:.2f}%") if v >= 0 else print(f"  {k}: {v:.2f}%")

    # ── Plot table ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison: Logistic Regression vs Ensemble RF",
                 fontsize=13, fontweight="bold")

    # Table
    ax_t = axes[0]
    ax_t.axis("off")
    table_data = []
    for _, row in cmp_df.iterrows():
        table_data.append([
            row["Model"],
            f"{row['POD']:.4f}" if pd.notna(row['POD']) else "—",
            f"{row['FAR']:.4f}" if pd.notna(row['FAR']) else "—",
            f"{row['CSI']:.4f}" if pd.notna(row['CSI']) else "—",
            f"{row['ROC-AUC']:.4f}" if pd.notna(row['ROC-AUC']) else "—",
            f"{row['PR-AUC']:.4f}" if pd.notna(row['PR-AUC']) else "—",
            f"{row['Brier']:.6f}" if pd.notna(row['Brier']) else "—",
        ])
    col_labels = ["Model", "POD↑", "FAR↓", "CSI↑", "ROC-AUC↑", "PR-AUC↑", "Brier↓"]
    tbl = ax_t.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)
    # Color header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Color rows
    row_colors = ["#FADBD8", "#D5F5E3"]
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(row_colors[(i-1) % len(row_colors)])
    ax_t.set_title("Metric Comparison Table", fontweight="bold", pad=10)

    # Bar chart: key metrics
    ax_b = axes[1]
    metrics_to_plot = ["POD", "CSI", "PR-AUC", "Brier"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    colors_lr  = "#E74C3C"
    colors_ens = "#2ECC71"

    for i, met in enumerate(metrics_to_plot):
        lr_v  = cmp_df.loc[0, met] if pd.notna(cmp_df.loc[0, met]) else 0
        ens_v = cmp_df.loc[1, met] if pd.notna(cmp_df.loc[1, met]) else 0
        ax_b.bar(x[i] - width/2, lr_v,  width, color=colors_lr,  alpha=0.85,
                 label="LR" if i == 0 else "")
        ax_b.bar(x[i] + width/2, ens_v, width, color=colors_ens, alpha=0.85,
                 label="Ensemble RF" if i == 0 else "")
        ax_b.text(x[i] - width/2, lr_v  + 0.002, f"{lr_v:.3f}",  ha="center", fontsize=8)
        ax_b.text(x[i] + width/2, ens_v + 0.002, f"{ens_v:.3f}", ha="center", fontsize=8)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(metrics_to_plot)
    ax_b.set_ylabel("Score")
    ax_b.set_title("Key Metric Comparison\n(Brier: lower is better)")
    ax_b.legend()
    ax_b.grid(True, axis="y", alpha=0.3)
    ax_b.set_ylim(0, max(ax_b.get_ylim()[1], 1.05))

    plt.tight_layout()
    plt.savefig("results/baselines/model_comparison_table.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── Save JSON ─────────────────────────────────────────────────
    out = {
        "comparison": cmp_df.to_dict(orient="records"),
        "improvements_pct": improvements if len(cmp_df) >= 2 else {}
    }
    with open("results/baselines/model_comparison.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n✅ Saved → results/baselines/model_comparison_table.png")
    print("✅ Saved → results/baselines/model_comparison.json")


if __name__ == "__main__":
    main()
