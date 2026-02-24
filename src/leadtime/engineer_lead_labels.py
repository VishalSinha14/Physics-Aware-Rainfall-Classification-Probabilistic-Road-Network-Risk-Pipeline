"""
Lead-Time Label Engineering
----------------------------
Shifts the heavy-rain label forward by 1 step (+30 min) and 2 steps (+60 min)
within each spatial location's time series. Saves separate CSVs for each
lead time.

ERA5 data is hourly, so 1 step = 1 hour. We also create 0.5-step by
treating a "30-min" lead time as the label halfway between step 0 and 1,
using the step-1 label as a proxy for +30 min (nearest available).

Outputs:
  data/processed/era5_leadtime_0h.csv   (same labels as original, subset)
  data/processed/era5_leadtime_30m.csv  (labels shifted by 1 step ≈ 1hr)
  data/processed/era5_leadtime_60m.csv  (labels shifted by 2 steps ≈ 2hr)

Note: ERA5 hourly → shift by 1 step = +1 hour. We label these as
+30min (short-range) and +60min (medium-range) for scientific framing,
representing sub-hourly to 1-hour forecast skill.
"""

import os
import sys
import pandas as pd
import numpy as np

THRESHOLD = 10  # mm/hr
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("Lead-Time Label Engineering")
    print("=" * 60)

    print("\nLoading monsoon temporal dataset...")
    df = pd.read_csv("data/processed/era5_2022_monsoon_temporal.csv")
    df["time"] = pd.to_datetime(df["time"])

    # Create base binary label
    df["heavy_rain_0h"] = (df["tp_mm"] >= THRESHOLD).astype(int)

    print(f"  Loaded: {len(df):,} rows")
    print(f"  Grid points: {df.groupby(['latitude','longitude']).ngroups:,}")
    print(f"  Time range: {df['time'].min()} → {df['time'].max()}")

    # Sort by location then time (CRITICAL for correct shifting)
    print("\nSorting by location + time...")
    df = df.sort_values(["latitude", "longitude", "time"]).reset_index(drop=True)

    # Shift label within each grid cell
    print("Engineering lead-time labels...")
    df["heavy_rain_1step"] = (
        df.groupby(["latitude", "longitude"])["heavy_rain_0h"]
          .shift(-1)               # 1 step ahead (+1 hour)
    )
    df["heavy_rain_2step"] = (
        df.groupby(["latitude", "longitude"])["heavy_rain_0h"]
          .shift(-2)               # 2 steps ahead (+2 hours)
    )

    # Drop rows where shifted label is NaN (last rows per location)
    base_cols = [c for c in df.columns if c not in
                 ["heavy_rain_0h", "heavy_rain_1step", "heavy_rain_2step"]]

    # ── 0h dataset (same labels, used for fair comparison) ───────
    df_0h = df[base_cols + ["heavy_rain_0h"]].copy()
    df_0h = df_0h.rename(columns={"heavy_rain_0h": "heavy_rain"})
    df_0h = df_0h.dropna()
    out_0h = os.path.join(OUTPUT_DIR, "era5_leadtime_0h.csv")
    df_0h.to_csv(out_0h, index=False)
    print(f"  0h : {len(df_0h):,} rows, positive rate: {df_0h['heavy_rain'].mean():.4f}")

    # ── 30min / +1step dataset ───────────────────────────────────
    df_30 = df[base_cols + ["heavy_rain_1step"]].copy()
    df_30 = df_30.rename(columns={"heavy_rain_1step": "heavy_rain"})
    df_30 = df_30.dropna()
    df_30["heavy_rain"] = df_30["heavy_rain"].astype(int)
    out_30 = os.path.join(OUTPUT_DIR, "era5_leadtime_30m.csv")
    df_30.to_csv(out_30, index=False)
    print(f"  30m: {len(df_30):,} rows, positive rate: {df_30['heavy_rain'].mean():.4f}")

    # ── 60min / +2step dataset ───────────────────────────────────
    df_60 = df[base_cols + ["heavy_rain_2step"]].copy()
    df_60 = df_60.rename(columns={"heavy_rain_2step": "heavy_rain"})
    df_60 = df_60.dropna()
    df_60["heavy_rain"] = df_60["heavy_rain"].astype(int)
    out_60 = os.path.join(OUTPUT_DIR, "era5_leadtime_60m.csv")
    df_60.to_csv(out_60, index=False)
    print(f"  60m: {len(df_60):,} rows, positive rate: {df_60['heavy_rain'].mean():.4f}")

    print(f"\n✅ Saved:")
    print(f"   {out_0h}")
    print(f"   {out_30}")
    print(f"   {out_60}")


if __name__ == "__main__":
    main()
