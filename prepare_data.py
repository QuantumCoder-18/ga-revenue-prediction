# Phase 3: Cleaning and preparing the data for machine learning
# We take the raw messy data and turn it into clean numbers the model can learn from

import pandas as pd
import numpy as np
import json

# ── Load the data (same as Phase 2) ──────────────────────────────────────────

print("Step 1: Loading the dataset...")

df = pd.read_csv("data/train_v2.csv", dtype=str, low_memory=False, nrows=100_000)

# Flatten the JSON columns — same as your working explore_data.py did
# Some columns like "device" and "totals" contain JSON text inside each cell
# We need to unpack them into proper separate columns

print("Step 2: Flattening JSON columns...")

json_cols = ["device", "geoNetwork", "totals", "trafficSource"]

for col in json_cols:
    if col not in df.columns:
        continue
    # Parse each cell's JSON text into a Python dictionary
    # then expand all those dictionaries into new columns
    try:
        expanded = df[col].apply(
            lambda x: json.loads(x) if isinstance(x, str) else {}
        )
        expanded_df = pd.json_normalize(expanded)
        # Rename new columns to "originalcol.newcol" format
        expanded_df.columns = [f"{col}.{c}" for c in expanded_df.columns]
        # Drop the original JSON column, add the new flat columns
        df = pd.concat([df.drop(columns=[col]), expanded_df], axis=1)
    except Exception as e:
        print(f"  Could not flatten {col}: {e}")

print(f"  Columns after flattening: {df.shape[1]}")

# ── Step A: Fix missing values ────────────────────────────────────────────────

print("\nStep 3: Fixing missing values...")

# Revenue: blank means no purchase — fill with 0
# This is the column we want to PREDICT (our target)
revenue_col = "totals.transactionRevenue"
if revenue_col in df.columns:
    df[revenue_col] = pd.to_numeric(
        df[revenue_col], errors="coerce"
    ).fillna(0) / 1_000_000   # convert from micro-dollars to real dollars
    print(f"  Revenue column: filled {(df[revenue_col] == 0).sum():,} blanks with 0")

# Pageviews, hits: blank means 0 activity
for col in ["totals.pageviews", "totals.hits", "totals.bounces", "totals.newVisits"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Visit number: blank means first visit — fill with 1
if "visitNumber" in df.columns:
    df["visitNumber"] = pd.to_numeric(df["visitNumber"], errors="coerce").fillna(1)

# Text columns: blank means unknown — fill with the word "unknown"
text_cols = [
    "device.browser", "device.deviceCategory", "device.operatingSystem",
    "geoNetwork.country", "geoNetwork.city", "geoNetwork.continent",
    "trafficSource.source", "trafficSource.medium", "trafficSource.channelGrouping",
]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("unknown")

print("  Missing values fixed!")

# ── Step B: Convert text columns to numbers ───────────────────────────────────

print("\nStep 4: Converting text columns to numbers...")

# This technique is called Label Encoding
# Every unique text value gets a unique number
# Example: Chrome=0, Firefox=1, Safari=2, Edge=3
# The model can now do math on these

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
encoded_cols = []   # keep track of which columns we encoded

for col in text_cols:
    if col not in df.columns:
        continue
    # Create a NEW column with "_encoded" at the end
    # We keep the original text column too so we can still read it
    new_col = col.replace(".", "_") + "_encoded"
    df[new_col] = le.fit_transform(df[col].astype(str))
    encoded_cols.append(new_col)
    print(f"  {col:<45} → {new_col}")

# ── Step C: Build new features from existing columns ─────────────────────────

print("\nStep 5: Building new features...")

# Parse the date column — "20171016" becomes a proper date Python understands
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    # Extract useful time-based signals from the date
    # The model can learn "Monday visitors buy more" or "December has more revenue"
    df["day_of_week"] = df["date"].dt.dayofweek    # 0=Monday, 6=Sunday
    df["month"]       = df["date"].dt.month         # 1=January, 12=December
    df["day_of_month"]= df["date"].dt.day           # 1 to 31
    df["is_weekend"]  = (df["date"].dt.dayofweek >= 5).astype(int)  # 1 if Sat/Sun

    print("  Created: day_of_week, month, day_of_month, is_weekend")

# Average revenue per country — countries like USA tend to spend more
# This gives the model a signal about how valuable each country typically is
if "geoNetwork.country" in df.columns and revenue_col in df.columns:
    country_avg = (
        df.groupby("geoNetwork.country")[revenue_col]
        .mean()
        .rename("country_avg_revenue")
    )
    df = df.join(country_avg, on="geoNetwork.country")
    df["country_avg_revenue"] = df["country_avg_revenue"].fillna(0)
    print("  Created: country_avg_revenue")

# Average revenue per channel — organic search vs paid ads vs social
channel_col = "trafficSource.channelGrouping"
if channel_col in df.columns and revenue_col in df.columns:
    channel_avg = (
        df.groupby(channel_col)[revenue_col]
        .mean()
        .rename("channel_avg_revenue")
    )
    df = df.join(channel_avg, on=channel_col)
    df["channel_avg_revenue"] = df["channel_avg_revenue"].fillna(0)
    print("  Created: channel_avg_revenue")

# ── Final summary ─────────────────────────────────────────────────────────────

print("\n--- FINAL DATASET SUMMARY ---")
print(f"  Rows:    {df.shape[0]:,}")
print(f"  Columns: {df.shape[1]}")

# Show the numeric feature columns we'll feed into the model
feature_cols = [
    "totals.pageviews", "totals.hits", "totals.bounces",
    "totals.newVisits", "visitNumber",
    "day_of_week", "month", "day_of_month", "is_weekend",
    "country_avg_revenue", "channel_avg_revenue",
] + encoded_cols

available = [c for c in feature_cols if c in df.columns]

print(f"\n  Features ready for ML model: {len(available)}")
for f in available:
    print(f"    {f}")

# Save the cleaned dataset so we can use it in Phase 4
print("\nStep 6: Saving cleaned data...")
df.to_csv("data/clean_data.csv", index=False)
print("  Saved to data/clean_data.csv")

print("\n✅ Phase 3 complete!")
print("   The data is clean, encoded, and ready for LightGBM.")
print("   Next: Phase 4 — train the ML model!")