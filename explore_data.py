import pandas as pd
import numpy as np
import json
from pandas import json_normalize

# ── Part 1: Load the data ─────────────────────────────────────────────────────

print("Loading dataset (sample)...")

df = pd.read_csv("data/train_v2.csv", nrows=100000)

print("Dataset loaded successfully!")

# ── Part 2: Convert JSON columns ──────────────────────────────────────────────

json_cols = ["device", "geoNetwork", "totals", "trafficSource"]

for col in json_cols:
    df[col] = df[col].apply(lambda x: json.loads(x) if pd.notnull(x) else {})

# Flatten JSON columns
device_df = json_normalize(df["device"])
geo_df = json_normalize(df["geoNetwork"])
totals_df = json_normalize(df["totals"])
traffic_df = json_normalize(df["trafficSource"])

# Merge all
df = pd.concat([df, device_df, geo_df, totals_df, traffic_df], axis=1)

print("JSON columns flattened!")

# ── Part 3: Basic info ────────────────────────────────────────────────────────

print("\n--- BASIC INFO ---")
print(f"Rows:    {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

# ── Part 4: First rows ────────────────────────────────────────────────────────

print("\n--- FIRST 3 ROWS ---")

cols = [
    "date", "visitNumber",
    "pageviews", "transactionRevenue",
    "deviceCategory", "country", "source"
]

existing = [c for c in cols if c in df.columns]
print(df[existing].head(3).to_string())

# ── Part 5: Missing values ────────────────────────────────────────────────────

print("\n--- MISSING VALUES ---")

missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0].head(10)

for col, count in missing.items():
    pct = count / len(df) * 100
    print(f"{col:<40} {count:>7,} ({pct:.1f}%)")

# ── Part 6: Revenue analysis ──────────────────────────────────────────────────

print("\n--- REVENUE ANALYSIS ---")

# Fix revenue column
df["transactionRevenue"] = pd.to_numeric(
    df["transactionRevenue"], errors="coerce"
).fillna(0) / 1_000_000

total_revenue = df["transactionRevenue"].sum()
buyers = (df["transactionRevenue"] > 0).sum()
total_visitors = len(df)
conversion_rate = buyers / total_visitors * 100

print(f"Total revenue:      ${total_revenue:,.2f}")
print(f"Buyers:             {buyers:,}")
print(f"Visitors:           {total_visitors:,}")
print(f"Conversion rate:    {conversion_rate:.2f}%")

# ── Part 7: Top countries ─────────────────────────────────────────────────────

print("\n--- TOP COUNTRIES ---")

if "country" in df.columns:
    top_countries = df["country"].value_counts().head(5)
    print(top_countries)

# ── Part 8: Device split ──────────────────────────────────────────────────────

print("\n--- DEVICE SPLIT ---")

if "deviceCategory" in df.columns:
    print(df["deviceCategory"].value_counts(normalize=True) * 100)

# ── Part 9: Traffic source ────────────────────────────────────────────────────

print("\n--- TRAFFIC SOURCE ---")

if "source" in df.columns:
    print(df["source"].value_counts().head(5))

print("\n✅ Phase 2 complete!")