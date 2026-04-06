# Phase 4: Training the LightGBM revenue prediction model
# We take our clean data from Phase 3 and teach a model to predict revenue

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# ── Step 1: Load the clean data from Phase 3 ─────────────────────────────────

print("Step 1: Loading clean data from Phase 3...")

df = pd.read_csv("data/clean_data.csv", low_memory=False)
print(f"  Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")

# ── Step 2: Define features and target ───────────────────────────────────────

print("\nStep 2: Selecting features and target...")

# The TARGET is what we want to predict — revenue
# We apply log1p() which means log(1 + revenue)
# Why? Because revenue ranges from $0 to $500,000+
# That huge range makes it hard for the model to learn
# log1p squashes it into a much smaller range (0 to ~13)
# We reverse this later with np.expm1() to get real dollar amounts back

target_col = "totals.transactionRevenue"
df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
df["target"] = np.log1p(df[target_col])   # log transform the revenue

# FEATURES are the columns the model uses to make predictions
# These are the clean numeric columns we built in Phase 3
feature_cols = [
    "totals.pageviews",       # how many pages they viewed
    "totals.hits",            # total interactions on the site
    "totals.bounces",         # did they leave after 1 page?
    "totals.newVisits",       # is this their first ever visit?
    "visitNumber",            # which visit number is this for them?
    "day_of_week",            # 0=Monday ... 6=Sunday
    "month",                  # 1=January ... 12=December
    "day_of_month",           # 1 to 31
    "is_weekend",             # 1 if Saturday or Sunday, else 0
    "country_avg_revenue",    # average revenue for this country
    "channel_avg_revenue",    # average revenue for this traffic channel
]

# Also include our encoded text columns
encoded_cols = [c for c in df.columns if c.endswith("_encoded")]
feature_cols += encoded_cols

# Keep only features that actually exist in our dataframe
feature_cols = [c for c in feature_cols if c in df.columns]

print(f"  Target column : {target_col}")
print(f"  Features used : {len(feature_cols)}")
for f in feature_cols:
    print(f"    {f}")

# ── Step 3: Split into training and test sets ─────────────────────────────────

print("\nStep 3: Splitting data into train and test sets...")

X = df[feature_cols].fillna(0)   # features — fillna(0) handles any remaining blanks
y = df["target"]                  # target — log1p(revenue)

# train_test_split divides the data:
# 80% goes to training   — the model learns from this
# 20% goes to testing    — we hide this and use it to check accuracy
# random_state=42 means we get the same split every time we run it
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Training rows : {len(X_train):,}  (80%)")
print(f"  Test rows     : {len(X_test):,}  (20%)")

# ── Step 4: Train the LightGBM model ─────────────────────────────────────────

print("\nStep 4: Training the LightGBM model...")
print("  (This may take 30-60 seconds...)")

# These are the model's settings — called hyperparameters
# n_estimators: how many decision trees to build (200 is a good start)
# learning_rate: how fast the model learns (smaller = more careful)
# num_leaves: complexity of each tree (31 is the default)
# random_state: makes results reproducible
model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1        # -1 means don't print training progress (keeps output clean)
)

# .fit() is where actual learning happens
# We show it the training features and the correct answers
model.fit(X_train, y_train)

print("  Model trained successfully!")

# ── Step 5: Measure accuracy ──────────────────────────────────────────────────

print("\nStep 5: Measuring model accuracy...")

# Ask the model to predict revenue for the TEST set
# Remember — it has never seen these rows before
y_pred = model.predict(X_test)

# Calculate RMSE — Root Mean Squared Error
# Lower is better. It tells us how far off our predictions are on average
# We're working in log scale so 1.6 is actually quite good
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculate R² Score — how much of the revenue variation the model explains
# 1.0 = perfect, 0.0 = no better than guessing the average
# 0.30+ is considered reasonable for this dataset
r2 = r2_score(y_test, y_pred)

print(f"  RMSE     : {rmse:.4f}  (lower is better)")
print(f"  R² Score : {r2:.4f}  (higher is better, max = 1.0)")

# Show some example predictions vs real values
print("\n  Sample predictions vs actual revenue:")
print(f"  {'Predicted ($)':>15}  {'Actual ($)':>12}")
print(f"  {'-'*15}  {'-'*12}")

# Convert back from log scale to real dollars using expm1
sample_preds   = np.expm1(y_pred[:8])
sample_actuals = np.expm1(y_test.values[:8])

for pred, actual in zip(sample_preds, sample_actuals):
    print(f"  ${pred:>13,.2f}  ${actual:>10,.2f}")

# ── Step 6: Feature importance ────────────────────────────────────────────────

print("\nStep 6: Feature importance — what did the model find most useful?")
print("  (Higher number = more important for predicting revenue)")
print()

importance = pd.DataFrame({
    "feature"   : feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

# Show top 10 most important features
for _, row in importance.head(10).iterrows():
    bar_len = int(row["importance"] / importance["importance"].max() * 30)
    bar = "█" * bar_len
    print(f"  {row['feature']:<45} {bar} {row['importance']:.0f}")

# ── Step 7: Save the model ────────────────────────────────────────────────────

print("\nStep 7: Saving the model...")

import joblib
os.makedirs("models", exist_ok=True)
joblib.dump((model, feature_cols), "models/lgbm_model.pkl")
print("  Model saved to models/lgbm_model.pkl")
print("  We can load this in Phase 5 without retraining every time")

print("\n✅ Phase 4 complete!")
print("   You have trained a real machine learning model!")
print("   Next: Phase 5 — build the Streamlit dashboard!")