"""
================================================================
model.py — Rent Price Prediction Model
================================================================
PURPOSE:
    Build a Random Forest model that predicts average monthly
    rent for any UK region based on economic features.

WHY RANDOM FOREST?
    - Handles non-linear relationships well (rents don't grow
      in a perfectly straight line)
    - Works with small datasets (140 rows is fine)
    - Gives feature importance — tells us WHAT drives rents
    - Hard to overfit compared to a single decision tree
    - Explainable to non-technical interviewers

WHAT THE MODEL LEARNS:
    Input features:
        - Year
        - Avg House Price
        - Median Annual Wage
        - BOE Base Rate
        - Price to Income Ratio
        - Affordability Ratio
        - Rent to Price Yield

    Target (what we predict):
        - Avg Monthly Rent

MODEL OUTPUTS:
    1. Model performance metrics (R², MAE, RMSE)
    2. Feature importance chart
    3. Actual vs Predicted rent chart
    4. 2025-2027 rent forecasts by region
    5. Saved model summary to outputs/model_summary.txt
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

CHART_DIR = "outputs/charts"
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ================================================================
# STEP 1 — PREPARE FEATURES
# ================================================================

def prepare_features(df):
    """
    Select and prepare features for the model.

    FEATURE SELECTION RATIONALE:
        We use economic drivers that a real analyst would
        identify as the key determinants of rental prices:

        1. Year              — captures time trend
        2. Region (encoded)  — captures regional differences
        3. Avg_House_Price   — higher prices = higher rents
        4. Median_Annual_Wage — wages set the ceiling on rents
        5. BOE_Base_Rate     — mortgage costs pass through to rents
        6. Price_to_Income   — affordability pressure on buyers
        7. Affordability_Ratio — demand signal from renters
        8. Rent_to_Price_Yield — landlord profitability signal

    LABEL ENCODING:
        Machine learning models need numbers, not text.
        We convert Region names to numbers using LabelEncoder.
        e.g. "London" → 4, "North East" → 6
    """

    print("\n[1/5] Preparing features...")

    # Encode region as a number
    le = LabelEncoder()
    df = df.copy()
    df["Region_Encoded"] = le.fit_transform(df["Region"])

    # Define features (X) and target (y)
    feature_cols = [
        "Year",
        "Region_Encoded",
        "Avg_House_Price",
        "Median_Annual_Wage",
        "BOE_Base_Rate",
        "Price_to_Income_Ratio",
        "Affordability_Ratio",
        "Rent_to_Price_Yield",
    ]

    X = df[feature_cols].copy()
    y = df["Avg_Monthly_Rent"].copy()

    # Drop rows where target or features have NaN
    # (Rent_Growth_YoY is NaN for 2011 — first year has no prior year)
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X[mask]
    y = y[mask]
    df_clean = df[mask].copy()

    print(f"    Features: {list(feature_cols)}")
    print(f"    Training samples: {len(X)}")
    print(f"    Target range: £{y.min():,.0f} — £{y.max():,.0f}")

    return X, y, df_clean, le, feature_cols


# ================================================================
# STEP 2 — TRAIN THE MODEL
# ================================================================

def train_model(X, y):
    """
    Split data, train Random Forest, evaluate performance.

    TRAIN/TEST SPLIT:
        80% of data trains the model, 20% tests it.
        We use random_state=42 for reproducibility.

    CROSS VALIDATION:
        We also run 5-fold cross validation — this splits the
        data 5 different ways and averages the score.
        This gives a more reliable accuracy estimate than a
        single train/test split on a small dataset.

    HYPERPARAMETERS EXPLAINED:
        n_estimators=200  — build 200 decision trees and average them
        max_depth=8       — limit tree depth to prevent overfitting
        min_samples_leaf=3 — each leaf needs at least 3 samples
        random_state=42   — reproducible results
    """

    print("\n[2/5] Training Random Forest model...")

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"    Training set: {len(X_train)} samples")
    print(f"    Test set:     {len(X_test)} samples")

    # Initialise and train the model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1        # Use all CPU cores
    )

    model.fit(X_train, y_train)
    print("    Model trained ✅")

    # ── Evaluate on test set ──────────────────────────────────
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n    Test Set Performance:")
    print(f"      R² Score:  {r2:.4f}  (1.0 = perfect)")
    print(f"      MAE:       £{mae:,.0f}  (avg prediction error)")
    print(f"      RMSE:      £{rmse:,.0f} (penalises large errors)")

    # ── Cross Validation ─────────────────────────────────────
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"\n    5-Fold Cross Validation R²:")
    print(f"      Scores: {[round(s, 3) for s in cv_scores]}")
    print(f"      Mean:   {cv_scores.mean():.4f}")
    print(f"      Std:    {cv_scores.std():.4f}")

    metrics = {
        "r2": r2, "mae": mae, "rmse": rmse,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()
    }

    return model, X_train, X_test, y_train, y_test, y_pred, metrics


# ================================================================
# STEP 3 — FEATURE IMPORTANCE CHART
# ================================================================

def chart_feature_importance(model, feature_cols):
    """
    Bar chart showing which features the model relies on most.

    WHAT THIS TELLS YOU IN AN INTERVIEW:
        Feature importance reveals the economic drivers of rent.
        If 'Median_Annual_Wage' is top — rents are wage-driven.
        If 'BOE_Base_Rate' is top — rents are rate-driven.
        If 'Avg_House_Price' is top — rents follow the property market.

        This is more insightful than just saying
        "my model is 92% accurate."
    """

    print("\n[3/5] Generating feature importance chart...")

    importance = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    # Clean up feature names for display
    name_map = {
        "Year":                  "Year (Time Trend)",
        "Region_Encoded":        "Region",
        "Avg_House_Price":       "Avg House Price",
        "Median_Annual_Wage":    "Median Annual Wage",
        "BOE_Base_Rate":         "BOE Base Rate",
        "Price_to_Income_Ratio": "Price to Income Ratio",
        "Affordability_Ratio":   "Affordability Ratio",
        "Rent_to_Price_Yield":   "Rent to Price Yield",
    }
    importance["Feature"] = importance["Feature"].map(name_map)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#d62728" if x == importance["Importance"].max()
              else "#1f77b4" for x in importance["Importance"]]

    bars = ax.barh(importance["Feature"], importance["Importance"],
                   color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, importance["Importance"]):
        ax.text(bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    ax.set_title("Random Forest — Feature Importance\n"
                 "What drives UK rental prices the most?",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_xlim(0, importance["Importance"].max() + 0.08)

    fig.text(0.99, 0.01, "Model: Random Forest Regressor | 200 trees",
             ha="right", fontsize=8, color="grey")

    plt.tight_layout()
    path = f"{CHART_DIR}/07_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ================================================================
# STEP 4 — ACTUAL VS PREDICTED CHART
# ================================================================

def chart_actual_vs_predicted(y_test, y_pred, metrics):
    """
    Scatter plot comparing actual rents vs model predictions.

    WHAT TO LOOK FOR:
        Points close to the diagonal line = accurate predictions.
        Points far from the line = model errors.
        A good model has points tightly clustered around the line.
    """

    print("\n[4/5] Generating actual vs predicted chart...")

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(y_test, y_pred, alpha=0.6, color="#1f77b4",
               edgecolors="white", s=60, label="Predictions")

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            "r--", linewidth=2, label="Perfect prediction")

    # Format axes as £
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"£{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"£{x:,.0f}"))

    # Add metrics annotation
    textstr = (f"R² = {metrics['r2']:.3f}\n"
               f"MAE = £{metrics['mae']:,.0f}\n"
               f"RMSE = £{metrics['rmse']:,.0f}")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_title("Actual vs Predicted Monthly Rent\n"
                 "Random Forest Model — Test Set",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Actual Monthly Rent (£)", fontsize=12)
    ax.set_ylabel("Predicted Monthly Rent (£)", fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = f"{CHART_DIR}/08_actual_vs_predicted.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ================================================================
# STEP 5 — FORECAST 2025-2027
# ================================================================

def forecast_future_rents(model, df, le, feature_cols):
    """
    Use the trained model to forecast rents for 2025, 2026, 2027.

    FORECASTING ASSUMPTIONS:
        - House prices grow at 3% per year (moderate recovery)
        - Wages grow at 4% per year (post-inflation catch-up)
        - BOE rate falls to 3.5% by 2027 (market consensus 2024)
        - Affordability and yield ratios recalculated each year

    IMPORTANT CAVEAT (say this in interviews):
        "The model extrapolates from historical patterns.
         Real forecasts would incorporate macroeconomic
         scenarios, policy changes, and expert judgement."
    """

    print("\n[5/5] Forecasting rents for 2025-2027...")

    # Get 2024 data as baseline
    base = df[df["Year"] == 2024].copy()

    forecast_years = [2025, 2026, 2027]

    # Assumptions for future years
    assumptions = {
        2025: {"price_growth": 0.03, "wage_growth": 0.04, "boe": 4.25},
        2026: {"price_growth": 0.03, "wage_growth": 0.04, "boe": 3.75},
        2027: {"price_growth": 0.04, "wage_growth": 0.04, "boe": 3.50},
    }

    all_forecasts = []
    prev = base.copy()

    for year in forecast_years:
        curr = prev.copy()
        curr["Year"] = year
        a = assumptions[year]

        curr["Avg_House_Price"]    = prev["Avg_House_Price"]    * (1 + a["price_growth"])
        curr["Median_Annual_Wage"] = prev["Median_Annual_Wage"] * (1 + a["wage_growth"])
        curr["BOE_Base_Rate"]      = a["boe"]

        # Recalculate derived features
        curr["Price_to_Income_Ratio"] = (
            curr["Avg_House_Price"] / curr["Median_Annual_Wage"]
        )
        curr["Affordability_Ratio"] = (
            (curr["Avg_Monthly_Rent"] * 12) / curr["Median_Annual_Wage"] * 100
        )
        curr["Rent_to_Price_Yield"] = (
            (curr["Avg_Monthly_Rent"] * 12) / curr["Avg_House_Price"] * 100
        )

        # Predict
        X_future = curr[feature_cols]
        curr["Predicted_Rent"] = model.predict(X_future).round(0)

        all_forecasts.append(curr[["Year","Region","Predicted_Rent"]])
        prev = curr
        prev["Avg_Monthly_Rent"] = curr["Predicted_Rent"]

    forecast_df = pd.concat(all_forecasts)

    # Print forecast table
    print("\n    Rent Forecast by Region (£/month):\n")
    pivot = forecast_df.pivot(
        index="Region", columns="Year", values="Predicted_Rent"
    ).astype(int)

    # Add 2024 actual for comparison
    actual_2024 = df[df["Year"]==2024].set_index("Region")["Avg_Monthly_Rent"].round(0).astype(int)
    pivot.insert(0, "2024 Actual", actual_2024)
    print(pivot.to_string())

    # Save forecast chart
    fig, ax = plt.subplots(figsize=(12, 7))

    regions = sorted(forecast_df["Region"].unique())
    colors  = plt.cm.tab10(np.linspace(0, 1, len(regions)))

    for i, region in enumerate(regions):
        # Historical line
        hist = df[df["Region"] == region].sort_values("Year")
        ax.plot(hist["Year"], hist["Avg_Monthly_Rent"],
                color=colors[i], linewidth=2, alpha=0.8)

        # Forecast line (dashed)
        fc = forecast_df[forecast_df["Region"] == region]
        # Connect from 2024 actual
        last_actual = hist[hist["Year"]==2024]["Avg_Monthly_Rent"].values[0]
        fc_years    = [2024] + list(fc["Year"])
        fc_values   = [last_actual] + list(fc["Predicted_Rent"])

        ax.plot(fc_years, fc_values, color=colors[i],
                linewidth=2, linestyle="--", alpha=0.8,
                label=region)

    # Shade forecast period
    ax.axvspan(2024.5, 2027, alpha=0.06, color="grey")
    ax.text(2025.2, ax.get_ylim()[1]*0.97, "Forecast\n2025–2027",
            fontsize=9, color="grey", ha="center")

    ax.axvline(x=2024.5, color="grey", linestyle=":", linewidth=1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"£{x:,.0f}"))
    ax.set_xticks(list(range(2011, 2028)))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title("UK Regional Rent Forecast (2025–2027)\n"
                 "Solid = Historical | Dashed = Model Forecast",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Average Monthly Rent (£)", fontsize=12)

    fig.text(0.99, 0.01,
             "Model: Random Forest | Assumptions: +3% house prices, +4% wages, BOE falling to 3.5%",
             ha="right", fontsize=7.5, color="grey")

    plt.tight_layout()
    path = f"{CHART_DIR}/09_rent_forecast_2027.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n    Saved: {path}")

    return forecast_df, pivot


# ================================================================
# STEP 6 — SAVE MODEL SUMMARY
# ================================================================

def save_model_summary(metrics, pivot):
    """Save a text summary of model performance and forecasts."""

    path = "outputs/model_summary.txt"
    with open(path, "w") as f:
        f.write("=" * 55 + "\n")
        f.write("  UK RENTAL ANALYSIS — Model Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write("MODEL: Random Forest Regressor\n")
        f.write("TARGET: Average Monthly Rent (£)\n\n")
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  R² Score (test set):      {metrics['r2']:.4f}\n")
        f.write(f"  MAE (test set):           £{metrics['mae']:,.0f}\n")
        f.write(f"  RMSE (test set):          £{metrics['rmse']:,.0f}\n")
        f.write(f"  CV R² Mean (5-fold):      {metrics['cv_mean']:.4f}\n")
        f.write(f"  CV R² Std (5-fold):       {metrics['cv_std']:.4f}\n\n")
        f.write("RENT FORECAST (£/month):\n")
        f.write(pivot.to_string())
        f.write("\n\nASSUMPTIONS:\n")
        f.write("  House price growth: +3-4% per year\n")
        f.write("  Wage growth: +4% per year\n")
        f.write("  BOE Base Rate: falling to 3.50% by 2027\n")

    print(f"\n    Model summary saved to: {path}")


# ================================================================
# MASTER FUNCTION
# ================================================================

def train_and_evaluate(df=None):
    """Run the full modelling pipeline."""

    print("=" * 55)
    print("  UK RENTAL ANALYSIS — Prediction Model")
    print("=" * 55)

    if df is None:
        df = pd.read_csv("data/processed/master.csv")

    X, y, df_clean, le, feature_cols = prepare_features(df)
    model, X_train, X_test, y_train, y_test, y_pred, metrics = train_model(X, y)
    chart_feature_importance(model, feature_cols)
    chart_actual_vs_predicted(y_test, y_pred, metrics)
    forecast_df, pivot = forecast_future_rents(model, df_clean, le, feature_cols)
    save_model_summary(metrics, pivot)

    print("\n" + "-" * 55)
    print("  Modelling complete!")
    print(f"  R² Score:  {metrics['r2']:.4f}")
    print(f"  MAE:       £{metrics['mae']:,.0f}/month")
    print(f"  CV Mean:   {metrics['cv_mean']:.4f}")
    print("-" * 55)

    return model, metrics, forecast_df


if __name__ == "__main__":
    train_and_evaluate()
    