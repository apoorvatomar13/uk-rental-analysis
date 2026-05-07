"""
================================================================
preprocessing.py — Data Cleaning, Merging & Feature Engineering
================================================================
PURPOSE:
    Take the 4 raw CSV files and transform them into one clean
    master dataset ready for analysis and modelling.

WHAT THIS SCRIPT DOES:
    1. Loads all 4 raw datasets
    2. Merges them into one master table on Year + Region
    3. Cleans and validates the data
    4. Engineers 6 key financial features:
         - Affordability Ratio
         - Rent to Price Yield
         - Real Rent Growth
         - Price to Income Ratio
         - Monthly Mortgage Cost
         - Rent vs Mortgage Gap
    5. Saves master dataset to data/processed/master.csv

WHY FEATURE ENGINEERING?
    Raw numbers alone don't tell the story. A rent of £1,500/month
    means very different things in London vs the North East.
    Ratios and derived metrics give us CONTEXT — and that's what
    analysts, economists and interviewers care about.
================================================================
"""

import pandas as pd
import numpy as np
import os

# ================================================================
# STEP 1 — LOAD ALL RAW DATASETS
# ================================================================

def load_all_data():
    """
    Load all 4 raw CSV files into DataFrames.
    """
    print("\n[1/4] Loading raw datasets...")

    df_rent  = pd.read_csv("data/raw/rental_data.csv")
    df_hpi   = pd.read_csv("data/raw/house_prices.csv")
    df_boe   = pd.read_csv("data/raw/boe_base_rate.csv")
    df_wages = pd.read_csv("data/raw/wage_data.csv")

    print(f"    Rental data:      {df_rent.shape}")
    print(f"    House price data: {df_hpi.shape}")
    print(f"    BOE rate data:    {df_boe.shape}")
    print(f"    Wage data:        {df_wages.shape}")

    return df_rent, df_hpi, df_boe, df_wages


# ================================================================
# STEP 2 — MERGE INTO ONE MASTER TABLE
# ================================================================

def merge_datasets(df_rent, df_hpi, df_boe, df_wages):
    """
    Merge all 4 datasets into one master DataFrame.

    HOW THE MERGE WORKS:
        - df_rent and df_hpi both have Year + Region columns
          so we merge them on both keys (like a SQL JOIN)
        - df_wages also has Year + Region so we add that too
        - df_boe only has Year (one rate per year, all regions)
          so we merge on Year only — every region gets the same
          BOE rate for that year (which is correct — it's national)

    RESULT:
        One row per Region per Year with all metrics together.
        10 regions × 14 years = 140 rows
    """

    print("\n[2/4] Merging datasets...")

    # Merge rent + house prices on Year AND Region
    df = pd.merge(df_rent, df_hpi, on=["Year", "Region"], how="inner")

    # Add wage data
    df = pd.merge(df, df_wages, on=["Year", "Region"], how="inner")

    # Add BOE rate — joins on Year only (national rate)
    df = pd.merge(df, df_boe, on="Year", how="left")

    print(f"    Master table shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")

    return df


# ================================================================
# STEP 3 — CLEAN & VALIDATE
# ================================================================

def clean_data(df):
    """
    Clean the merged dataset.

    CHECKS:
        - No missing values
        - All numeric columns are correct type
        - No negative rents or prices
    """

    print("\n[3/4] Cleaning & validating...")

    # Check for missing values
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        print(f"    Filling {nulls} missing values with column median...")
        df = df.fillna(df.median(numeric_only=True))
    else:
        print("    No missing values found ✅")

    # Ensure correct data types
    numeric_cols = [
        "Avg_Monthly_Rent", "Avg_House_Price",
        "Median_Annual_Wage", "BOE_Base_Rate"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove any rows where rent or price is zero/negative
    before = len(df)
    df = df[(df["Avg_Monthly_Rent"] > 0) & (df["Avg_House_Price"] > 0)]
    after = len(df)
    if before != after:
        print(f"    Removed {before - after} invalid rows")

    # Sort by Region and Year for clean ordering
    df = df.sort_values(["Region", "Year"]).reset_index(drop=True)

    print(f"    Clean dataset: {df.shape[0]} rows, {df.shape[1]} columns ✅")

    return df


# ================================================================
# STEP 4 — FEATURE ENGINEERING
# ================================================================

def engineer_features(df):
    """
    Create 6 derived financial metrics that power the analysis.

    These are the metrics a real property analyst, economist, or
    FP&A analyst would calculate. Each one tells a different part
    of the rental crisis story.
    """

    print("\n[4/4] Engineering financial features...")

    # ── Feature 1: Affordability Ratio ───────────────────────
    # What % of gross annual income is spent on rent?
    # Formula: (Monthly Rent × 12) / Annual Wage × 100
    #
    # BENCHMARK: Above 30% = unaffordable (UN-Habitat standard)
    #            Above 40% = severely unaffordable
    # London has been above 40% for several years
    df["Affordability_Ratio"] = (
        (df["Avg_Monthly_Rent"] * 12) / df["Median_Annual_Wage"] * 100
    ).round(2)

    print("    ✅ Affordability Ratio (rent as % of income)")

    # ── Feature 2: Rent to Price Yield ───────────────────────
    # Annual rental income as % of property value
    # Formula: (Monthly Rent × 12) / House Price × 100
    #
    # WHY IT MATTERS: This is what landlords use to evaluate
    # whether buying to rent is profitable.
    # Typical UK gross yield: 4-7%
    # Below 4% = poor investment for landlords
    df["Rent_to_Price_Yield"] = (
        (df["Avg_Monthly_Rent"] * 12) / df["Avg_House_Price"] * 100
    ).round(2)

    print("    ✅ Rent to Price Yield (landlord return metric)")

    # ── Feature 3: Year-on-Year Rent Growth % ────────────────
    # How much did rent grow compared to the previous year?
    # Formula: (This Year Rent - Last Year Rent) / Last Year Rent × 100
    #
    # This reveals the post-COVID rental surge clearly
    df["Rent_Growth_YoY"] = (
        df.groupby("Region")["Avg_Monthly_Rent"]
        .pct_change() * 100
    ).round(2)

    print("    ✅ Year-on-Year Rent Growth %")

    # ── Feature 4: House Price to Income Ratio ───────────────
    # How many years of salary does a house cost?
    # Formula: House Price / Annual Wage
    #
    # BENCHMARK: Historically 3-4x in the UK
    # London is now above 12x — the crisis in one number
    df["Price_to_Income_Ratio"] = (
        df["Avg_House_Price"] / df["Median_Annual_Wage"]
    ).round(2)

    print("    ✅ House Price to Income Ratio")

    # ── Feature 5: Estimated Monthly Mortgage Cost ───────────
    # What would a monthly mortgage cost on this property?
    #
    # ASSUMPTIONS (standard UK mortgage):
    #   - 10% deposit (so loan = 90% of house price)
    #   - Mortgage rate = BOE Base Rate + 2% (typical lender spread)
    #   - 25 year repayment mortgage
    #
    # FORMULA (standard mortgage payment):
    #   M = P × [r(1+r)^n] / [(1+r)^n - 1]
    #   Where: P = loan amount, r = monthly rate, n = months
    loan = df["Avg_House_Price"] * 0.90          # 90% LTV
    annual_rate = (df["BOE_Base_Rate"] + 2) / 100  # BOE + 2% spread
    monthly_rate = annual_rate / 12
    n = 25 * 12                                   # 300 months

    # Handle edge case where rate is 0 (avoid division by zero)
    df["Monthly_Mortgage_Cost"] = np.where(
        monthly_rate > 0,
        loan * (monthly_rate * (1 + monthly_rate) ** n) /
        ((1 + monthly_rate) ** n - 1),
        loan / n
    ).round(2)

    print("    ✅ Estimated Monthly Mortgage Cost")

    # ── Feature 6: Rent vs Mortgage Gap ──────────────────────
    # Is renting cheaper or more expensive than buying?
    # Formula: Monthly Rent - Monthly Mortgage Cost
    #
    # Positive = renting is MORE expensive than mortgage
    # Negative = renting is CHEAPER than mortgage
    #
    # In 2023-24 with high BOE rates, mortgages became very
    # expensive — this gap narrowed or flipped in many regions
    df["Rent_vs_Mortgage_Gap"] = (
        df["Avg_Monthly_Rent"] - df["Monthly_Mortgage_Cost"]
    ).round(2)

    print("    ✅ Rent vs Mortgage Gap (rent - mortgage cost)")

    return df


# ================================================================
# MASTER FUNCTION
# ================================================================

def preprocess_all():
    """
    Run all preprocessing steps and save the master dataset.
    """

    print("=" * 55)
    print("  UK RENTAL ANALYSIS — Preprocessing")
    print("=" * 55)

    # Run all steps
    df_rent, df_hpi, df_boe, df_wages = load_all_data()
    df = merge_datasets(df_rent, df_hpi, df_boe, df_wages)
    df = clean_data(df)
    df = engineer_features(df)

    # Save master dataset
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/master.csv"
    df.to_csv(output_path, index=False)

    # Summary
    print("\n" + "-" * 55)
    print(f"  ✅ Master dataset saved to: {output_path}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\n  Columns in master dataset:")
    for col in df.columns:
        print(f"    - {col}")
    print("-" * 55)

    # Preview key metrics
    print("\n  Sample — London 2024:")
    london_2024 = df[(df["Region"] == "London") & (df["Year"] == 2024)]
    if not london_2024.empty:
        row = london_2024.iloc[0]
        print(f"    Avg Monthly Rent:      £{row['Avg_Monthly_Rent']:,.0f}")
        print(f"    Avg House Price:       £{row['Avg_House_Price']:,.0f}")
        print(f"    Median Annual Wage:    £{row['Median_Annual_Wage']:,.0f}")
        print(f"    Affordability Ratio:   {row['Affordability_Ratio']:.1f}%")
        print(f"    Price to Income Ratio: {row['Price_to_Income_Ratio']:.1f}x")
        print(f"    Rent vs Mortgage Gap:  £{row['Rent_vs_Mortgage_Gap']:,.0f}/month")

    return df


if __name__ == "__main__":
    preprocess_all()
