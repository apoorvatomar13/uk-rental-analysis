"""
================================================================
analysis.py — Financial Analysis & Visualisations
================================================================
PURPOSE:
    Generate 6 professional charts that tell the complete story
    of the UK rental crisis using our master dataset.

CHARTS PRODUCED:
    1. Average Monthly Rent by Region (2011-2024) — Line chart
    2. Affordability Ratio by Region (2024) — Bar chart
    3. Rent Growth vs House Price Growth — Dual line chart
    4. BOE Base Rate vs National Avg Rent Growth — Overlay chart
    5. Price to Income Ratio by Region (2024) — Horizontal bar
    6. Rent vs Mortgage Cost Over Time — Area chart (London)

WHY THESE CHARTS?
    Each chart answers a specific question a recruiter, analyst
    or policymaker would ask about the housing crisis. Together
    they build a complete narrative — which is exactly what
    FP&A analysts do: turn numbers into a story.
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── Global chart styling ──────────────────────────────────────
# A consistent professional style across all charts
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("tab10", 10)   # 10 distinct colours for 10 regions

# Output folder for saved charts
CHART_DIR = "outputs/charts"
os.makedirs(CHART_DIR, exist_ok=True)


# ================================================================
# CHART 1 — Average Monthly Rent by Region (2011-2024)
# ================================================================

def chart_rent_by_region(df):
    """
    Line chart showing how rents have changed in every UK region
    from 2011 to 2024.

    STORY THIS TELLS:
        - London has always been most expensive
        - Post-2021 every region surged sharply
        - The gap between London and the rest is widening
    """

    print("    Generating Chart 1: Rent by Region...")

    fig, ax = plt.subplots(figsize=(12, 7))

    regions = df["Region"].unique()

    for i, region in enumerate(sorted(regions)):
        region_data = df[df["Region"] == region].sort_values("Year")
        ax.plot(
            region_data["Year"],
            region_data["Avg_Monthly_Rent"],
            marker="o",
            markersize=4,
            linewidth=2,
            label=region,
            color=COLORS[i]
        )

    # Add vertical line at 2021 — the post-COVID rental surge
    ax.axvline(x=2021, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
    ax.text(2021.1, ax.get_ylim()[1] * 0.95, "Post-COVID\nSurge",
            color="red", fontsize=9, alpha=0.8)

    # Formatting
    ax.set_title("Average Monthly Rent by UK Region (2011–2024)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Average Monthly Rent (£)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"£{x:,.0f}"
    ))
    ax.set_xticks(df["Year"].unique())
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Add data source note
    fig.text(0.99, 0.01, "Source: ONS Private Rental Market Statistics",
             ha="right", fontsize=8, color="grey")

    plt.tight_layout()
    path = f"{CHART_DIR}/01_rent_by_region.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {path}")


# ================================================================
# CHART 2 — Affordability Ratio by Region (2024)
# ================================================================

def chart_affordability(df):
    """
    Bar chart showing what % of income goes on rent in each
    region in 2024.

    STORY THIS TELLS:
        - London is severely unaffordable (>40%)
        - Even cheaper regions are now at or above 30%
        - The 30% threshold (UN-Habitat standard) is breached
          almost everywhere
    """

    print("    Generating Chart 2: Affordability Ratio 2024...")

    df_2024 = df[df["Year"] == 2024].sort_values(
        "Affordability_Ratio", ascending=True
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    # Colour bars by severity
    bar_colors = [
        "#d62728" if x > 40 else   # Severely unaffordable — red
        "#ff7f0e" if x > 30 else   # Unaffordable — orange
        "#2ca02c"                   # Borderline — green
        for x in df_2024["Affordability_Ratio"]
    ]

    bars = ax.barh(
        df_2024["Region"],
        df_2024["Affordability_Ratio"],
        color=bar_colors,
        edgecolor="white",
        height=0.6
    )

    # Add value labels on bars
    for bar, val in zip(bars, df_2024["Affordability_Ratio"]):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", fontsize=10, fontweight="bold"
        )

    # Benchmark line at 30%
    ax.axvline(x=30, color="black", linestyle="--",
               linewidth=1.5, alpha=0.7, label="30% affordability threshold")

    # Legend for colours
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="> 40% — Severely unaffordable"),
        Patch(facecolor="#ff7f0e", label="> 30% — Unaffordable"),
        Patch(facecolor="#2ca02c", label="< 30% — Borderline affordable"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.set_title("Housing Affordability by Region — 2024\n"
                 "(Annual Rent as % of Median Gross Income)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Rent as % of Annual Income", fontsize=12)
    ax.set_xlim(0, df_2024["Affordability_Ratio"].max() + 8)

    fig.text(0.99, 0.01,
             "Source: ONS PRMS & ONS ASHE | Threshold: UN-Habitat Standard",
             ha="right", fontsize=8, color="grey")

    plt.tight_layout()
    path = f"{CHART_DIR}/02_affordability_2024.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {path}")


# ================================================================
# CHART 3 — Rent Growth vs House Price Growth (National Avg)
# ================================================================

def chart_rent_vs_price_growth(df):
    """
    Dual line chart comparing how rents and house prices have
    grown since 2011 (indexed to 100 = 2011 baseline).

    WHY INDEX TO 100?
        Rents and house prices are in different £ ranges.
        Indexing lets us compare GROWTH RATES fairly on one chart.
        This is a standard technique in economic analysis.

    STORY THIS TELLS:
        - House prices grew faster than rents pre-2021
        - Post-2021 rent growth caught up and exceeded price growth
        - This is the rental crisis in one chart
    """

    print("    Generating Chart 3: Rent vs House Price Growth...")

    # Calculate national average per year
    national = df.groupby("Year").agg(
        Avg_Rent=("Avg_Monthly_Rent", "mean"),
        Avg_Price=("Avg_House_Price", "mean")
    ).reset_index()

    # Index to 100 at 2011
    base_rent  = national.loc[national["Year"] == 2011, "Avg_Rent"].values[0]
    base_price = national.loc[national["Year"] == 2011, "Avg_Price"].values[0]

    national["Rent_Index"]  = (national["Avg_Rent"]  / base_rent  * 100).round(1)
    national["Price_Index"] = (national["Avg_Price"] / base_price * 100).round(1)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(national["Year"], national["Rent_Index"],
            marker="o", linewidth=2.5, color="#d62728",
            label="Average Rent (indexed)", markersize=5)

    ax.plot(national["Year"], national["Price_Index"],
            marker="s", linewidth=2.5, color="#1f77b4",
            label="Average House Price (indexed)", markersize=5)

    # Shade the post-COVID period
    ax.axvspan(2021, 2024, alpha=0.08, color="red",
               label="Post-COVID rental surge")

    ax.axhline(y=100, color="grey", linestyle=":", linewidth=1)
    ax.text(2011.1, 101, "2011 baseline = 100",
            fontsize=8, color="grey")

    # Annotate final values
    ax.annotate(f"{national['Rent_Index'].iloc[-1]:.0f}",
                xy=(2024, national["Rent_Index"].iloc[-1]),
                xytext=(10, 0), textcoords="offset points",
                fontsize=10, color="#d62728", fontweight="bold")

    ax.annotate(f"{national['Price_Index'].iloc[-1]:.0f}",
                xy=(2024, national["Price_Index"].iloc[-1]),
                xytext=(10, 0), textcoords="offset points",
                fontsize=10, color="#1f77b4", fontweight="bold")

    ax.set_title("UK Rent Growth vs House Price Growth (2011–2024)\n"
                 "Indexed: 2011 = 100",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Index (2011 = 100)", fontsize=12)
    ax.set_xticks(national["Year"])
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=10)

    fig.text(0.99, 0.01,
             "Source: ONS PRMS | HM Land Registry HPI",
             ha="right", fontsize=8, color="grey")

    plt.tight_layout()
    path = f"{CHART_DIR}/03_rent_vs_price_growth.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {path}")


# ================================================================
# CHART 4 — BOE Base Rate vs Rent Growth
# ================================================================

def chart_boe_vs_rent(df):
    """
    Overlay chart showing the BOE base rate alongside national
    average rent growth year-on-year.

    STORY THIS TELLS:
        - Low rates (2012-2021) kept rents relatively stable
        - When BOE raised rates sharply in 2022-23, landlords
          passed mortgage cost increases to tenants
        - Rent growth peaked alongside rate rises — causal link
    """

    print("    Generating Chart 4: BOE Rate vs Rent Growth...")

    national = df.groupby("Year").agg(
        Rent_Growth=("Rent_Growth_YoY", "mean"),
        BOE_Rate=("BOE_Base_Rate", "first")
    ).reset_index().dropna()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis — Rent Growth
    color_rent = "#d62728"
    ax1.bar(national["Year"], national["Rent_Growth"],
            color=color_rent, alpha=0.6, label="Avg Rent Growth YoY (%)",
            width=0.4)
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Average Rent Growth YoY (%)", color=color_rent, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_rent)
    ax1.set_xticks(national["Year"])
    ax1.tick_params(axis="x", rotation=45)

    # Right axis — BOE Rate
    ax2 = ax1.twinx()
    color_boe = "#1f77b4"
    ax2.plot(national["Year"], national["BOE_Rate"],
             color=color_boe, linewidth=2.5, marker="D",
             markersize=6, label="BOE Base Rate (%)")
    ax2.set_ylabel("BOE Base Rate (%)", color=color_boe, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_boe)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=10)

    ax1.set_title("Bank of England Base Rate vs UK Rent Growth (2012–2024)\n"
                  "Higher rates → higher landlord costs → higher rents",
                  fontsize=14, fontweight="bold", pad=15)

    fig.text(0.99, 0.01,
             "Source: ONS PRMS | Bank of England",
             ha="right", fontsize=8, color="grey")

    plt.tight_layout()
    path = f"{CHART_DIR}/04_boe_vs_rent_growth.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {path}")


# ================================================================
# CHART 5 — Price to Income Ratio by Region (2024)
# ================================================================

def chart_price_to_income(df):
    """
    Horizontal bar chart showing how many years of salary a home
    costs in each region in 2024.

    STORY THIS TELLS:
        - London at 12x is catastrophically unaffordable
        - Even the North East at ~5x is above the historic norm
        - The UK housing ladder is broken almost everywhere
    """

    print("    Generating Chart 5: Price to Income Ratio 2024...")

    df_2024 = df[df["Year"] == 2024].sort_values(
        "Price_to_Income_Ratio", ascending=True
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    bar_colors = [
        "#d62728" if x > 10 else
        "#ff7f0e" if x > 7  else
        "#1f77b4"
        for x in df_2024["Price_to_Income_Ratio"]
    ]

    bars = ax.barh(
        df_2024["Region"],
        df_2024["Price_to_Income_Ratio"],
        color=bar_colors,
        edgecolor="white",
        height=0.6
    )

    for bar, val in zip(bars, df_2024["Price_to_Income_Ratio"]):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}x",
            va="center", fontsize=10, fontweight="bold"
        )

    # Historic norm line
    ax.axvline(x=4, color="green", linestyle="--",
               linewidth=1.5, alpha=0.8, label="Historic UK norm (4x)")
    ax.axvline(x=7, color="orange", linestyle="--",
               linewidth=1.5, alpha=0.8, label="Severely unaffordable (7x)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="> 10x — Crisis level"),
        Patch(facecolor="#ff7f0e", label="7–10x — Severely unaffordable"),
        Patch(facecolor="#1f77b4", label="< 7x — Unaffordable"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.set_title("House Price to Income Ratio by Region — 2024\n"
                 "(House Price ÷ Median Annual Wage)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Years of Salary to Buy Average Home", fontsize=12)
    ax.set_xlim(0, df_2024["Price_to_Income_Ratio"].max() + 2)

    fig.text(0.99, 0.01,
             "Source: HM Land Registry HPI | ONS ASHE",
             ha="right", fontsize=8, color="grey")

    plt.tight_layout()
    path = f"{CHART_DIR}/05_price_to_income_2024.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {path}")


# ================================================================
# CHART 6 — Rent vs Mortgage Cost in London (2011-2024)
# ================================================================

def chart_rent_vs_mortgage(df):
    """
    Area chart comparing monthly rent vs estimated mortgage cost
    in London over time.

    STORY THIS TELLS:
        - Pre-2022: mortgages were cheaper than rent (low rates)
        - 2022-2023: BOE rate rises made mortgages much more
          expensive than renting — trapping renters
        - This is why London renters CAN'T save to buy —
          renting costs too much and mortgages cost even more
    """

    print("    Generating Chart 6: Rent vs Mortgage — London...")

    london = df[df["Region"] == "London"].sort_values("Year")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(london["Year"], london["Avg_Monthly_Rent"],
                    alpha=0.4, color="#d62728", label="Monthly Rent")
    ax.fill_between(london["Year"], london["Monthly_Mortgage_Cost"],
                    alpha=0.4, color="#1f77b4", label="Est. Monthly Mortgage")

    ax.plot(london["Year"], london["Avg_Monthly_Rent"],
            color="#d62728", linewidth=2.5, marker="o", markersize=5)
    ax.plot(london["Year"], london["Monthly_Mortgage_Cost"],
            color="#1f77b4", linewidth=2.5, marker="s", markersize=5)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"£{x:,.0f}"
    ))
    ax.set_xticks(london["Year"])
    ax.tick_params(axis="x", rotation=45)

    ax.set_title("Monthly Rent vs Estimated Mortgage Cost — London (2011–2024)\n"
                 "Mortgage assumes 90% LTV, 25yr term, BOE rate + 2% spread",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Monthly Cost (£)", fontsize=12)
    ax.legend(fontsize=11)

    fig.text(0.99, 0.01,
             "Source: ONS PRMS | Land Registry | Bank of England",
             ha="right", fontsize=8, color="grey")

    plt.tight_layout()
    path = f"{CHART_DIR}/06_rent_vs_mortgage_london.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {path}")


# ================================================================
# MASTER FUNCTION
# ================================================================

def run_analysis(df=None):
    """
    Run all 6 charts. Loads master.csv if df not passed in.
    """

    print("=" * 55)
    print("  UK RENTAL ANALYSIS — Analysis & Visualisations")
    print("=" * 55)

    if df is None:
        df = pd.read_csv("data/processed/master.csv")
        print(f"\n  Loaded master dataset: {df.shape}")

    print("\n  Generating 6 charts...\n")

    chart_rent_by_region(df)
    chart_affordability(df)
    chart_rent_vs_price_growth(df)
    chart_boe_vs_rent(df)
    chart_price_to_income(df)
    chart_rent_vs_mortgage(df)

    print("\n" + "-" * 55)
    print("  All charts saved to outputs/charts/")
    print("\n  Key Findings:")
    print("  ─────────────────────────────────────────────")

    # Auto-generate key findings from the data
    df_2024 = df[df["Year"] == 2024]
    df_2011 = df[df["Year"] == 2011]

    london_afford = df_2024[df_2024["Region"]=="London"]["Affordability_Ratio"].values[0]
    nat_afford    = df_2024["Affordability_Ratio"].mean()
    above_30      = (df_2024["Affordability_Ratio"] > 30).sum()
    london_pti    = df_2024[df_2024["Region"]=="London"]["Price_to_Income_Ratio"].values[0]

    avg_rent_2011 = df_2011["Avg_Monthly_Rent"].mean()
    avg_rent_2024 = df_2024["Avg_Monthly_Rent"].mean()
    rent_growth   = ((avg_rent_2024 - avg_rent_2011) / avg_rent_2011 * 100)

    print(f"  1. London affordability ratio: {london_afford:.1f}% of income on rent")
    print(f"  2. National avg affordability: {nat_afford:.1f}%")
    print(f"  3. Regions above 30% threshold: {above_30} out of 10")
    print(f"  4. London price-to-income ratio: {london_pti:.1f}x salary")
    print(f"  5. National avg rent growth (2011-2024): +{rent_growth:.0f}%")
    print("  ─────────────────────────────────────────────")

    return df


if __name__ == "__main__":
    run_analysis()
