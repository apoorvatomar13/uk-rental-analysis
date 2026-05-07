"""
================================================================
dashboard.py — Interactive Plotly Dash Dashboard
================================================================
PURPOSE:
    A fully interactive web dashboard that lets users explore
    the UK rental crisis data and run live rent predictions.

HOW TO RUN:
    python src/dashboard.py
    Then open: http://127.0.0.1:8050 in your browser

DASHBOARD SECTIONS:
    1. KPI Cards     — headline numbers at a glance
    2. Rent Trends   — interactive line chart by region
    3. Affordability — bar chart with region filter
    4. Rent Predictor — live ML model prediction panel

TECH STACK:
    Dash       — Python web framework (built on Flask)
    Plotly     — interactive charts
    Bootstrap  — clean professional styling
================================================================
"""

import pandas as pd
import numpy as np
import os
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# ── Load data ─────────────────────────────────────────────────
def load_data():
    df = pd.read_csv("data/processed/master.csv")
    return df

# ── Train model silently for the predictor panel ──────────────
def train_model_for_dashboard(df):
    le = LabelEncoder()
    df = df.copy()
    df["Region_Encoded"] = le.fit_transform(df["Region"])

    feature_cols = [
        "Year", "Region_Encoded", "Avg_House_Price",
        "Median_Annual_Wage", "BOE_Base_Rate",
        "Price_to_Income_Ratio", "Affordability_Ratio",
        "Rent_to_Price_Yield",
    ]

    mask = df[feature_cols].notnull().all(axis=1)
    X = df.loc[mask, feature_cols]
    y = df.loc[mask, "Avg_Monthly_Rent"]

    model = RandomForestRegressor(
        n_estimators=200, max_depth=8,
        min_samples_leaf=3, random_state=42
    )
    model.fit(X, y)
    return model, le, feature_cols


# ================================================================
# COLOUR PALETTE
# ================================================================
COLORS = {
    "primary":    "#1f77b4",
    "danger":     "#d62728",
    "warning":    "#ff7f0e",
    "success":    "#2ca02c",
    "dark":       "#2c3e50",
    "light_bg":   "#f8f9fa",
    "card_bg":    "#ffffff",
}

REGIONS = [
    "London", "South East", "South West", "East of England",
    "East Midlands", "West Midlands",
    "Yorkshire and The Humber", "North West",
    "North East", "Wales"
]


# ================================================================
# BUILD THE DASH APP
# ================================================================

def build_app(df, model, le, feature_cols):
    """
    Construct the full Dash application layout and callbacks.

    LAYOUT STRUCTURE:
        Header
        ├── Row 1: 4 KPI Cards
        ├── Row 2: Rent Trend Chart | Affordability Chart
        ├── Row 3: Rent vs House Price | BOE Rate Chart
        └── Row 4: Live Rent Predictor Panel
    """

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        title="UK Rental Market Dashboard"
    )

    # ── Pre-compute summary stats ─────────────────────────────
    df_2024 = df[df["Year"] == 2024]
    df_2011 = df[df["Year"] == 2011]

    nat_avg_rent_2024 = df_2024["Avg_Monthly_Rent"].mean()
    nat_avg_rent_2011 = df_2011["Avg_Monthly_Rent"].mean()
    rent_growth_pct   = ((nat_avg_rent_2024 - nat_avg_rent_2011)
                         / nat_avg_rent_2011 * 100)
    avg_afford_2024   = df_2024["Affordability_Ratio"].mean()
    london_pti        = df_2024[
        df_2024["Region"] == "London"
    ]["Price_to_Income_Ratio"].values[0]
    regions_over_30   = (df_2024["Affordability_Ratio"] > 30).sum()


    # ── Helper: KPI Card ──────────────────────────────────────
    def kpi_card(title, value, subtitle, color):
        return dbc.Card([
            dbc.CardBody([
                html.P(title, className="text-muted mb-1",
                       style={"fontSize": "0.85rem", "fontWeight": "600"}),
                html.H3(value, style={"color": color, "fontWeight": "800"}),
                html.P(subtitle, className="text-muted mb-0",
                       style={"fontSize": "0.80rem"}),
            ])
        ], style={"borderLeft": f"4px solid {color}",
                  "borderRadius": "8px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})


    # ================================================================
    # APP LAYOUT
    # ================================================================
    app.layout = dbc.Container([

        # ── Header ───────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2("🏠 UK Rental Market Dashboard",
                            style={"color": COLORS["dark"],
                                   "fontWeight": "800", "marginBottom": "4px"}),
                    html.P(
                        "Analysing rent affordability, house prices & the rental crisis across UK regions (2011–2024)",
                        className="text-muted mb-0",
                        style={"fontSize": "0.95rem"}
                    ),
                    html.Hr()
                ])
            ])
        ], className="mt-3"),

        # ── KPI Cards ─────────────────────────────────────────
        dbc.Row([
            dbc.Col(kpi_card(
                "National Avg Rent (2024)",
                f"£{nat_avg_rent_2024:,.0f}/mo",
                f"+{rent_growth_pct:.0f}% since 2011",
                COLORS["danger"]
            ), width=3),
            dbc.Col(kpi_card(
                "Avg Affordability Ratio",
                f"{avg_afford_2024:.1f}%",
                "of income spent on rent (threshold: 30%)",
                COLORS["warning"]
            ), width=3),
            dbc.Col(kpi_card(
                "London Price-to-Income",
                f"{london_pti:.1f}x",
                "years of salary to buy avg London home",
                COLORS["danger"]
            ), width=3),
            dbc.Col(kpi_card(
                "Regions Above 30% Threshold",
                f"{regions_over_30} / 10",
                "regions with unaffordable rents in 2024",
                COLORS["warning"]
            ), width=3),
        ], className="mb-4"),

        # ── Row 2: Rent Trends + Affordability ───────────────
        dbc.Row([

            # Left: Rent Trend Chart with region filter
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col(html.H6("Average Monthly Rent by Region",
                                            className="mb-0 fw-bold"), width=7),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="region-filter",
                                    options=[{"label": r, "value": r}
                                             for r in REGIONS],
                                    value=REGIONS,
                                    multi=True,
                                    clearable=False,
                                    style={"fontSize": "0.82rem"}
                                ), width=5
                            )
                        ], align="center")
                    ]),
                    dbc.CardBody(
                        dcc.Graph(id="rent-trend-chart", style={"height": "350px"})
                    )
                ], style={"borderRadius": "8px",
                          "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})
            ], width=7),

            # Right: Affordability bar chart 2024
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H6("Affordability Ratio by Region — 2024",
                                className="mb-0 fw-bold")
                    ),
                    dbc.CardBody(
                        dcc.Graph(id="affordability-chart",
                                  style={"height": "350px"})
                    )
                ], style={"borderRadius": "8px",
                          "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})
            ], width=5),

        ], className="mb-4"),

        # ── Row 3: Indexed Growth + BOE Rate ─────────────────
        dbc.Row([

            # Left: Rent vs Price Growth (indexed)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H6("Rent Growth vs House Price Growth (2011=100)",
                                className="mb-0 fw-bold")
                    ),
                    dbc.CardBody(
                        dcc.Graph(id="growth-chart", style={"height": "320px"})
                    )
                ], style={"borderRadius": "8px",
                          "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})
            ], width=6),

            # Right: BOE Rate vs Rent Growth
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H6("BOE Base Rate vs Avg Rent Growth YoY",
                                className="mb-0 fw-bold")
                    ),
                    dbc.CardBody(
                        dcc.Graph(id="boe-chart", style={"height": "320px"})
                    )
                ], style={"borderRadius": "8px",
                          "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})
            ], width=6),

        ], className="mb-4"),

        # ── Row 4: Live Rent Predictor ────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H6("🔮 Live Rent Predictor — ML Model",
                                className="mb-0 fw-bold")
                    ),
                    dbc.CardBody([
                        dbc.Row([

                            # Controls
                            dbc.Col([
                                html.Label("Region", className="fw-semibold"),
                                dcc.Dropdown(
                                    id="pred-region",
                                    options=[{"label": r, "value": r}
                                             for r in REGIONS],
                                    value="London",
                                    clearable=False
                                ),
                                html.Br(),
                                html.Label("Forecast Year",
                                           className="fw-semibold"),
                                dcc.Slider(
                                    id="pred-year",
                                    min=2025, max=2030, step=1,
                                    value=2025,
                                    marks={y: str(y)
                                           for y in range(2025, 2031)},
                                ),
                                html.Br(),
                                html.Label("Avg House Price (£)",
                                           className="fw-semibold"),
                                dcc.Slider(
                                    id="pred-price",
                                    min=100000, max=900000,
                                    step=10000, value=524000,
                                    marks={
                                        100000: "£100k",
                                        300000: "£300k",
                                        500000: "£500k",
                                        700000: "£700k",
                                        900000: "£900k"
                                    },
                                ),
                                html.Br(),
                                html.Label("Median Annual Wage (£)",
                                           className="fw-semibold"),
                                dcc.Slider(
                                    id="pred-wage",
                                    min=20000, max=70000,
                                    step=1000, value=44000,
                                    marks={
                                        20000: "£20k",
                                        35000: "£35k",
                                        50000: "£50k",
                                        70000: "£70k"
                                    },
                                ),
                                html.Br(),
                                html.Label("BOE Base Rate (%)",
                                           className="fw-semibold"),
                                dcc.Slider(
                                    id="pred-boe",
                                    min=0.1, max=6.0,
                                    step=0.25, value=4.25,
                                    marks={
                                        0.1: "0.1%", 2.0: "2%",
                                        4.0: "4%",   6.0: "6%"
                                    },
                                ),
                            ], width=7),

                            # Prediction output
                            dbc.Col([
                                html.Div([
                                    html.P("Predicted Monthly Rent",
                                           className="text-muted mb-2",
                                           style={"fontSize": "1rem",
                                                  "fontWeight": "600"}),
                                    html.H1(id="prediction-output",
                                            style={"color": COLORS["danger"],
                                                   "fontWeight": "900",
                                                   "fontSize": "3.5rem"}),
                                    html.P(id="prediction-context",
                                           className="text-muted",
                                           style={"fontSize": "0.9rem"}),
                                    html.Hr(),
                                    html.Div(id="prediction-details",
                                             style={"fontSize": "0.88rem"})
                                ], style={"textAlign": "center",
                                          "paddingTop": "20px"})
                            ], width=5),
                        ])
                    ])
                ], style={"borderRadius": "8px",
                          "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"})
            ])
        ], className="mb-4"),

        # ── Footer ────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P(
                    "Data Sources: ONS Private Rental Market Statistics | "
                    "HM Land Registry House Price Index | "
                    "Bank of England | ONS ASHE Earnings Survey",
                    className="text-muted text-center",
                    style={"fontSize": "0.78rem"}
                )
            ])
        ])

    ], fluid=True, style={"backgroundColor": COLORS["light_bg"],
                          "minHeight": "100vh"})


    # ================================================================
    # CALLBACKS — These make the dashboard interactive
    # A callback runs automatically when the user changes an input
    # ================================================================

    # ── Callback 1: Rent Trend Chart ──────────────────────────
    @app.callback(
        Output("rent-trend-chart", "figure"),
        Input("region-filter", "value")
    )
    def update_rent_chart(selected_regions):
        """Redraw rent trend chart when regions are selected/deselected."""
        if not selected_regions:
            selected_regions = REGIONS

        fig = go.Figure()
        colors = px.colors.qualitative.Tab10

        for i, region in enumerate(sorted(selected_regions)):
            data = df[df["Region"] == region].sort_values("Year")
            fig.add_trace(go.Scatter(
                x=data["Year"],
                y=data["Avg_Monthly_Rent"],
                mode="lines+markers",
                name=region,
                line=dict(width=2, color=colors[i % 10]),
                marker=dict(size=5),
                hovertemplate=(
                    f"<b>{region}</b><br>"
                    "Year: %{x}<br>"
                    "Rent: £%{y:,.0f}/month<extra></extra>"
                )
            ))

        fig.add_vline(x=2021, line_dash="dash",
                      line_color="red", opacity=0.5,
                      annotation_text="Post-COVID surge",
                      annotation_position="top right")

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
            yaxis=dict(tickprefix="£", tickformat=","),
            xaxis=dict(dtick=1),
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig


    # ── Callback 2: Affordability Chart ───────────────────────
    @app.callback(
        Output("affordability-chart", "figure"),
        Input("region-filter", "value")
    )
    def update_affordability(selected_regions):
        if not selected_regions:
            selected_regions = REGIONS

        data = df[
            (df["Year"] == 2024) &
            (df["Region"].isin(selected_regions))
        ].sort_values("Affordability_Ratio")

        bar_colors = [
            COLORS["danger"]  if x > 40 else
            COLORS["warning"] if x > 30 else
            COLORS["success"]
            for x in data["Affordability_Ratio"]
        ]

        fig = go.Figure(go.Bar(
            x=data["Affordability_Ratio"],
            y=data["Region"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in data["Affordability_Ratio"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Affordability: %{x:.1f}%<extra></extra>"
            )
        ))

        fig.add_vline(x=30, line_dash="dash",
                      line_color="black", opacity=0.6,
                      annotation_text="30% threshold")

        fig.update_layout(
            margin=dict(l=10, r=50, t=10, b=10),
            xaxis=dict(title="Rent as % of Income", ticksuffix="%"),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig


    # ── Callback 3: Growth Index Chart ────────────────────────
    @app.callback(
        Output("growth-chart", "figure"),
        Input("region-filter", "value")
    )
    def update_growth_chart(_):
        national = df.groupby("Year").agg(
            Rent=("Avg_Monthly_Rent", "mean"),
            Price=("Avg_House_Price", "mean")
        ).reset_index()

        base_r = national.loc[national["Year"]==2011, "Rent"].values[0]
        base_p = national.loc[national["Year"]==2011, "Price"].values[0]
        national["Rent_Idx"]  = national["Rent"]  / base_r * 100
        national["Price_Idx"] = national["Price"] / base_p * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=national["Year"], y=national["Rent_Idx"],
            mode="lines+markers", name="Rent",
            line=dict(color=COLORS["danger"], width=2.5),
            hovertemplate="Year: %{x}<br>Rent Index: %{y:.1f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=national["Year"], y=national["Price_Idx"],
            mode="lines+markers", name="House Price",
            line=dict(color=COLORS["primary"], width=2.5),
            hovertemplate="Year: %{x}<br>Price Index: %{y:.1f}<extra></extra>"
        ))

        fig.add_hrect(y0=95, y1=105, line_width=0,
                      fillcolor="grey", opacity=0.08)
        fig.add_vrect(x0=2020.8, x1=2024.2, line_width=0,
                      fillcolor="red", opacity=0.05)

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.25),
            yaxis=dict(title="Index (2011 = 100)"),
            xaxis=dict(dtick=1),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig


    # ── Callback 4: BOE Chart ─────────────────────────────────
    @app.callback(
        Output("boe-chart", "figure"),
        Input("region-filter", "value")
    )
    def update_boe_chart(_):
        national = df.groupby("Year").agg(
            Rent_Growth=("Rent_Growth_YoY", "mean"),
            BOE=("BOE_Base_Rate", "first")
        ).reset_index().dropna()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=national["Year"], y=national["Rent_Growth"],
            name="Rent Growth YoY (%)",
            marker_color=COLORS["danger"],
            opacity=0.65,
            hovertemplate="Year: %{x}<br>Rent Growth: %{y:.1f}%<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=national["Year"], y=national["BOE"],
            mode="lines+markers", name="BOE Rate (%)",
            line=dict(color=COLORS["primary"], width=2.5),
            yaxis="y2",
            hovertemplate="Year: %{x}<br>BOE Rate: %{y:.2f}%<extra></extra>"
        ))

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.25),
            yaxis=dict(title="Rent Growth (%)", ticksuffix="%"),
            yaxis2=dict(title="BOE Rate (%)", overlaying="y",
                        side="right", ticksuffix="%"),
            xaxis=dict(dtick=1),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig


    # ── Callback 5: Live Rent Predictor ───────────────────────
    @app.callback(
        Output("prediction-output",  "children"),
        Output("prediction-context", "children"),
        Output("prediction-details", "children"),
        Input("pred-region", "value"),
        Input("pred-year",   "value"),
        Input("pred-price",  "value"),
        Input("pred-wage",   "value"),
        Input("pred-boe",    "value"),
    )
    def predict_rent(region, year, house_price, wage, boe_rate):
        """
        Run the ML model in real time when sliders change.
        This fires every time the user adjusts any slider.
        """

        # Encode the region
        try:
            region_encoded = le.transform([region])[0]
        except Exception:
            region_encoded = 0

        # Derive the ratio features from the inputs
        price_to_income  = house_price / wage
        # Use the last known rent as proxy for affordability calc
        last_rent = df[df["Region"] == region]["Avg_Monthly_Rent"].iloc[-1]
        affordability    = (last_rent * 12) / wage * 100
        rent_yield       = (last_rent * 12) / house_price * 100

        X_input = pd.DataFrame([{
            "Year":                  year,
            "Region_Encoded":        region_encoded,
            "Avg_House_Price":       house_price,
            "Median_Annual_Wage":    wage,
            "BOE_Base_Rate":         boe_rate,
            "Price_to_Income_Ratio": price_to_income,
            "Affordability_Ratio":   affordability,
            "Rent_to_Price_Yield":   rent_yield,
        }])

        predicted = model.predict(X_input)[0]
        annual    = predicted * 12

        # Context sentence
        afford_pct = (predicted * 12) / wage * 100
        context    = f"≈ £{annual:,.0f}/year | {afford_pct:.1f}% of income"

        # Detail breakdown
        details = html.Div([
            dbc.Row([
                dbc.Col(html.Small("House Price", className="text-muted")),
                dbc.Col(html.Small(f"£{house_price:,}", className="fw-bold"),
                        className="text-end")
            ]),
            dbc.Row([
                dbc.Col(html.Small("Median Wage", className="text-muted")),
                dbc.Col(html.Small(f"£{wage:,}/yr", className="fw-bold"),
                        className="text-end")
            ]),
            dbc.Row([
                dbc.Col(html.Small("BOE Rate", className="text-muted")),
                dbc.Col(html.Small(f"{boe_rate}%", className="fw-bold"),
                        className="text-end")
            ]),
            dbc.Row([
                dbc.Col(html.Small("Price/Income Ratio",
                                   className="text-muted")),
                dbc.Col(html.Small(f"{price_to_income:.1f}x",
                                   className="fw-bold"),
                        className="text-end")
            ]),
            html.Hr(className="my-2"),
            html.Small(
                "⚠️ Forecast only. Based on historical patterns 2011–2024.",
                className="text-muted"
            )
        ])

        return f"£{predicted:,.0f}", context, details


    return app


# ================================================================
# LAUNCH
# ================================================================

def launch_dashboard(df=None):
    if df is None:
        df = load_data()

    print("  Training model for dashboard predictor...")
    model, le, feature_cols = train_model_for_dashboard(df)

    print("  Building dashboard layout...")
    app = build_app(df, model, le, feature_cols)

    print("\n" + "=" * 55)
    print("  Dashboard running at: http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop")
    print("=" * 55 + "\n")

    app.run(debug=False, port=8050)


if __name__ == "__main__":
    launch_dashboard()
