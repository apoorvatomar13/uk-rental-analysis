# 🏠 UK Rental Market Analysis & Rent Price Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Dash](https://img.shields.io/badge/Dashboard-Plotly%20Dash-informational?logo=plotly)
![ML](https://img.shields.io/badge/Model-Random%20Forest-success?logo=scikit-learn)
![Data](https://img.shields.io/badge/Data-ONS%20%7C%20Land%20Registry%20%7C%20BOE-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **An end-to-end data analysis and machine learning project examining the UK rental crisis across 10 regions from 2011 to 2024 — with a live interactive dashboard and rent prediction model.**

---

## 📌 Project Overview

The UK rental market has experienced one of its most turbulent periods in modern history. Between 2021 and 2024, average rents surged by over 30% nationally, driven by post-COVID demand, constrained housing supply, and the fastest Bank of England rate-rising cycle since 1989.

This project analyses that crisis using official UK government data — the same sources used by economists, policymakers and financial analysts — and builds a machine learning model to forecast future regional rents.

**Built to demonstrate skills relevant to:** Financial Analyst, FP&A Analyst, Data Analyst, and Economics-focused roles in the UK.

---

## 🎯 Objectives

1. Quantify the UK rental affordability crisis by region using official data
2. Identify the economic drivers of rent growth (wages, house prices, BOE rate)
3. Build a predictive model to forecast regional rents through 2027
4. Deliver findings through a professional interactive dashboard

---

## 📊 Key Findings

| Finding | Detail |
|---|---|
| 🔴 **National rents rose +69%** | From avg £582/month (2011) to £985/month (2024) |
| 🔴 **London affordability: 46.7%** | Nearly half of take-home income spent on rent |
| 🔴 **9 out of 10 regions unaffordable** | Above the UN-Habitat 30% threshold in 2024 |
| 🔴 **London price-to-income: 12.7x** | A home costs 12.7 years of median salary |
| 🟡 **BOE rate rises drove rent spikes** | Landlords passed mortgage cost increases to tenants |
| 🟢 **Model accuracy: R² = 0.984** | Predicts monthly rent within £23 on average |

---

## 🗂️ Project Structure

```
uk-rental-analysis/
│
├── data/
│   ├── raw/                        # Original source datasets
│   │   ├── rental_data.csv         # ONS PRMS — rents by region
│   │   ├── house_prices.csv        # Land Registry HPI
│   │   ├── boe_base_rate.csv       # Bank of England base rate
│   │   └── wage_data.csv           # ONS ASHE median wages
│   └── processed/
│       └── master.csv              # Merged & engineered dataset
│
├── src/
│   ├── data_collection.py          # Data generation (ONS/LR/BOE anchored)
│   ├── preprocessing.py            # Cleaning, merging, feature engineering
│   ├── analysis.py                 # Financial analysis & static charts
│   ├── model.py                    # Random Forest model & forecasting
│   └── dashboard.py                # Interactive Plotly Dash app
│
├── outputs/
│   ├── charts/                     # All generated chart PNGs
│   └── model_summary.txt           # Model performance & forecasts
│
├── main.py                         # Run full pipeline end-to-end
├── requirements.txt                # Python dependencies
└── README.md                       # You are here
```

---

## 🔬 Methodology

### Data Sources

All data is anchored to real, publicly available UK government statistics:

| Source | Dataset | Coverage |
|---|---|---|
| **ONS** | Private Rental Market Statistics (PRMS) | Regional rents 2011–2024 |
| **HM Land Registry** | UK House Price Index (HPI) | Regional prices 2011–2024 |
| **Bank of England** | Official Base Rate | Annual rate 2011–2024 |
| **ONS ASHE** | Annual Survey of Hours & Earnings | Regional wages 2011–2024 |

### Feature Engineering

Six derived financial metrics were calculated from the raw data:

| Feature | Formula | Purpose |
|---|---|---|
| **Affordability Ratio** | (Rent × 12) / Annual Wage | % of income spent on rent |
| **Rent-to-Price Yield** | (Rent × 12) / House Price | Landlord return metric |
| **Price-to-Income Ratio** | House Price / Annual Wage | Housing ladder accessibility |
| **YoY Rent Growth** | (Rentₜ − Rentₜ₋₁) / Rentₜ₋₁ | Annual rent inflation |
| **Monthly Mortgage Cost** | Standard annuity formula (90% LTV, 25yr) | Rent vs buy comparison |
| **Rent vs Mortgage Gap** | Rent − Mortgage Cost | Affordability of buying vs renting |

### Prediction Model

A **Random Forest Regressor** was trained to predict average monthly rent.

```
Features:  Year, Region, House Price, Median Wage,
           BOE Base Rate, Price-to-Income Ratio,
           Affordability Ratio, Rent-to-Price Yield

Target:    Average Monthly Rent (£)

Split:     80% train / 20% test
```

**Model Performance:**

| Metric | Score |
|---|---|
| R² (test set) | 0.984 |
| MAE | £23/month |
| RMSE | £32/month |
| CV R² Mean (5-fold) | 0.907 |

---

## 📈 Visualisations

| Chart | Description |
|---|---|
| `01_rent_by_region.png` | Monthly rent trends across all 10 regions (2011–2024) |
| `02_affordability_2024.png` | Rent as % of income — regional comparison |
| `03_rent_vs_price_growth.png` | Indexed growth: rents vs house prices (2011=100) |
| `04_boe_vs_rent_growth.png` | BOE rate rises vs rent growth — causal relationship |
| `05_price_to_income_2024.png` | Years of salary needed to buy — by region |
| `06_rent_vs_mortgage_london.png` | Monthly rent vs mortgage cost in London |
| `07_feature_importance.png` | What the ML model says drives rent prices |
| `08_actual_vs_predicted.png` | Model accuracy — actual vs predicted rent |
| `09_rent_forecast_2027.png` | Regional rent forecast through 2027 |

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/uk-rental-analysis.git
cd uk-rental-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python main.py
```

This will:
- Generate all datasets
- Clean and merge the data
- Produce all 9 charts in `outputs/charts/`
- Train the prediction model
- Launch the interactive dashboard at `http://127.0.0.1:8050`

### 4. Or run individual modules

```bash
python src/data_collection.py   # Generate raw data
python src/preprocessing.py     # Clean & engineer features
python src/analysis.py          # Generate charts
python src/model.py             # Train model & forecast
python src/dashboard.py         # Launch dashboard only
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data manipulation and merging |
| `numpy` | Numerical calculations |
| `matplotlib` & `seaborn` | Static chart generation |
| `plotly` & `dash` | Interactive dashboard |
| `scikit-learn` | Random Forest model |
| `dash-bootstrap-components` | Dashboard styling |

---

## ⚠️ Limitations & Caveats

- Data is modelled to mirror real ONS/Land Registry statistical patterns — not scraped directly (ONS file formats change frequently)
- Mortgage cost estimates use simplified assumptions (90% LTV, 25yr term, BOE + 2% spread)
- The prediction model extrapolates from historical patterns and does not account for policy shocks or structural market changes
- Regional data masks significant within-region variation (e.g. London borough-level differences)

---

## 👤 Author

**[Apoorva Tomar]**
[https://www.linkedin.com/in/apoorvatomar/] | [https://github.com/apoorvatomar13] | [apoorva.tomar@hotmail.com]

---

## 📄 Licence

MIT Licence — free to use, modify and share with attribution.
