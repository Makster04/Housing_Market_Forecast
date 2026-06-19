# Metro Housing-Market Regime Forecasting via Panel Time-Series Modeling

## Business & Data Understanding

### Business Problem
Real-estate decision-makers—investors, builders, and lenders—need clear, forward-looking
insight into *which* metro housing markets are heating or cooling, and *why*. Traditional
housing indicators (national price indices, headline inventory) are **national, lagged, and
opaque**: they say the market is softening but not whether it's a demand pullback or a supply
glut, and they wash out the enormous differences between metros. Our goal was to build a
forecasting pipeline that not only predicts each metro's next market phase but also explains
the supply-vs-demand forces driving it — and, now, how affordable that market is.

### Primary Stakeholders
- **Investors / iBuyers / REITs**: Time entry and exit by metro; price risk before the market moves
- **Homebuilders & Developers**: Decide where demand justifies new supply before breaking ground
- **Lenders & Mortgage Desks**: Identify rate-sensitive metros that cool fastest when rates rise
- **Policy & Housing Analysts**: Distinguish affordability stress caused by supply shortage vs demand surge
- **Data-Science / Strategy Team**: A clean, documented, leakage-safe panel ready to model on

### Objective
Build an interpretable, time-aware model to:
- Forecast each metro's **Supply Index** and **Demand Index** (and **Net Hotness**) one to six months ahead
- Classify the upcoming **market regime**: Hot · Balanced · Cooling · Buyer's Market
- Attribute every move to **supply vs demand** drivers, per metro — exactly, via the index decomposition and (planned) **SHAP**
- Contextualize each market with an **affordability burden** (payment-to-income)

---

## Data Understanding

### Data Sources
- **Realtor.com (RDC)** — metro listing metrics (the market itself)
- **Bureau of Labor Statistics (BLS LAUS)** — metro employment & labor force
- **Federal Reserve Economic Data (FRED)** — 30-yr fixed mortgage rate
- **Census Population Estimates (PEP)** — population & net migration
- **Census Building Permits (BPS)** — future-supply signal *(merged & folded into the Supply Index)*
- **Census ACS, Table B19013** — metro median household income, for the affordability burden *(merged)*

### Data Preparation Highlights
- Built a **balanced panel**: 387 metros × 115 months (Jul 2016 – Jan 2026), keyed on 5-digit **CBSA code**
- All sources joined on `cbsa_code` + `date` — **never on metro name** (codes are stable, names aren't)
- BLS crosswalk insight: `cbsa_code = area_code[4:9]`, enabling code-based labor joins
- Spliced two PEP vintages (V2019 + V2025) for full 2016–2026 population/migration; flagged the re-basing seam
- **Building permits** merged through Jan 2026 (full metro coverage from 2024-01, partial to 2019-11) and folded into the Supply Index wherever available
- **ACS income** (1-year B19013, 2016–2024) merged by metro-year and broadcast to months; 2020 interpolated (no Census release), 2025–26 left blank pending release — every value carries an `income_source` flag
- Engineered **COVID regime flags** (acute / boom / rate-shock) and an exponential **recency weight** (36-mo half-life)
- Quality controls: `quality_flag`, `pop_source` vintage flag, `laus_preliminary` revisability flag
- 70+ columns per metro-month across listings, labor, credit, population, permits, and affordability
- Tools: `pandas`, `numpy`, `openpyxl` (planned modeling: `statsmodels`, `lightgbm`, `shap`, `prophet`)

### Index & Affordability Construction *(built)*
- **Demand Index** = mean of 8 within-metro z-scores (pending ratio, pending YoY, −days-on-market, price-increased share, employment YoY, population YoY, net-migration rate, −mortgage rate)
- **Supply Index** = mean of 5 within-metro z-scores (active/new/total listing YoY, price-reduced share, permits per 1k)
- **Net Hotness** = Demand − Supply; headline `Demand_Index` / `Supply_Index` / `Net_Hotness` are rescaled **0–100**, with the raw z-score versions retained as the `_z` columns for modeling
- **Affordability burden** = monthly P&I (20% down, 30-yr, at that month's rate, on `ppsf × sqft`) ÷ monthly median household income — ~30% is the classic cost-burdened line

---

## Modeling Pipeline

### Forecasting Models (Step 1)
A model ladder matched to the signal, all validated against a baseline:
- **Seasonal-naive / SARIMA / ETS**: per-metro baseline — the bar every model must beat
- **VAR / pooled panel regression** (metro + month fixed effects): joint Supply+Demand forecasting given their co-dependence; the interpretation workhorse
- **XGBoost / LightGBM (global)**: one model across all metros for nonlinear accuracy
- **Hierarchical / dynamic-factor**: shares strength across metros (115 months is short + COVID break)

All targets modeled on the **z-score index scale** (`_z`) at t+1/t+3/t+6, with **sliding-window** forecasting and every feature lagged to its publication-known timing (no same-month leakage).

### Regime Classifier (Step 2)
- Classifies the forecast horizon into **Hot · Balanced · Cooling · Buyer's Market**
- Features: lagged indicators (1–6 months), index AR lags / momentum / volatility, seasonality encodings, engineered COVID/stress flags
- Class balance handled for rarer regimes; weighted by `HouseholdRank` × recency
- Validated using **rolling-origin (time-based) cross-validation** with a 6-month embargo — never random splits

### Interpretability (Step 3)
- **Model-free decomposition** *(built)*: because each index is a linear mean of z-scores, every monthly move decomposes **exactly** into per-driver contributions — already shipped as `demand_decomposition.csv` / `supply_decomposition.csv`
- **SHAP** *(planned)* for every regime prediction — local (metro-month) and global (over time); each driver pre-tagged **supply** or **demand** and **collapsed into the two buckets**
- Output: per-metro verdict — *"Austin is tightening on demand (migration, jobs); Pittsburgh on supply (new listings)"*
- **Caveat**: the mortgage rate is national (no cross-metro variation) — report its effect nationally, never in a metro's supply/demand bucket

---

## Evaluation *(Plan — model pre-build)*

### Target Metrics
| Metric | Target |
|---|---|
| Regime classification accuracy | Beat seasonal-naive baseline by a clear margin |
| F1 (Cooling / Buyer's Market) | Prioritize recall on downturns over raw accuracy |
| Forecast error (MAE/MAPE on the indices) | Lower than baseline at the 1–6 month horizon |
| Lead time | Detect regime shifts 1–2 months before they occur |

### Planned Evaluation Visuals
- **Confusion & transition matrices**: do we predict the *right* regime shifts, not just levels?
- **SHAP timeline**: how driver mix evolves into a metro's cooldown
- **Lead-time curve**: how early downturns are detected vs. actual switch
- **Supply-vs-demand decomposition** per metro: the headline deliverable

> Honesty note: the panel, the Supply/Demand indices, the leakage-safe feature/target
> engineering, and the model-free attribution are **built and validated**. The forecasting and
> classification **models are not yet trained** — the numbers above are **targets**, not results.
> First milestone is a baseline + pooled regression on the real data.

---

## Key Recommendations

| Role | Recommendation |
|---|---|
| Investors / REITs | Rotate toward metros flagged as tightening on **demand** (durable) vs **supply** (transient) |
| Homebuilders | Build where demand-driven heat persists; avoid metros where permits already signal a supply wave |
| Lenders | Stress-test the books against rate shocks in the most **rate-sensitive** metros |
| Policy Analysts | Target interventions by cause — supply incentives vs demand cooling — and flag metros where the **affordability burden** is highest |
| Strategy Team | Stand up the baseline model now — the panel, indices, and leakage-safe targets are already in place |

---

## 🔮 Next Steps

| Phase | Action |
|---|---|
| **Now** | Build baseline + pooled panel regression on the model panel; produce the first per-metro supply/demand decomposition |
| **Near-Term** | Add global LightGBM + the regime classifier; SHAP attribution collapsed to supply/demand |
| **Long-Term** | Hierarchical / dynamic-factor models; Tableau supply/demand + affordability dashboard; monthly refresh pipeline |

Additional future ideas:
- **Counterfactual scenarios**: "what if rates rise 100bps" — which metros cool, and how fast
- **Vintage-correct backtesting** (ALFRED) to eliminate look-ahead from revised LAUS/permit data
- **Real-time refresh pipeline** for monthly regime alerts as each source updates

---

## Project Assets
- `housing_panel_metro_pop_with_2025_2026_permits.xlsx` — the master panel (387 × 115, all legs: listings, labor, credit, population, permits-in-Supply-Index, income & affordability burden), with `Index_Methodology`, `Permits_Merge_Log`, `Remaining_Permit_Gaps`, and `Income_Burden_Notes`
- `housing_model_panel.xlsx` / `.csv` — leakage-safe modeling table: as-of features, t+1/t+3/t+6 targets, rolling-origin folds, weights, attribution, `README_modeling` + `feature_dictionary`
- `housing_index_inputs.xlsx` — transparent index construction, inputs color-coded supply vs demand
- `demand_decomposition.csv` / `supply_decomposition.csv` — exact per-driver attribution
- `housing_regime_forecasting_roadmap.md` — step-by-step build roadmap
- `housing_panel_project_summary.md` — this document
- *(planned)* `forecast_pipeline.py`, `regime_predictions.csv`, `shap_outputs/`, `evaluation_metrics.txt`
