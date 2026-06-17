# 🏙️ Project Roadmap: Metro Housing Supply & Demand Regime Forecasting via Panel Time-Series Modeling

## 🔍 Project Overview
**Objective**: Build a leakage-safe panel time-series pipeline that forecasts each U.S. metro's **Supply Index** and **Demand Index** one to six months ahead, classifies the upcoming **market regime** (Hot · Balanced · Cooling · Buyer's Market), and attributes every move to **supply vs demand** drivers — using a balanced panel of 387 metros × 115 months (Jul 2016 – Jan 2026) merged from Realtor.com, BLS, FRED, and Census.

**Core question**: Not just *is* a metro heating or cooling, but *why* — a demand pull (migration, jobs, rates) or a supply build (new listings, price cuts, permits).

---

## ✅ Step 1: Data Collection & Preparation
**Datasets Used**: Realtor.com (RDC) listings, BLS LAUS, FRED `MORTGAGE30US`, Census PEP, Census BPS permits

**Goals**:
- Build a **balanced panel** keyed on 5-digit `cbsa_code` + `date` — join on code, never metro name
- Crosswalk BLS LAUS to metros via `cbsa_code = area_code[4:9]`
- Splice two Census PEP vintages (V2019 + V2025) for continuous 2016–2026 population/migration; flag the re-basing seam with `pop_source`
- Merge Census building-permits leg (`permits_total_units`, `permits_1unit_units`, `permits_5plus_units`); full metro coverage from 2024-01, partial back to 2019-11
- Engineer COVID regime flags (`covid_acute`, `covid_boom`, `rate_shock`) and an exponential `recency_weight` (36-month half-life)
- Quality controls: `quality_flag`, `pop_source` vintage flag, `laus_preliminary` revisability flag

**Visuals**:
- Data-availability timeline per source
- Missingness heatmap before/after merge
- Permit-coverage-by-month bar (the 2019-11 → 2024-01 ramp)

---

## 🧮 Step 2: Index Construction
**Datasets Used**: merged panel from Step 1

**Goals**:
- Transform raw signals to **YoY growth** (stationary, seasonally controlled) and **standardize within each metro** (z-score over its own history)
- **Demand_Index_z** = mean of 8 z-scored signals: pending ratio, pending YoY, *−*days-on-market, price-increased share, employment YoY, population YoY, net-migration rate, *−*mortgage rate
- **Supply_Index_z** = mean of 5 z-scored signals: active/new/total listing YoY, price-reduced share, permits per 1k
- **Net_Hotness_z** = Demand_Index_z − Supply_Index_z (z-score scale, centered on 0, ≈ −3 to +3)
- Rescale all three to **0–100** as the headline `Demand_Index`, `Supply_Index`, `Net_Hotness` columns — `100 × (z − min) / (max − min)` across the panel — for dashboards and regime thresholds; retain the z-score versions as the `_z` columns for modeling
- Invert days-on-market and mortgage rate (higher values weaken demand)

**Visuals**:
- Per-metro Demand vs Supply index over time
- Supply/demand **quadrant scatter** (x = Supply, y = Demand), one dot per metro
- National choropleth colored by Net_Hotness

---

## 🏗️ Step 3: Feature Engineering
**Datasets Used**: index panel from Step 2

**Goals**:
- **Lag every feature to what was actually known** at the forecast origin: listings & labor lag 1 month, permits lag 2, mortgage real-time (lag 0), population as-is — the leakage-critical step
- Autoregressive features: index lags (1/3/6 mo), 3-month momentum, 6-month rolling volatility
- Regime memory: prior-month regime, consecutive-months-in-regime run length
- Seasonality encodings (`month_sin`, `month_cos`) and known stress flags (COVID / rate-shock)
- Log-levels of listing counts for stationarity-friendly scale information
- Weights: `recency_weight` × household-size weight → `sample_weight`

**Visuals**:
- Correlation heatmap (raw vs engineered)
- Rolling stats around regime switches
- Feature-availability map (warm-up period before YoY/lags exist)

---

## 🧠 Step 4: Regime Labeling
**Dataset Used**: index panel

**Goals**:
- Classify each metro-month from the `Net_Hotness` (0–100 headline) thresholds → **Hot** (≥75) · **Balanced** (≥50) · **Cooling** (≥25) · **Buyer's Market** (<25)
- Tag the **Supply/Demand Quadrant** (each axis split at 50): Frozen, Buyer's/cooling, Hot, Active-but-balanced
- These become the classification targets at each forecast horizon

**Visuals**:
- Regime distribution bar plot
- Demand vs Supply scatter colored by regime
- Regime transition (Markov) matrix

---

## 🔍 Step 5: Historical Pattern Analysis
**Datasets Used**: index + regime-labeled panel

**Goals**:
- Analyze indicator behavior near regime transitions, focused on the structural breaks: **2020 COVID freeze**, **2021 boom**, **2022–23 rate shock**
- Contrast Sun Belt (structurally hot on demand) vs Rust Belt metros
- Validate that the indices and regimes move sensibly through known episodes

**Visuals**:
- Time-series overlay (Demand, Supply, mortgage rate) with regime zones shaded
- Driver mix into a known metro cooldown

---

## 📈 Step 6: Forecasting the Indices (Step 8.1)
**Datasets Used**: leakage-safe model panel (features + t+1/t+3/t+6 targets)

**Goals**:
- Identify targets: `Demand_Index_z`, `Supply_Index_z` (and `Net_Hotness_z`) at 1/3/6-month horizons; model level and/or change. Modeling uses the z-score scale, not the 0–100 headline columns
- Match model to signal:
  - **Seasonal-naive / SARIMA / ETS**: per-metro baseline — the bar every model must beat
  - **VAR / pooled panel regression** (metro + month fixed effects): joint Supply+Demand forecasting given their co-dependence; the interpretation workhorse
  - **Prophet**: trend/seasonal indicators
  - **XGBoost / LightGBM (global)**: one model across all metros for nonlinear accuracy
  - **Hierarchical / dynamic-factor**: share strength across metros (115 months is short + a COVID break)
- Run **sliding-window** forecasts for the next 1–6 months; save as `future_forecasts_df`

**Visuals**:
- Actual vs forecasted index, per metro
- Residual / error bands at each horizon
- Forecast fan chart past the last actual month

---

## 🧩 Step 7: Regime Classification on Forecasted Indices (Step 8.2)
**Input**: `future_forecasts_df`

**Goals**:
- Feed forecasted indices into the regime classifier to predict the **upcoming regime** + class probabilities per metro-month
- Prioritize **recall on downturns** (Cooling / Buyer's Market) over raw accuracy
- Save to `regime_predictions.csv`

**Visuals**:
- Stacked bar of predicted regime probabilities
- Forecasted regime timeline (2026 →)

---

## 🤖 Step 8: Retrospective Regime Classifier
**Datasets Used**: leakage-safe model panel + historical regime labels

**Goals**:
- Train a classifier on historical lagged features to predict the regime h months ahead
- Handle class imbalance (rarer regimes); weight by `HouseholdRank`
- **XGBoost classifier with rolling-origin (time-based) cross-validation** — never random splits; 6-month embargo between train and validation

**Visuals**:
- Confusion & transition matrices
- Classification report table
- SHAP feature importance

---

## 🎯 Step 9: Attribution — Supply vs Demand
**Input**: index panel + trained classifier

**Goals**:
- **Model-free decomposition (exact)**: because each index is a linear mean of z-scores, the monthly move decomposes exactly into per-driver contributions — ranks what moved each metro with no model
- **SHAP**: per-prediction attribution, each driver pre-tagged supply or demand, then **collapsed into the two buckets**
- Output per-metro verdict — *"Austin is tightening on demand (migration, jobs); Pittsburgh on supply (new listings)"*
- **Caveat**: the mortgage rate is national (no cross-metro variation) — report its effect nationally or via a rate × metro-trait interaction; never attribute it per metro

**Visuals**:
- Diverging driver-contribution bars per metro-month
- SHAP timeline of how the driver mix evolves into a cooldown
- Supply-vs-demand decomposition map (the headline deliverable)

---

## 📅 Step 10: Forecasting Logic & Evaluation
**Input**: forecasted indices + regime classifier

**Goals**:
- Iterative pipeline per future month: (1) forecast indices → (2) feed classifier → (3) emit regime + probabilities → `master_forecast_table`
- Evaluate against the baseline: regime accuracy, F1 on downturns, MAE/MAPE on the indices, and **lead time** (how early shifts are detected)
- **Vintage-correct backtesting** (ALFRED) to remove look-ahead from revised LAUS/permit data

**Visuals**:
- Regime-probability timeline
- Lead-time curve
- Feature-drift chart

---

## 📌 Deliverables
**Built**
- `housing_panel_metro_pop_with_2025_2026_permits.xlsx` — master panel (387 × 115, all legs, permits folded into Supply Index)
- `housing_model_panel.xlsx` / `.csv` — leakage-safe modeling table: as-of features, t+1/t+3/t+6 targets, rolling-origin folds, weights, attribution
- `housing_index_inputs.xlsx` — transparent index-construction sheet, inputs color-coded by supply vs demand
- `demand_decomposition.csv` / `supply_decomposition.csv` — exact per-driver attribution

**Planned**
- `forecast_pipeline.py` — end-to-end pipeline
- `forecasted_indicators.csv`, `regime_predictions.csv`, `evaluation_metrics.txt`
- SHAP outputs + Tableau supply/demand decomposition dashboard

---

## 🚦 Status
The panel and index construction are **complete and validated**; the feature/target engineering and leakage-safe splits are **built**. The forecasting and classification models are **not yet trained** — figures in the modeling steps are targets, not results. First milestone: seasonal-naive baseline + pooled panel regression on the real data, then the first per-metro supply/demand decomposition.
