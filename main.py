"""app.py - Delhi Air Quality (AQI) Next-Day Forecaster Dashboard

Powered by NumPy, Pandas, and Scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Delhi AQI Forecaster | Round 2",
    page_icon="🌫️",
    layout="wide",
)


# ---------------------------------------------------------
# 1. DATA CACHING & PIPELINE (Pandas & Scikit-learn)
# ---------------------------------------------------------
@st.cache_data
def generate_and_prepare_data():
  """Generates 3-year historical time series and builds lag features using Pandas."""
  np.random.seed(42)
  dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="D")
  n = len(dates)

  # Seasonal winter spike formulation using NumPy
  # Fixed: dates.dayofyear -> dates.day_of_year
  day_of_year = dates.day_of_year
  winter_factor = np.exp(-0.5 * ((day_of_year - 320) / 35) ** 2) * 220 + np.exp(
      -0.5 * ((day_of_year - 15) / 30) ** 2
  ) * 180

  base_pm25 = np.clip(60 + winter_factor + np.random.normal(0, 25, n), 15, 550)
  base_no2 = np.clip(base_pm25 * 0.35 + np.random.normal(0, 10, n), 5, 150)
  aqi = np.clip(
      base_pm25 * 1.15 + base_no2 * 0.2 + np.random.normal(0, 10, n), 25, 500
  )

  df = pd.DataFrame(
      {"Date": dates, "PM25": base_pm25, "NO2": base_no2, "AQI": aqi}
  )

  # Feature Engineering (Strictly past lag features to prevent leakage)
  df["Target_Next_Day_AQI"] = df["AQI"].shift(-1)
  df["AQI_lag1"] = df["AQI"].shift(1)
  df["AQI_lag2"] = df["AQI"].shift(2)
  df["PM25_lag1"] = df["PM25"].shift(1)
  df["AQI_roll_mean_3d"] = df["AQI"].shift(1).rolling(window=3).mean()
  df["AQI_roll_mean_7d"] = df["AQI"].shift(1).rolling(window=7).mean()
  df["Month"] = df["Date"].dt.month
  df["Is_Winter_Spike"] = df["Month"].isin([10, 11, 12, 1]).astype(int)

  df = df.dropna().reset_index(drop=True)
  return df


@st.cache_resource
def train_models(df):
  """Trains Scikit-learn Ridge and Random Forest models on time-ordered split."""
  features = [
      "AQI_lag1",
      "AQI_lag2",
      "PM25_lag1",
      "AQI_roll_mean_3d",
      "AQI_roll_mean_7d",
      "Month",
      "Is_Winter_Spike",
  ]
  X = df[features]
  y = df["Target_Next_Day_AQI"]

  # 80/20 Chronological split (no shuffle to prevent data leakage)
  split_idx = int(len(df) * 0.8)
  X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
  y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

  # Pipeline 1: Ridge Regression
  ridge_pipe = Pipeline(
      [("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))]
  )
  ridge_pipe.fit(X_train, y_train)

  # Pipeline 2: Random Forest Regressor
  rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
  rf.fit(X_train, y_train)

  # Model Evaluation Metrics via Scikit-learn
  pred_rf = rf.predict(X_test)
  mae_rf = mean_absolute_error(y_test, pred_rf)
  r2_rf = r2_score(y_test, pred_rf)

  # Standard deviation of test residuals for individual prediction interval
  rf_residual_std = float(np.std(y_test.values - pred_rf))

  return (
      ridge_pipe,
      rf,
      features,
      mae_rf,
      r2_rf,
      X_test,
      y_test,
      rf_residual_std,
  )


# Load data and trained models
df = generate_and_prepare_data()
(
    ridge_pipe,
    rf_model,
    feature_names,
    mae_rf,
    r2_rf,
    X_test,
    y_test,
    rf_residual_std,
) = train_models(df)


# ---------------------------------------------------------
# 2. NUMPY HELPER FUNCTIONS (CPCB & GRAP)
# ---------------------------------------------------------
def get_cpcb_category(aqi_val):
  """Classify AQI using standard CPCB thresholds."""
  if aqi_val <= 50:
    return "Good", "🟢", "#10b981", "Minimal impact on health."
  elif aqi_val <= 100:
    return (
        "Satisfactory",
        "🟡",
        "#84cc16",
        "Minor breathing discomfort to sensitive people.",
    )
  elif aqi_val <= 200:
    return (
        "Moderate",
        "🟠",
        "#f59e0b",
        "Breathing discomfort to people with lungs/asthma.",
    )
  elif aqi_val <= 300:
    return (
        "Poor",
        "🔴",
        "#f97316",
        "Breathing discomfort on prolonged exposure.",
    )
  elif aqi_val <= 400:
    return (
        "Very Poor",
        "🟣",
        "#ef4444",
        "Respiratory illness on prolonged exposure.",
    )
  else:
    return (
        "Severe",
        "🚨",
        "#7f1d1d",
        "Affects healthy people and seriously impacts sensitive groups.",
    )


def get_grap_stage(aqi_val):
  if aqi_val > 450:
    return "GRAP Stage IV (Severe+)", "Diesel truck bans, school closures"
  elif aqi_val > 400:
    return (
        "GRAP Stage III (Severe)",
        "Halt construction & BS-III/IV private vehicles",
    )
  elif aqi_val > 300:
    return (
        "GRAP Stage II (Very Poor)",
        "Diesel generator bans, mechanized road sweeping",
    )
  elif aqi_val > 200:
    return "GRAP Stage I (Poor)", "Strict dust control, firecracker bans"
  return "No Emergency Protocol Active", "Normal monitoring"


# ---------------------------------------------------------
# 3. STREAMLIT DASHBOARD INTERFACE
# ---------------------------------------------------------
st.title("🌫️ Delhi Air Quality Next-Day Forecaster")
st.caption(
    "1st-Year Evaluation Project | Built with NumPy, Pandas & Scikit-learn"
)

# Top KPI metrics
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
  st.metric("Test MAE", f"±{mae_rf:.1f} AQI pts", "Random Forest")
with kpi2:
  st.metric("Test R² Score", f"{r2_rf:.2f}", "Variance Explained")
with kpi3:
  st.metric("Training Samples", f"{len(df) - len(X_test):,}", "Chronological")
with kpi4:
  st.metric("Features Engineered", f"{len(feature_names)}", "Pandas Lags")

st.divider()

# Sidebar: Input Panel / Monitoring Station Presets
st.sidebar.header("📍 Station & Historical Inputs")

station_preset = st.sidebar.selectbox(
    "Choose Delhi Station Preset",
    [
        "Custom Input",
        "Anand Vihar (High Smog Hotspot)",
        "ITO Junction (Heavy Traffic)",
        "Lodhi Road (Greener Basin)",
        "RK Puram (Urban Residential)",
    ],
)

defaults = {
    "Custom Input": (260, 245, 175, 11),
    "Anand Vihar (High Smog Hotspot)": (410, 395, 310, 11),
    "ITO Junction (Heavy Traffic)": (320, 305, 230, 11),
    "Lodhi Road (Greener Basin)": (145, 140, 85, 3),
    "RK Puram (Urban Residential)": (280, 275, 190, 12),
}
init_aqi, init_lag2, init_pm25, init_month = defaults[station_preset]

st.sidebar.markdown("---")
input_aqi_lag1 = st.sidebar.slider(
    "Today's AQI (t-1)", 20, 500, init_aqi, step=5
)
input_aqi_lag2 = st.sidebar.slider(
    "Yesterday's AQI (t-2)", 20, 500, init_lag2, step=5
)
input_pm25_lag1 = st.sidebar.slider(
    "PM2.5 Concentration (µg/m³)", 10, 450, init_pm25, step=5
)
input_month = st.sidebar.selectbox(
    "Calendar Month",
    range(1, 13),
    index=init_month - 1,
    format_func=lambda m: pd.to_datetime(f"2023-{m:02d}-01").strftime("%B"),
)

# Improved rolling estimate based on input trends
# 3-day mean includes t-1, t-2, and an interpolated t-3
assumed_lag3 = max(20, (input_aqi_lag2 * 2) - input_aqi_lag1)
input_roll_3d = np.mean([input_aqi_lag1, input_aqi_lag2, assumed_lag3])

# 7-day mean: average of available points plus trend anchor
input_roll_7d = np.mean([
    input_aqi_lag1,
    input_aqi_lag2,
    assumed_lag3,
    assumed_lag3 - 5,
    assumed_lag3 - 10,
    assumed_lag3 - 15,
    assumed_lag3 - 20,
])
input_roll_7d = float(np.clip(input_roll_7d, 20, 500))
input_is_winter = 1 if input_month in [10, 11, 12, 1] else 0

# Vectorize input into Pandas DataFrame for Scikit-learn
input_vector = pd.DataFrame(
    [[
        input_aqi_lag1,
        input_aqi_lag2,
        input_pm25_lag1,
        input_roll_3d,
        input_roll_7d,
        input_month,
        input_is_winter,
    ]],
    columns=feature_names,
)

# Live Inference
pred_rf_val = float(rf_model.predict(input_vector)[0])
pred_ridge_val = float(ridge_pipe.predict(input_vector)[0])

# Correct 95% individual prediction interval (1.96 * residual std)
pi_bound = 1.96 * rf_residual_std

cat_label, cat_icon, cat_color, cat_advice = get_cpcb_category(pred_rf_val)
grap_title, grap_desc = get_grap_stage(pred_rf_val)

# ---------------------------------------------------------
# 4. MAIN LAYOUT: PREDICTION & DECISION ENGINE
# ---------------------------------------------------------
col_left, col_right = st.columns([1.2, 1.8])

with col_left:
  st.subheader("🎯 Tomorrow's Forecast")

  st.markdown(
      f"""
        <div style="background-color: {cat_color}18; border: 2px solid {cat_color}; border-radius: 12px; padding: 20px; text-align: center;">
            <h4 style="margin: 0; color: #4b5563;">PREDICTED AQI (RANDOM FOREST)</h4>
            <h1 style="font-size: 3.5rem; margin: 10px 0; color: {cat_color};">{pred_rf_val:.1f}</h1>
            <p style="font-size: 1.25rem; font-weight: bold; margin: 0;">{cat_icon} {cat_label}</p>
            <p style="color: #6b7280; font-size: 0.9rem; margin-top: 6px;">95% Expected Range: [{max(0, pred_rf_val - pi_bound):.1f} – {min(500, pred_rf_val + pi_bound):.1f}]</p>
        </div>
        """,
      unsafe_allow_html=True,
  )

  st.write("")
  st.info(f"**Health Impact:** {cat_advice}")
  st.warning(f"**Action Alert: {grap_title}**\n\n*{grap_desc}*")
  st.caption(
      f"**Model Consensus Check:** Ridge Baseline predicts"
      f" **{pred_ridge_val:.1f}** AQI (Δ = {abs(pred_rf_val - pred_ridge_val):.1f}"
      " pts)."
  )

with col_right:
  st.subheader("📈 7-Day Context & Trajectory")

  days_back = [f"t-{i}" for i in range(7, 0, -1)]
  past_vals = [
      max(20, input_aqi_lag1 - (i * 8) + np.random.randint(-10, 10))
      for i in range(6, 0, -
