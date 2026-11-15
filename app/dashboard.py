"""Streamlit inference interface for PharmaStock AI."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.inference import ROOT as PROJECT_ROOT, build_demand_features, load_artifact, predict_demand
from src.inventory_optimizer import assess_batches, reorder_recommendation

st.set_page_config(page_title="PharmaStock AI", page_icon="💊", layout="wide")

@st.cache_data
def load_data():
    sales = pd.read_csv(PROJECT_ROOT / "data/synthetic/daily_sales.csv", parse_dates=["date"])
    products = pd.read_csv(PROJECT_ROOT / "data/synthetic/products.csv")
    snapshots = pd.read_csv(PROJECT_ROOT / "data/synthetic/inventory_snapshots.csv", parse_dates=["snapshot_date"])
    batches = pd.read_csv(PROJECT_ROOT / "data/synthetic/batch_inventory.csv", parse_dates=["expiry_date"])
    return sales, products, snapshots, batches

@st.cache_resource
def load_models():
    return load_artifact("demand_model.pkl"), load_artifact("expiry_risk_model.pkl")

st.title("PharmaStock AI")
st.caption("Medicine demand forecasting and expiry-aware inventory decision support")
st.warning("Demonstration uses synthetic data. Predictions are advisory and require human review. They are not medication or treatment advice. Batch-level on-hand quantities are estimated by allocating synthetic product stock in FEFO order because the dataset has no batch stock snapshots.")
try:
    sales, products, snapshots, batches = load_data()
    demand_artifact, expiry_artifact = load_models()
except Exception as exc:
    st.error(f"Could not load project data or model artifacts: {exc}")
    st.stop()

product_options = products["product_id"].astype(str).tolist()
product_id = st.selectbox("Medicine", product_options, format_func=lambda pid: f"{pid} — {products.loc[products.product_id.astype(str) == pid, 'medicine_name'].iloc[0]}")
product = products.loc[products.product_id.astype(str) == product_id].iloc[0]
forecast_origin = pd.to_datetime(sales["date"]).max().normalize() + pd.Timedelta(days=1)
feature_row = build_demand_features(sales, product_id, str(product["category"]), forecast_origin)
forecast = predict_demand(feature_row, demand_artifact)
snapshot_rows = snapshots.loc[snapshots.product_id.astype(str) == product_id].sort_values("snapshot_date")
if snapshot_rows.empty:
    st.error("No inventory snapshot is available for this medicine.")
    st.stop()
snapshot = snapshot_rows.iloc[-1]
as_of = pd.Timestamp(snapshot["snapshot_date"])
current_stock = float(snapshot["available_stock"])
lead_time = st.sidebar.number_input("Supplier lead time (days)", min_value=0, max_value=90, value=7)
inventory = reorder_recommendation(current_stock, forecast, float(feature_row["std_daily_30d"]), int(lead_time))

col1, col2, col3, col4 = st.columns(4)
col1.metric("30-day demand forecast", f"{forecast:,.0f} units")
col2.metric("Available product stock", f"{current_stock:,.0f} units", help=f"Synthetic inventory snapshot: {as_of.date()}")
col3.metric("Reorder point", f"{inventory['reorder_point']:,.0f} units")
col4.metric("Suggested order", f"{inventory['suggested_order_quantity']:,.0f} units")
if inventory["reorder_alert"]:
    st.info("Reorder review suggested: available stock is at or below the calculated reorder point.")
else:
    st.success("No reorder alert at the selected lead time.")

st.subheader("Demand forecast")
st.write(f"Saved {demand_artifact.get('model_name', 'demand model')} predicts **{forecast:,.1f} units** for the 30 days after {forecast_origin.date()}. Inputs include recent sales, demand variability, month, category and product identity.")
with st.expander("Forecast inputs"):
    st.json(feature_row)

st.subheader("Batch expiry and FEFO review")
product_batches = batches.loc[batches.product_id.astype(str) == product_id].copy()
product_batches = product_batches.loc[
    (pd.to_datetime(product_batches["manufacture_date"]) <= as_of)
    & (pd.to_datetime(product_batches["expiry_date"]) >= as_of)
]
if product_batches.empty:
    st.info("No synthetic batch records are available for this medicine.")
else:
    assessment = assess_batches(product_batches, current_stock, forecast, feature_row["avg_daily_30d"], feature_row["sales_last_90d"] / 90.0, as_of, expiry_artifact)
    assessment["expiry_date"] = pd.to_datetime(assessment["expiry_date"]).dt.date
    assessment["risk_probability"] = assessment["risk_probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(assessment[["batch_id", "expiry_date", "days_to_expiry", "allocated_stock", "expected_demand_before_expiry", "estimated_unsold_units", "risk_probability", "risk_flag"]].rename(columns={"allocated_stock": "Estimated stock allocation", "expected_demand_before_expiry": "Demand forecast before expiry", "estimated_unsold_units": "Potential unsold units", "risk_probability": "Model risk probability", "risk_flag": "Model risk flag"}), width="stretch", hide_index=True)
    st.write(f"Review **{int(assessment['risk_flag'].sum())} of {len(assessment)}** batches flagged by the saved classifier. Batches are ordered by expiry date (FEFO).")

st.subheader("Held-out evaluation")
for title, path in [("Demand forecasting test metrics", PROJECT_ROOT / "reports/demand_model_metrics.csv"), ("Expiry-risk test metrics", PROJECT_ROOT / "reports/expiry_model_metrics.csv")]:
    if path.exists():
        st.write(title)
        st.dataframe(pd.read_csv(path).rename(columns={"Unnamed: 0": "Model"}), hide_index=True, width="stretch")
st.caption("Synthetic data and simulated labels do not establish real-world pharmacy performance. Validate with licensed operational data before use.")
