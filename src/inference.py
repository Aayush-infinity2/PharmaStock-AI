"""Load saved PharmaStock models and turn operational inputs into predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_artifact(name: str) -> dict[str, Any]:
    """Load one of the project model artifacts from ``models``."""
    if name not in {"demand_model.pkl", "expiry_risk_model.pkl"}:
        raise ValueError(f"Unsupported model artifact: {name}")
    path = ROOT / "models" / name
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    artifact = joblib.load(path)
    required = {"model", "feature_columns"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError(f"Invalid model artifact: {path}")
    return artifact


def _aligned_features(record: Mapping[str, Any], columns: list[str]) -> pd.DataFrame:
    """Reproduce training-time one-hot encoding and align to saved columns."""
    raw = pd.DataFrame([dict(record)])
    encoded = pd.get_dummies(raw)
    return encoded.reindex(columns=columns, fill_value=0)


def predict_demand(record: Mapping[str, Any], artifact: dict[str, Any] | None = None) -> float:
    """Predict total demand for the trained 30-day horizon."""
    artifact = artifact or load_artifact("demand_model.pkl")
    required = {
        "sales_last_7d", "sales_last_30d", "sales_last_90d", "avg_daily_30d",
        "std_daily_30d", "month", "category", "product_id",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Missing demand input fields: {', '.join(sorted(missing))}")
    features = _aligned_features(record, list(artifact["feature_columns"]))
    prediction = float(np.asarray(artifact["model"].predict(features)).reshape(-1)[0])
    return max(0.0, prediction)


def predict_expiry_risk(
    record: Mapping[str, Any], artifact: dict[str, Any] | None = None
) -> dict[str, float | int]:
    """Return saved-classifier risk probability and thresholded advisory flag."""
    artifact = artifact or load_artifact("expiry_risk_model.pkl")
    required = {
        "product_id", "current_stock", "days_to_expiry", "avg_daily_demand_30d",
        "avg_daily_demand_90d", "expected_demand", "stock_demand_ratio",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Missing expiry input fields: {', '.join(sorted(missing))}")
    features = _aligned_features(record, list(artifact["feature_columns"]))
    probability = float(artifact["model"].predict_proba(features)[0, 1])
    threshold = float(artifact.get("threshold", 0.5))
    return {
        "risk_probability": probability,
        "risk_flag": int(probability >= threshold),
        "threshold": threshold,
    }


def build_demand_features(
    sales: pd.DataFrame, product_id: str, category: str, origin: pd.Timestamp
) -> dict[str, Any]:
    """Build leakage-safe 7/30/90-day features ending before forecast origin."""
    required = {"date", "product_id", "quantity_sold"}
    if not required.issubset(sales.columns):
        raise ValueError(f"Sales data must contain {sorted(required)}")
    history = sales.loc[
        (sales["product_id"].astype(str) == str(product_id))
        & (pd.to_datetime(sales["date"]) < pd.Timestamp(origin))
    ].copy()
    daily = history.groupby(pd.to_datetime(history["date"]).dt.normalize())["quantity_sold"].sum()
    days = pd.date_range(end=pd.Timestamp(origin).normalize() - pd.Timedelta(days=1), periods=90)
    daily = daily.reindex(days, fill_value=0)
    last7, last30, last90 = daily.tail(7), daily.tail(30), daily.tail(90)
    return {
        "sales_last_7d": float(last7.sum()),
        "sales_last_30d": float(last30.sum()),
        "sales_last_90d": float(last90.sum()),
        "avg_daily_30d": float(last30.mean()),
        "std_daily_30d": float(last30.std(ddof=1) or 0.0),
        "month": int(pd.Timestamp(origin).month),
        "category": str(category),
        "product_id": str(product_id),
    }
