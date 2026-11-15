"""Transparent operational inventory guidance for the PharmaStock demo."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def reorder_recommendation(
    current_stock: float,
    forecast_30d: float,
    daily_demand_std: float,
    lead_time_days: int = 7,
    review_period_days: int = 30,
    service_z: float = 1.65,
) -> dict[str, float | int | bool]:
    """Calculate reorder point and order-up-to quantity using forecast demand."""
    if current_stock < 0 or forecast_30d < 0 or daily_demand_std < 0:
        raise ValueError("Stock, forecast and variability must be non-negative")
    if lead_time_days < 0 or review_period_days < 1:
        raise ValueError("Lead time must be non-negative and review period positive")
    average_daily = forecast_30d / 30.0
    safety_stock = service_z * daily_demand_std * math.sqrt(max(lead_time_days, 0))
    reorder_point = average_daily * lead_time_days + safety_stock
    target_stock = average_daily * (lead_time_days + review_period_days) + safety_stock
    order_quantity = max(0.0, target_stock - current_stock)
    return {
        "average_daily_demand": average_daily,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "target_stock": target_stock,
        "suggested_order_quantity": order_quantity,
        "reorder_alert": bool(current_stock <= reorder_point),
    }


def allocate_stock_fefo(batches: pd.DataFrame, available_stock: float) -> pd.DataFrame:
    """Allocate product stock across batches by earliest expiry first.

    ``batch_quantity`` is used only as a proportional allocation weight because
    this synthetic dataset has no batch-level on-hand stock snapshots.
    """
    required = {"batch_id", "expiry_date", "batch_quantity"}
    if not required.issubset(batches.columns):
        raise ValueError(f"Batch data must contain {sorted(required)}")
    if available_stock < 0:
        raise ValueError("Available stock must be non-negative")
    result = batches.copy()
    weights = pd.to_numeric(result["batch_quantity"], errors="coerce").fillna(0).clip(lower=0)
    total = float(weights.sum())
    result["allocated_stock"] = (available_stock * weights / total) if total else 0.0
    result["expiry_date"] = pd.to_datetime(result["expiry_date"])
    return result.sort_values(["expiry_date", "batch_id"]).reset_index(drop=True)


def assess_batches(
    batches: pd.DataFrame,
    available_stock: float,
    forecast_30d: float,
    avg_daily_demand_30d: float,
    avg_daily_demand_90d: float,
    as_of: pd.Timestamp,
    expiry_artifact: dict[str, Any],
) -> pd.DataFrame:
    """Create FEFO-ordered, model-scored batch alerts for one product."""
    from src.inference import predict_expiry_risk

    allocated = allocate_stock_fefo(batches, available_stock)
    daily_demand = max(0.0, forecast_30d / 30.0)
    historical_daily_demand = max(0.0, float(avg_daily_demand_90d))
    rows = []
    for batch in allocated.to_dict("records"):
        days = max(0, int((pd.Timestamp(batch["expiry_date"]).normalize() - pd.Timestamp(as_of).normalize()).days))
        stock = float(batch["allocated_stock"])
        # Match the feature definition used to train the expiry classifier.
        model_expected_demand = historical_daily_demand * days
        forecast_demand_before_expiry = daily_demand * days
        model_result = predict_expiry_risk({
            "product_id": str(batch["product_id"]),
            "current_stock": stock,
            "days_to_expiry": days,
            "avg_daily_demand_30d": float(avg_daily_demand_30d),
            "avg_daily_demand_90d": float(avg_daily_demand_90d),
            "expected_demand": model_expected_demand,
            "stock_demand_ratio": stock / max(model_expected_demand, 1.0),
        }, expiry_artifact)
        row = dict(batch)
        row.update({
            "days_to_expiry": days,
            "expected_demand_before_expiry": forecast_demand_before_expiry,
            "estimated_unsold_units": max(0.0, stock - forecast_demand_before_expiry),
            **model_result,
        })
        rows.append(row)
    return pd.DataFrame(rows)
