import numpy as np
import pandas as pd
from src.inference import build_demand_features, predict_demand, predict_expiry_risk
from src.inventory_optimizer import allocate_stock_fefo, reorder_recommendation

def test_saved_demand_model_returns_nonnegative_prediction():
    row = {"sales_last_7d":220,"sales_last_30d":900,"sales_last_90d":2700,"avg_daily_30d":30,"std_daily_30d":5,"month":12,"category":"Analgesic","product_id":"MED001"}
    result = predict_demand(row)
    assert np.isfinite(result) and result >= 0

def test_saved_expiry_model_returns_probability_and_flag():
    row = {"product_id":"MED001","current_stock":150,"days_to_expiry":60,"avg_daily_demand_30d":30,"avg_daily_demand_90d":29,"expected_demand":1800,"stock_demand_ratio":150/1800}
    result = predict_expiry_risk(row)
    assert 0 <= result["risk_probability"] <= 1
    assert result["risk_flag"] in (0, 1)

def test_demand_features_exclude_forecast_origin_and_future():
    sales = pd.DataFrame({"date":pd.to_datetime(["2025-01-01","2025-01-02","2025-01-03"]),"product_id":["MED001"]*3,"quantity_sold":[2,3,1000]})
    result = build_demand_features(sales,"MED001","Analgesic",pd.Timestamp("2025-01-03"))
    assert result["sales_last_90d"] == 5

def test_fefo_orders_earliest_expiry_and_reconciles_stock():
    batches = pd.DataFrame({"batch_id":["later","sooner"],"product_id":["MED001"]*2,"expiry_date":["2026-12-01","2026-06-01"],"batch_quantity":[3,1]})
    result = allocate_stock_fefo(batches,400)
    assert result.iloc[0]["batch_id"] == "sooner"
    assert result["allocated_stock"].sum() == 400

def test_reorder_quantity_is_zero_above_target():
    result = reorder_recommendation(1000,100,2,lead_time_days=7)
    assert result["suggested_order_quantity"] == 0
    assert not result["reorder_alert"]
