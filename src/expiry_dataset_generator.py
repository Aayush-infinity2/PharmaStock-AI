
import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sales = pd.read_csv(
    os.path.join(BASE, "data/synthetic/daily_sales.csv"),
    parse_dates=["date"]
)

products = pd.read_csv(
    os.path.join(BASE, "data/synthetic/products.csv")
)

# Historical average demand for each product
sales["date"] = pd.to_datetime(sales["date"])



# Snapshot dates
snapshot_dates = pd.to_datetime([
    "2025-07-01",
    "2025-08-01",
    "2025-09-01",
    "2025-10-01",
    "2025-11-01"
])

records = []

for product in products.to_dict("records"):

    product_id = product["product_id"]



    # Generate 5 batches for every product and snapshot
    for snapshot_date in snapshot_dates:
        history_90 = sales[(sales["product_id"] == product_id) &(sales["date"] < snapshot_date) &(sales["date"] >= snapshot_date - pd.Timedelta(days=90))]
        history_30 = history_90[history_90["date"] >= snapshot_date - pd.Timedelta(days=30)]
        avg90 = history_90["quantity_sold"].mean()
        avg30 = history_30["quantity_sold"].mean()

        if pd.isna(avg90) or pd.isna(avg30):
            continue

        batch_data = []

        for batch_num in range(1, 6):

            expiry_days = int(
                rng.integers(30, 366)
            )

            expiry_date = (
                snapshot_date +
                pd.Timedelta(days=expiry_days)
            )

            stock = int(rng.integers(100, 1001))

            batch_data.append({
                "batch_id": (
                    f"{product_id}_"
                    f"{snapshot_date.strftime('%Y%m')}_B{batch_num}"
                ),
                "stock": stock,
                "expiry_date": expiry_date
            })

        # Simulate future demand until all batches expire.
        # FEFO: earliest expiry batch is consumed first.
        remaining = {
            b["batch_id"]: b["stock"]
            for b in batch_data
        }

        future_demand = []

        max_days = max(
            (b["expiry_date"] - snapshot_date).days
            for b in batch_data
        )

        for day in range(1, max_days + 1):

            date = snapshot_date + pd.Timedelta(days=day)

            seasonal_factor = (
                1.3 if date.month in [6, 7, 8]
                else 1.0
            )

            expected = (
                avg90
                * seasonal_factor
            )

            daily_demand = int(
                rng.poisson(expected)
            )

            future_demand.append({
                "date": date,
                "quantity": daily_demand
            })

            # Allocate demand using FEFO
            eligible = sorted(
                [
                    b for b in batch_data
                    if b["expiry_date"] >= date
                    and remaining[b["batch_id"]] > 0
                ],
                key=lambda x: x["expiry_date"]
            )

            demand_left = daily_demand

            for batch in eligible:

                batch_id = batch["batch_id"]

                consumed = min(
                    remaining[batch_id],
                    demand_left
                )

                remaining[batch_id] -= consumed
                demand_left -= consumed

                if demand_left <= 0:
                    break

        # Create one training row per batch
        for batch in batch_data:

            batch_id = batch["batch_id"]
            expiry_date = batch["expiry_date"]

            days_to_expiry = (
                expiry_date - snapshot_date
            ).days

            current_stock = batch["stock"]

            # Estimate demand using historical information
            expected_demand = (
                avg90
                * days_to_expiry
            )

            stock_demand_ratio = (
                current_stock / max(expected_demand, 1)
            )

            # Remaining stock at expiry from simulation
            unsold_quantity = remaining[batch_id]

            unsold_percentage = (
                unsold_quantity / current_stock
            )

            expiry_risk = int(
                unsold_percentage >= 0.10
            )

            records.append({

                "snapshot_date": snapshot_date,
                "product_id": product_id,
                "batch_id": batch_id,

                "current_stock": current_stock,
                "days_to_expiry": days_to_expiry,

                "avg_daily_demand_30d":
                    avg30,

                "avg_daily_demand_90d":
                    avg90,

                "expected_demand": expected_demand,

                "stock_demand_ratio":
                    stock_demand_ratio,

                "expiry_risk": expiry_risk
            })


expiry_df = pd.DataFrame(records)

output_dir = os.path.join(
    BASE, "data", "processed"
)

os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir, "expiry_risk_dataset.csv"
)

expiry_df.to_csv(output_path, index=False)

print("Expiry dataset generated!")
print("Total records:", len(expiry_df))
print("\nRisk distribution:")
print(expiry_df["expiry_risk"].value_counts())
print("\nRisk percentage:")
print(expiry_df["expiry_risk"].value_counts(normalize=True) * 100)
print("\nSaved to:", output_path)
