
import os
import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Output directory
OUTPUT_DIR = "data/synthetic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Medicine master data
medicines = [
    "Paracetamol", "Amoxicillin", "Cetirizine",
    "Azithromycin", "Ibuprofen", "Metformin",
    "Pantoprazole", "Vitamin C", "Omeprazole",
    "Aspirin"
]

# Generate 100 product IDs
products = []

for i in range(100):
    products.append({
        "product_id": f"MED{i+1:03d}",
        "medicine_name": medicines[i % len(medicines)],
        "category": ["Analgesic", "Antibiotic",
                     "Antihistamine", "Vitamin",
                     "Chronic"][i % 5],
        "unit_price": np.random.randint(20, 500),
        "daily_base_demand": np.random.randint(5, 40)
    })

products_df = pd.DataFrame(products)

# 1. Generate daily sales
dates = pd.date_range(
    start="2024-01-01",
    end="2025-12-31",
    freq="D"
)

sales = []

for product in products:
    for date in dates:

        seasonal_factor = (
            1.3 if date.month in [6, 7, 8] else 1.0
        )

        weekend_factor = (
            0.85 if date.dayofweek >= 5 else 1.0
        )

        demand = (
            product["daily_base_demand"]
            * seasonal_factor
            * weekend_factor
        )

        quantity = np.random.poisson(demand)

        sales.append({
            "date": date,
            "product_id": product["product_id"],
            "medicine_name": product["medicine_name"],
            "quantity_sold": quantity,
            "unit_price": product["unit_price"],
            "revenue": quantity * product["unit_price"]
        })

sales_df = pd.DataFrame(sales)

# 2. Generate batch inventory
batches = []

for product in products:

    for batch_no in range(1, 4):

        manufacture_date = (
            pd.Timestamp("2025-01-01")
            + pd.Timedelta(days=np.random.randint(0, 365))
        )

        expiry_date = (
            manufacture_date
            + pd.Timedelta(days=np.random.choice([365, 540, 730]))
        )

        batches.append({
            "batch_id": f"{product['product_id']}_B{batch_no}",
            "product_id": product["product_id"],
            "medicine_name": product["medicine_name"],
            "batch_quantity": np.random.randint(100, 1000),
            "manufacture_date": manufacture_date,
            "expiry_date": expiry_date
        })

batches_df = pd.DataFrame(batches)

# 3. Generate inventory snapshots
snapshots = []

for product in products:

    for snapshot_date in pd.date_range(
        "2025-07-01", "2025-12-01", freq="MS"
    ):

        snapshots.append({
            "snapshot_date": snapshot_date,
            "product_id": product["product_id"],
            "medicine_name": product["medicine_name"],
            "available_stock": np.random.randint(50, 1500),
            "reorder_level": np.random.randint(100, 300)
        })

snapshots_df = pd.DataFrame(snapshots)

# Save CSV files
products_df.to_csv(
    f"{OUTPUT_DIR}/products.csv", index=False
)

sales_df.to_csv(
    f"{OUTPUT_DIR}/daily_sales.csv", index=False
)

batches_df.to_csv(
    f"{OUTPUT_DIR}/batch_inventory.csv", index=False
)

snapshots_df.to_csv(
    f"{OUTPUT_DIR}/inventory_snapshots.csv", index=False
)

print("Dataset generation completed!")

print("Products:", len(products_df))
print("Sales records:", len(sales_df))
print("Batch records:", len(batches_df))
print("Inventory snapshots:", len(snapshots_df))
