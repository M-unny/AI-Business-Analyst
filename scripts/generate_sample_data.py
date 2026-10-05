"""Create a synthetic retail dataset (with deliberate data-quality issues) at
datasets/sample_sales.csv so you can run the pipeline without downloading data."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

PRODUCTS = {
    "Laptop": ("Technology", 900, 0.72), "Phone": ("Technology", 600, 0.70),
    "Monitor": ("Technology", 250, 0.68), "Desk": ("Furniture", 320, 0.80),
    "Chair": ("Furniture", 180, 0.78), "Bookcase": ("Furniture", 210, 0.82),
    "Notebook": ("Office Supplies", 6, 0.55), "Printer Paper": ("Office Supplies", 25, 0.60),
    "Stapler": ("Office Supplies", 14, 0.58),
}
names = list(PRODUCTS)

dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, N), unit="D")
prod = rng.choice(names, N)
qty = rng.integers(1, 8, N)
unit_price = np.array([PRODUCTS[p][1] for p in prod]) * rng.uniform(0.9, 1.1, N)
discount = rng.choice([0, 0.05, 0.1, 0.2], N, p=[0.5, 0.2, 0.2, 0.1])
revenue = qty * unit_price * (1 - discount)
cost = qty * np.array([PRODUCTS[p][1] * PRODUCTS[p][2] for p in prod])

df = pd.DataFrame({
    "order_id": [f"ORD-{i:06d}" for i in range(N)],
    "order_date": dates.strftime("%Y-%m-%d"),
    "customer_id": [f"CUST-{c:04d}" for c in rng.integers(1, 600, N)],
    "product": prod,
    "category": [PRODUCTS[p][0] for p in prod],
    "region": rng.choice(["East", "West", "North", "South"], N, p=[0.35, 0.25, 0.2, 0.2]),
    "quantity": qty,
    "unit_price": unit_price.round(2),
    "discount": discount,
    "revenue": revenue.round(2),
    "cost": cost.round(2),
    "profit": (revenue - cost).round(2),
    "returned": rng.choice(["Yes", "No"], N, p=[0.04, 0.96]),
}).sort_values("order_date").reset_index(drop=True)

# deliberate quality problems
df.loc[rng.choice(N, 100, replace=False), "region"] = np.nan
df.loc[rng.choice(N, 60, replace=False), "discount"] = np.nan
df.loc[rng.choice(N, 5, replace=False), "revenue"] *= 12          # outliers
df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)   # duplicates

out = Path(__file__).resolve().parents[1] / "datasets" / "sample_sales.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df):,} rows to {out}")
