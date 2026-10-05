import numpy as np
import pandas as pd

from app.data.profiler import profile_dataset


def roles_of(df):
    return profile_dataset(df).semantic


def test_sales_roles():
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame({
        "Order ID": [f"A-{i}" for i in range(n)],
        "Order Date": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
        "CustomerID": rng.integers(1, 30, n),
        "Product Name": rng.choice(["a", "b", "c"], n),
        "Region": rng.choice(["E", "W"], n),
        "Qty": rng.integers(1, 5, n),
        "Total Sales": rng.uniform(5, 100, n),
        "Profit": rng.uniform(-10, 30, n),
    })
    s = roles_of(df)
    assert s.best_column("date") == "Order Date"
    assert s.best_column("order_id") == "Order ID"
    assert s.best_column("customer") == "CustomerID"
    assert s.best_column("product") == "Product Name"
    assert s.best_column("region") == "Region"
    assert s.best_column("quantity") == "Qty"
    assert s.best_column("revenue") == "Total Sales"
    assert s.best_column("profit") == "Profit"


def test_margin_is_not_profit_and_unit_price_is_not_quantity():
    n = 30
    df = pd.DataFrame({"profit_margin": np.linspace(0.1, 0.4, n), "unit_price": np.linspace(5, 50, n)})
    s = roles_of(df)
    assert s.primary["profit_margin"].role == "margin"
    assert s.primary["unit_price"].role == "price"


def test_date_detected_without_a_name_hint():
    df = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=30)})
    m = roles_of(df).primary["when"]
    assert m.role == "date" and 0.5 <= m.confidence < 1.0


def test_name_alone_is_not_enough():
    # 'revenue' holds text labels -> must not be treated as the revenue measure
    df = pd.DataFrame({"revenue": ["high", "low", "medium"] * 10})
    assert "revenue" not in roles_of(df).roles


def test_implausible_values_reject_role():
    df = pd.DataFrame({"age": np.linspace(500, 900, 30)})
    assert "age" not in roles_of(df).roles


def test_plausible_age_accepted():
    df = pd.DataFrame({"customer_age": np.linspace(18, 80, 30)})
    assert roles_of(df).best_column("age") == "customer_age"


def test_target_requires_binary_column():
    df = pd.DataFrame({"Churn": ["Yes", "No"] * 20, "tenure_months": range(40)})
    s = roles_of(df)
    assert s.best_column("target") == "Churn"
    assert s.best_column("tenure") == "tenure_months"


def test_marketing_roles():
    n = 30
    df = pd.DataFrame({
        "campaign": ["c1", "c2", "c3"] * 10, "ad_spend": np.linspace(100, 500, n),
        "clicks": np.arange(n) * 10, "impressions": np.arange(n) * 1000,
        "conversions": np.arange(n), "click_through_rate": np.linspace(0.01, 0.05, n),
    })
    s = roles_of(df)
    for role, c in [("campaign", "campaign"), ("spend", "ad_spend"), ("clicks", "clicks"),
                    ("impressions", "impressions"), ("conversions", "conversions")]:
        assert s.best_column(role) == c
    assert "click_through_rate" not in s.primary or s.primary["click_through_rate"].role != "clicks"


def test_unmatched_columns_reported():
    df = pd.DataFrame({"zzz": range(30), "qqq": ["a", "b"] * 15})
    s = roles_of(df)
    assert set(s.unmatched_columns) == {"zzz", "qqq"}
