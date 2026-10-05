import json

import numpy as np
import pandas as pd
import pytest

from app.data.profiler import profile_column, profile_dataset


def col(profile, name):
    return next(c for c in profile.columns if c.name == name)


@pytest.fixture
def sales_df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "order_id": [f"ORD-{i:05d}" for i in range(n)],
        "order_date": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
        "customer_id": rng.integers(1000, 1050, n),
        "region": rng.choice(["East", "West", "North"], n),
        "quantity": rng.integers(1, 10, n),
        "revenue": rng.uniform(10, 500, n).round(2),
        "returned": rng.choice(["Yes", "No"], n),
        "comment": ["This is a long free-text comment written by a customer about the order " + str(i)
                    for i in range(n)],
    })


def test_type_inference(sales_df):
    p = profile_dataset(sales_df)
    assert col(p, "order_id").inferred_type == "id"
    assert col(p, "order_date").inferred_type == "datetime"
    assert col(p, "order_date").date_format == "iso"
    assert col(p, "customer_id").inferred_type == "id"        # numeric dtype, id-like name
    assert col(p, "region").inferred_type == "categorical"
    assert col(p, "quantity").inferred_type == "numeric"
    assert col(p, "quantity").integer_like is True
    assert col(p, "revenue").inferred_type == "numeric"
    assert col(p, "returned").inferred_type == "boolean"
    assert col(p, "comment").inferred_type == "text"


def test_metadata_matches_phase2_shape(sales_df):
    m = profile_dataset(sales_df).metadata.model_dump()
    for key in ("rows", "columns", "numeric_columns", "categorical_columns",
                "date_columns", "missing_columns", "duplicate_rows"):
        assert key in m
    assert m["rows"] == 200 and m["columns"] == 8
    assert m["date_columns"] == ["order_date"]


def test_numeric_stats():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    c = profile_dataset(df).columns[0]
    assert c.min_value == 1 and c.max_value == 100
    assert c.mean == pytest.approx(22.0) and c.median == 3
    assert c.outlier_count == 1


def test_missing_and_duplicates():
    df = pd.DataFrame({"a": [1, 2, 2, None], "b": ["x", "y", "y", "z"]})
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)       # exact duplicate row
    p = profile_dataset(df)
    assert p.metadata.duplicate_rows == 2                        # rows 1,2 and the appended copy
    assert p.metadata.missing_columns == ["a"]
    assert col(p, "a").missing_count == 1
    assert col(p, "a").missing_pct == pytest.approx(20.0)
    assert any("duplicate" in w for w in p.warnings)


def test_ambiguous_dates_are_flagged_not_guessed():
    df = pd.DataFrame({"d": ["01/02/2024", "03/04/2024", "05/06/2024", "07/08/2024"] * 5})
    p = profile_dataset(df)
    c = p.columns[0]
    assert c.inferred_type == "datetime" and c.date_format == "ambiguous"
    assert any("ambiguous" in w for w in p.warnings)


def test_dayfirst_is_resolved_from_evidence():
    df = pd.DataFrame({"d": ["25/01/2024", "03/02/2024", "04/03/2024"] * 5})
    c = profile_dataset(df).columns[0]
    assert c.date_format == "dayfirst"
    assert c.min_value.startswith("2024-01-25")


def test_monthfirst_is_resolved_from_evidence():
    df = pd.DataFrame({"d": ["01/25/2024", "02/03/2024", "03/04/2024"] * 5})
    c = profile_dataset(df).columns[0]
    assert c.date_format == "monthfirst"
    assert c.min_value.startswith("2024-01-25")


def test_text_dates_with_month_names():
    df = pd.DataFrame({"d": ["Jan 5, 2024", "Feb 10, 2024", "Mar 15, 2024"] * 5})
    assert profile_dataset(df).columns[0].inferred_type == "datetime"


def test_year_like_integers_are_not_dates():
    df = pd.DataFrame({"year": ["2019", "2020", "2021", "2022"] * 5})
    assert profile_dataset(df).columns[0].inferred_type == "numeric"


def test_numbers_stored_as_text_are_detected():
    df = pd.DataFrame({"price": ["$1,200.50", "$300", "1,000", "(50)"] * 5})
    c = profile_dataset(df).columns[0]
    assert c.inferred_type == "numeric" and c.stored_as_text
    assert c.max_value == 1200.5 and c.min_value == -50


def test_leading_zero_codes_are_not_numeric():
    df = pd.DataFrame({"zip_text": ["00123", "04567", "00999", "01234"] * 5})
    assert profile_dataset(df).columns[0].inferred_type != "numeric"


def test_empty_and_constant_columns():
    df = pd.DataFrame({"empty": [None] * 5, "const": ["a"] * 5, "v": range(5)})
    p = profile_dataset(df)
    assert col(p, "empty").inferred_type == "empty"
    assert any("constant" in n for n in col(p, "const").notes)


def test_boolean_variants():
    for values in (["Y", "N"], ["true", "false"], [0, 1], [True, False]):
        df = pd.DataFrame({"flag": values * 5})
        assert profile_dataset(df).columns[0].inferred_type == "boolean", values


def test_pandas_string_dtype_supported():
    s = pd.Series(["a", "b", "a", "c"] * 10, dtype="string", name="grp")
    assert profile_column(s, len(s)).inferred_type == "categorical"


def test_input_is_not_mutated(sales_df):
    before = sales_df.copy()
    profile_dataset(sales_df)
    pd.testing.assert_frame_equal(before, sales_df)


def test_quality_score_bounds():
    clean = pd.DataFrame({"a": range(50), "b": [f"v{i % 5}" for i in range(50)]})
    dirty = pd.DataFrame({"a": [None] * 50, "b": ["x"] * 50})
    assert profile_dataset(clean).quality_score > 95
    assert 0 <= profile_dataset(dirty).quality_score < profile_dataset(clean).quality_score


def test_profile_is_json_serialisable(sales_df):
    json.dumps(profile_dataset(sales_df).model_dump(mode="json"))
