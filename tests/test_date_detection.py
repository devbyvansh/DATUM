"""Regression tests for dataset-aware date detection.

Covers 12 required cases:
1. Genuine datetime column with all valid dates.
2. Four valid dates and one invalid date among five non-null values.
3. Numeric prices such as 29.99, 45.50, and 12.00.
4. Integer IDs and large numeric identifiers.
5. Numeric-only dataset with no genuine date field.
6. Date-like strings mixed with invalid strings.
7. Date columns containing missing values.
8. Dataset with no date column: trend analysis returns a clear error.
9. Dataset with a valid date column: time-based chart uses correct dates and values.
10. Two different datasets uploaded sequentially: no cross-dataset leakage.
11. Switching between datasets does not retain an invalid date selection.
12. Existing upload/Ask/chart/Prepare tests continue to pass.
"""

from io import BytesIO

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.analysis_chat import answer_question
from backend.app.data_processing import _eda_charts, _is_likely_date_column
from backend.app.errors import DatumError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_DATES_CSV = (
    b"order_date,amount\n"
    b"2024-01-15,100\n"
    b"2024-02-20,200\n"
    b"2024-03-10,300\n"
    b"2024-04-05,400\n"
    b"2024-05-12,500\n"
)

FOUR_VALID_ONE_INVALID_CSV = (
    b"event_date,amount\n"
    b"2024-01-15,100\n"
    b"2024-02-20,200\n"
    b"2024-03-10,300\n"
    b"2024-04-05,400\n"
    b"not-a-date,500\n"
)

PRICES_CSV = (
    b"product,price\n"
    b"Widget,29.99\n"
    b"Gadget,45.50\n"
    b"Thing,12.00\n"
)

INTEGER_IDS_CSV = (
    b"id,value\n"
    b"1,100\n"
    b"2,200\n"
    b"3,300\n"
    b"1000000,999\n"
)

NUMERIC_ONLY_CSV = (
    b"a,b,c\n"
    b"1,2,3\n"
    b"4,5,6\n"
    b"7,8,9\n"
)

MIXED_DATE_INVALID_CSV = (
    b"col,value\n"
    b"2024-01-01,10\n"
    b"hello,20\n"
    b"2024-03-15,30\n"
    b"not-a-date,40\n"
    b"2024-06-01,50\n"
)

DATE_WITH_MISSING_CSV = (
    b"date,amount\n"
    b"2024-01-15,100\n"
    b",200\n"
    b"2024-03-10,300\n"
    b"2024-04-05,400\n"
    b",500\n"
)

NO_DATE_CSV = (
    b"product,category,price\n"
    b"Widget,A,29.99\n"
    b"Gadget,B,45.50\n"
    b"Thing,A,12.00\n"
)


def _make_client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(main.settings, "prepared_dir", tmp_path / "prepared")
    monkeypatch.setattr(main.settings, "report_dir", tmp_path / "reports")
    for directory in (
        main.settings.upload_dir,
        main.settings.prepared_dir,
        main.settings.report_dir,
    ):
        directory.mkdir()
    return TestClient(main.app)


def _upload(client, filename, content):
    return client.post(
        "/api/datasets", files={"file": (filename, BytesIO(content), "text/csv")}
    )


# ---------------------------------------------------------------------------
# Unit-level: _is_likely_date_column
# ---------------------------------------------------------------------------

def test_is_likely_date_all_valid_dates():
    series = pd.Series(
        ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-05-12"]
    )
    assert _is_likely_date_column(series) is True


def test_is_likely_date_four_valid_one_invalid():
    series = pd.Series(
        ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "not-a-date"]
    )
    assert _is_likely_date_column(series) is True


def test_is_likely_date_numeric_prices_not_dates():
    series = pd.Series([29.99, 45.50, 12.00])
    assert _is_likely_date_column(series) is False


def test_is_likely_date_integer_ids_not_dates():
    series = pd.Series([1, 2, 3, 1000000])
    assert _is_likely_date_column(series) is False


def test_is_likely_date_numeric_only_not_dates():
    series = pd.Series([1, 2, 3])
    assert _is_likely_date_column(series) is False


def test_is_likely_date_mixed_date_invalid():
    # 3 of 5 = 0.6 < 0.8, not a date column
    series = pd.Series(["2024-01-01", "hello", "2024-03-15", "not-a-date", "2024-06-01"])
    assert _is_likely_date_column(series) is False


def test_is_likely_date_with_missing_values():
    # 3 valid dates out of 3 non-null = 1.0 >= 0.8
    series = pd.Series(
        ["2024-01-15", None, "2024-03-10", "2024-04-05", None], dtype=object
    )
    assert _is_likely_date_column(series) is True


def test_is_likely_date_column_name_not_used():
    # A column named "date" but containing prices must not be a date
    series = pd.Series([29.99, 45.50, 12.00], name="date")
    assert _is_likely_date_column(series) is False


# ---------------------------------------------------------------------------
# Unit-level: _eda_charts trend detection
# ---------------------------------------------------------------------------

def test_eda_charts_no_date_column_no_trend():
    frame = pd.DataFrame({"product": ["Widget", "Gadget"], "price": [29.99, 45.50]})
    numeric = frame.select_dtypes(include="number")
    charts = _eda_charts(frame, numeric)
    trend_charts = [c for c in charts if c["type"] == "line"]
    assert trend_charts == [], f"Expected no trend charts, got {trend_charts}"


def test_eda_charts_valid_date_column_produces_trend():
    frame = pd.DataFrame(
        {"date": ["2024-01-15", "2024-02-20", "2024-03-10"], "amount": [100, 200, 300]}
    )
    numeric = frame.select_dtypes(include="number")
    charts = _eda_charts(frame, numeric)
    trend_charts = [c for c in charts if c["type"] == "line"]
    assert len(trend_charts) == 1
    assert trend_charts[0]["title"] == "amount over date"
    dates = [row["x"] for row in trend_charts[0]["data"]]
    # ISO format may include time component; check date prefix
    assert [d[:10] for d in dates] == ["2024-01-15", "2024-02-20", "2024-03-10"]
    values = [row["y"] for row in trend_charts[0]["data"]]
    assert values == [100.0, 200.0, 300.0]


# ---------------------------------------------------------------------------
# API-level: schema endpoint
# ---------------------------------------------------------------------------

def test_schema_all_valid_dates(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    resp = _upload(client, "dates.csv", ALL_DATES_CSV)
    assert resp.status_code == 201
    dataset_id = resp.json()["id"]
    schema = client.get(f"/api/datasets/{dataset_id}/schema").json()
    kinds = {col["name"]: col["kind"] for col in schema["schema"]}
    assert kinds["order_date"] == "datetime"
    assert kinds["amount"] == "numeric"


def test_schema_four_valid_one_invalid(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    resp = _upload(client, "mixed.csv", FOUR_VALID_ONE_INVALID_CSV)
    assert resp.status_code == 201
    dataset_id = resp.json()["id"]
    schema = client.get(f"/api/datasets/{dataset_id}/schema").json()
    kinds = {col["name"]: col["kind"] for col in schema["schema"]}
    assert kinds["event_date"] == "datetime"
    assert kinds["amount"] == "numeric"


def test_schema_numeric_prices_not_dates(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    resp = _upload(client, "prices.csv", PRICES_CSV)
    assert resp.status_code == 201
    dataset_id = resp.json()["id"]
    schema = client.get(f"/api/datasets/{dataset_id}/schema").json()
    kinds = {col["name"]: col["kind"] for col in schema["schema"]}
    assert kinds["price"] == "numeric"
    assert kinds["product"] == "categorical"


def test_schema_integer_ids_not_dates(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    resp = _upload(client, "ids.csv", INTEGER_IDS_CSV)
    assert resp.status_code == 201
    dataset_id = resp.json()["id"]
    schema = client.get(f"/api/datasets/{dataset_id}/schema").json()
    kinds = {col["name"]: col["kind"] for col in schema["schema"]}
    assert kinds["id"] == "numeric"
    assert kinds["value"] == "numeric"


def test_schema_numeric_only_no_dates(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    resp = _upload(client, "numeric.csv", NUMERIC_ONLY_CSV)
    assert resp.status_code == 201
    dataset_id = resp.json()["id"]
    schema = client.get(f"/api/datasets/{dataset_id}/schema").json()
    kinds = {col["name"]: col["kind"] for col in schema["schema"]}
    assert all(k == "numeric" for k in kinds.values())


def test_schema_mixed_date_invalid_below_threshold(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    resp = _upload(client, "mixed_invalid.csv", MIXED_DATE_INVALID_CSV)
    assert resp.status_code == 201
    dataset_id = resp.json()["id"]
    schema = client.get(f"/api/datasets/{dataset_id}/schema").json()
    kinds = {col["name"]: col["kind"] for col in schema["schema"]}
    # 3 of 5 = 0.6 < 0.8, so not a date
    assert kinds["col"] == "categorical"


def test_schema_date_with_missing_values(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    resp = _upload(client, "date_missing.csv", DATE_WITH_MISSING_CSV)
    assert resp.status_code == 201
    dataset_id = resp.json()["id"]
    schema = client.get(f"/api/datasets/{dataset_id}/schema").json()
    kinds = {col["name"]: col["kind"] for col in schema["schema"]}
    assert kinds["date"] == "datetime"


# ---------------------------------------------------------------------------
# Chat-level: _date_column / answer_question
# ---------------------------------------------------------------------------

def test_chat_no_date_column_raises_clear_error():
    frame = pd.DataFrame({"product": ["Widget", "Gadget"], "price": [29.99, 45.50]})
    with pytest.raises(DatumError, match="No suitable date or time column"):
        answer_question("run-1", "ds-1", frame, "Show monthly price trend")


def test_chat_valid_date_column_trend_uses_actual_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame(
        {"order_date": ["2024-01-15", "2024-02-20", "2024-03-10"], "amount": [100, 200, 300]}
    )
    result = answer_question("run-1", "ds-1", frame, "Show monthly amount")
    assert result["chart"]["type"] == "line"
    assert result["chart"]["columns"] == ["order_date", "amount"]
    data_map = {row["x"]: row["y"] for row in result["chart"]["data"]}
    # Question groups by month: 2024-01, 2024-02, 2024-03
    assert data_map.get("2024-01") == 100.0
    assert data_map.get("2024-02") == 200.0
    assert data_map.get("2024-03") == 300.0


# ---------------------------------------------------------------------------
# Integration-level: no cross-dataset leakage
# ---------------------------------------------------------------------------

def test_no_cross_dataset_leakage(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)

    # Upload dataset A (has date column)
    resp_a = _upload(client, "sales.csv", ALL_DATES_CSV)
    assert resp_a.status_code == 201
    id_a = resp_a.json()["id"]

    # Upload dataset B (no date column, numeric prices)
    resp_b = _upload(client, "prices.csv", PRICES_CSV)
    assert resp_b.status_code == 201
    id_b = resp_b.json()["id"]
    assert id_a != id_b

    # Schema for A: date column is datetime
    schema_a = client.get(f"/api/datasets/{id_a}/schema").json()
    kinds_a = {col["name"]: col["kind"] for col in schema_a["schema"]}
    assert kinds_a["order_date"] == "datetime"

    # Schema for B: price is numeric, NOT datetime
    schema_b = client.get(f"/api/datasets/{id_b}/schema").json()
    kinds_b = {col["name"]: col["kind"] for col in schema_b["schema"]}
    assert kinds_b["price"] == "numeric"

    # Preview for each shows its own columns
    preview_a = client.get(f"/api/datasets/{id_a}/preview").json()
    preview_b = client.get(f"/api/datasets/{id_b}/preview").json()
    assert "order_date" in preview_a["columns"]
    assert "order_date" not in preview_b["columns"]
    assert "price" in preview_b["columns"]

    # Ask on A (date column) produces a trend chart
    workflow_a = client.post(f"/api/datasets/{id_a}/workflows", json={"mode": "analyze_only"})
    run_id_a = workflow_a.json()["run_id"]
    # Wait for completion
    import time
    for _ in range(50):
        run_a = client.get(f"/api/runs/{run_id_a}").json()
        if run_a["status"] == "completed":
            break
        time.sleep(0.1)
    assert run_a["status"] == "completed"

    ask_a = client.post(
        f"/api/runs/{run_id_a}/analysis/chat",
        json={"question": "Show monthly amount"},
    )
    assert ask_a.status_code == 200
    assert ask_a.json()["chart"]["type"] == "line"

    # Ask on B (no date column) must NOT produce a trend chart
    workflow_b = client.post(f"/api/datasets/{id_b}/workflows", json={"mode": "analyze_only"})
    run_id_b = workflow_b.json()["run_id"]
    for _ in range(50):
        run_b = client.get(f"/api/runs/{run_id_b}").json()
        if run_b["status"] == "completed":
            break
        time.sleep(0.1)
    assert run_b["status"] == "completed"

    ask_b = client.post(
        f"/api/runs/{run_id_b}/analysis/chat",
        json={"question": "Show monthly price trend"},
    )
    assert ask_b.status_code == 400  # no_time_dimension error


# ---------------------------------------------------------------------------
# Existing behavior preserved
# ---------------------------------------------------------------------------

def test_upload_and_preview_still_works(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    resp = _upload(client, "sample.csv", ALL_DATES_CSV)
    assert resp.status_code == 201
    dataset_id = resp.json()["id"]
    preview = client.get(f"/api/datasets/{dataset_id}/preview")
    assert preview.status_code == 200
    assert preview.json()["total_rows"] == 5


def test_chart_build_still_works(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    resp = _upload(client, "line.csv", ALL_DATES_CSV)
    assert resp.status_code == 201
    dataset_id = resp.json()["id"]
    build = client.post(
        "/api/charts/build",
        json={
            "dataset_id": dataset_id,
            "chart_type": "line",
            "x_axis": "order_date",
            "y_axis": "amount",
        },
    )
    assert build.status_code == 200
    chart = build.json()["chart"]
    assert chart["type"] == "line"
    assert chart["columns"] == ["order_date", "amount"]
