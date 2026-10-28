import io
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.analysis_chat import answer_question


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


def _wait_for_completion(client, run_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in {"completed", "failed", "rejected"}:
            return run
        time.sleep(0.1)
    return client.get(f"/api/runs/{run_id}").json()


# ---------------------------------------------------------------------------
# Fixture dataset A: Sales data WITH date column, numeric + categorical
# ---------------------------------------------------------------------------
SALES_CSV = (
    b"date,product,revenue,quantity,region\n"
    b"2024-01-10,Widget,120.50,5,North\n"
    b"2024-01-15,Gadget,85.00,3,South\n"
    b"2024-02-05,Widget,200.00,10,North\n"
    b"2024-02-20,Gadget,95.50,4,East\n"
    b"2024-03-01,Widget,150.75,7,West\n"
)


# ---------------------------------------------------------------------------
# Fixture dataset B: Product inventory with NO date column, mixed types
# ---------------------------------------------------------------------------
PRODUCT_CSV = (
    b"product_id,product_name,category,stock,price,supplier\n"
    b"101,Widget A,Electronics,50,29.99,Acme Corp\n"
    b"102,Gadget B,Electronics,30,45.50,Globex\n"
    b"103,Widget C,Home,100,12.00,Acme Corp\n"
    b"104,Gadget D,Home,25,35.75,Initech\n"
    b"105,Widget E,Office,75,18.25,Globex\n"
)


# ---------------------------------------------------------------------------
# Fixture dataset C: Employee data WITH missing values
# ---------------------------------------------------------------------------
EMPLOYEE_CSV = (
    b"employee_id,name,department,salary,hire_date,rating\n"
    b"1,Alice Smith,Engineering,95000,2023-01-15,4.5\n"
    b"2,Bob Jones,,85000,2022-06-10,3.8\n"
    b"3,Carol Lee,Engineering,,2023-03-01,4.2\n"
    b"4,David Brown,Marketing,78000,2022-09-05,4.0\n"
    b"5,Eve Davis,Engineering,102000,2021-11-20,\n"
)


# ---------------------------------------------------------------------------
# Shared helper: upload dataset + run analyze_only workflow
# ---------------------------------------------------------------------------
def _upload_and_analyze(client, filename, content):
    upload = client.post(
        "/api/datasets",
        files={"file": (filename, io.BytesIO(content), "text/csv")},
    )
    assert upload.status_code == 201, f"Upload failed: {upload.text}"
    dataset_id = upload.json()["id"]

    preview = client.get(f"/api/datasets/{dataset_id}/preview")
    assert preview.status_code == 200
    return dataset_id, upload.json(), preview.json()


# ===========================================================================
# Test: Dataset A — Sales (has date column)
# ===========================================================================
def test_dataset_a_sales_schema_preview_and_isolation(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    dataset_id, meta, preview = _upload_and_analyze(client, "sales.csv", SALES_CSV)

    # Schema detection infers types from actual columns, not hardcoded
    schema = client.get(f"/api/datasets/{dataset_id}/schema")
    assert schema.status_code == 200
    kinds = {col["name"]: col["kind"] for col in schema.json()["schema"]}
    assert kinds["date"] == "datetime"
    assert kinds["product"] == "categorical"
    assert kinds["revenue"] == "numeric"
    assert kinds["quantity"] == "numeric"
    assert kinds["region"] == "categorical"
    assert meta["metadata"]["row_count"] == 5
    assert meta["metadata"]["column_count"] == 5

    # Preview shows actual rows from this dataset only
    assert preview["total_rows"] == 5
    assert preview["rows"][0]["date"] == "2024-01-10"
    assert preview["rows"][0]["product"] == "Widget"


def test_dataset_a_sales_ask_uses_actual_data(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    dataset_id, _, _ = _upload_and_analyze(client, "sales.csv", SALES_CSV)

    workflow = client.post(
        f"/api/datasets/{dataset_id}/workflows",
        json={"mode": "analyze_only"},
    )
    run_id = workflow.json()["run_id"]
    run = _wait_for_completion(client, run_id)
    assert run["status"] == "completed"

    # Ask row count — must reflect this dataset's 5 rows, not any other
    result = client.post(
        f"/api/runs/{run_id}/analysis/chat",
        json={"question": "How many rows and columns"},
    )
    assert result.status_code == 200
    body = result.json()
    assert body["provider"]["status"] == "not_required"
    assert "5 rows" in body["answer"] and "5 columns" in body["answer"]

    # Ask about column names — must reflect actual columns
    result2 = client.post(
        f"/api/runs/{run_id}/analysis/chat",
        json={"question": "what are the columns"},
    )
    assert result2.status_code == 200
    body2 = result2.json()
    assert "date" in body2["evidence"][0]["value"] and "revenue" in body2["evidence"][0]["value"]


def test_dataset_a_sales_chart_build_uses_actual_columns(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    dataset_id, _, _ = _upload_and_analyze(client, "sales.csv", SALES_CSV)

    response = client.post(
        "/api/charts/build",
        json={
            "dataset_id": dataset_id,
            "chart_type": "line",
            "x_axis": "date",
            "y_axis": "revenue",
        },
    )
    assert response.status_code == 200
    chart = response.json()["chart"]
    assert chart["type"] == "line"
    assert chart["columns"] == ["date", "revenue"]
    assert len(chart["data"]) >= 5
    # Data values must come from this dataset, not hardcoded
    data_map = {row["x"]: row["y"] for row in chart["data"]}
    assert data_map.get("2024-01-10") == 120.5
    assert data_map.get("2024-02-05") == 200.0


def test_dataset_a_sales_prepare_for_ml(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    dataset_id, _, _ = _upload_and_analyze(client, "sales.csv", SALES_CSV)

    workflow = client.post(
        f"/api/datasets/{dataset_id}/workflows",
        json={
            "mode": "prepare_for_ml",
            "preparation": {"impute_median": ["revenue"]},
        },
    )
    run_id = workflow.json()["run_id"]
    # prepare_for_ml interrupts for approval
    pending = client.get(f"/api/runs/{run_id}").json()
    assert pending["status"] == "needs_approval"
    approved = client.post(f"/api/runs/{run_id}/approval", json={"decision": "approved"})
    assert approved.status_code == 200
    run = _wait_for_completion(client, run_id)
    assert run["status"] == "completed"

    prepared = client.get(f"/api/runs/{run_id}/prepared-preview")
    assert prepared.status_code == 200
    body = prepared.json()
    assert body["total_rows"] == 5
    assert body["total_columns"] == 5


# ===========================================================================
# Test: Dataset B — Product inventory (NO date column)
# ===========================================================================
def test_dataset_b_product_no_date_column(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    dataset_id, meta, preview = _upload_and_analyze(client, "products.csv", PRODUCT_CSV)

    # Schema: no date column present; all types inferred from data
    schema = client.get(f"/api/datasets/{dataset_id}/schema")
    kinds = {col["name"]: col["kind"] for col in schema.json()["schema"]}
    assert "date" not in kinds
    assert kinds["product_id"] == "numeric"
    assert kinds["product_name"] == "categorical"
    assert kinds["category"] == "categorical"
    assert kinds["stock"] == "numeric"
    assert kinds["price"] == "numeric"
    assert kinds["supplier"] == "categorical"
    assert meta["metadata"]["row_count"] == 5
    assert preview["total_rows"] == 5


def test_dataset_b_product_ask_no_date_column(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    dataset_id, _, _ = _upload_and_analyze(client, "products.csv", PRODUCT_CSV)

    workflow = client.post(
        f"/api/datasets/{dataset_id}/workflows",
        json={"mode": "analyze_only"},
    )
    run_id = workflow.json()["run_id"]
    run = _wait_for_completion(client, run_id)
    assert run["status"] == "completed"

    # Ask about shape — must reflect this dataset
    result = client.post(
        f"/api/runs/{run_id}/analysis/chat",
        json={"question": "How many rows and columns"},
    )
    assert result.status_code == 200
    body = result.json()
    assert "5 rows" in body["answer"] and "6 columns" in body["answer"]

    # Ask about a specific column summary using actual column names
    result2 = client.post(
        f"/api/runs/{run_id}/analysis/chat",
        json={"question": "What is the price column summary?"},
    )
    assert result2.status_code == 200
    body2 = result2.json()
    assert body2["evidence"][0]["value"] == "price"


def test_dataset_b_product_chart_build_no_date(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    dataset_id, _, _ = _upload_and_analyze(client, "products.csv", PRODUCT_CSV)

    # Bar chart: category X, price Y — no date column needed
    response = client.post(
        "/api/charts/build",
        json={
            "dataset_id": dataset_id,
            "chart_type": "bar",
            "x_axis": "category",
            "y_axis": "price",
            "aggregation": "mean",
        },
    )
    assert response.status_code == 200
    chart = response.json()["chart"]
    assert chart["type"] == "bar"
    assert chart["columns"] == ["category", "price"]

    # Histogram on stock (single numeric field)
    response2 = client.post(
        "/api/charts/build",
        json={
            "dataset_id": dataset_id,
            "chart_type": "histogram",
            "x_axis": "stock",
        },
    )
    assert response2.status_code == 200
    chart2 = response2.json()["chart"]
    assert chart2["type"] == "histogram"
    assert chart2["columns"] == ["stock"]


def test_dataset_b_product_prepare_for_ml(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    dataset_id, _, _ = _upload_and_analyze(client, "products.csv", PRODUCT_CSV)

    workflow = client.post(
        f"/api/datasets/{dataset_id}/workflows",
        json={
            "mode": "prepare_for_ml",
            "preparation": {"impute_mode": ["category"]},
        },
    )
    run_id = workflow.json()["run_id"]
    pending = client.get(f"/api/runs/{run_id}").json()
    assert pending["status"] == "needs_approval"
    approved = client.post(f"/api/runs/{run_id}/approval", json={"decision": "approved"})
    assert approved.status_code == 200
    run = _wait_for_completion(client, run_id)
    assert run["status"] == "completed"

    prepared = client.get(f"/api/runs/{run_id}/prepared-preview")
    assert prepared.status_code == 200
    body = prepared.json()
    assert body["total_rows"] == 5
    assert body["total_columns"] == 6


# ===========================================================================
# Test: Dataset C — Employee data WITH missing values
# ===========================================================================
def test_dataset_c_employee_missing_values(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    dataset_id, meta, preview = _upload_and_analyze(client, "employees.csv", EMPLOYEE_CSV)

    schema = client.get(f"/api/datasets/{dataset_id}/schema")
    kinds = {col["name"]: col["kind"] for col in schema.json()["schema"]}
    assert kinds["employee_id"] == "numeric"
    assert kinds["name"] == "categorical"
    assert kinds["department"] == "categorical"
    assert kinds["salary"] == "numeric"
    assert kinds["hire_date"] == "datetime"
    assert kinds["rating"] == "numeric"
    assert meta["metadata"]["row_count"] == 5
    assert preview["total_rows"] == 5

    # Preview should show actual data including blanks (null in JSON)
    rows = preview["rows"]
    assert any(r.get("department") in (None, "") for r in rows)  # Bob's missing department
    assert any(r.get("salary") in (None, "") for r in rows)  # Carol's missing salary
    assert any(r.get("rating") in (None, "") for r in rows)  # Eve's missing rating


def test_dataset_c_employee_ask_missing_values(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    dataset_id, _, _ = _upload_and_analyze(client, "employees.csv", EMPLOYEE_CSV)

    workflow = client.post(
        f"/api/datasets/{dataset_id}/workflows",
        json={"mode": "analyze_only"},
    )
    run_id = workflow.json()["run_id"]
    run = _wait_for_completion(client, run_id)
    assert run["status"] == "completed"

    # Ask about missing values — actual count from this dataset
    result = client.post(
        f"/api/runs/{run_id}/analysis/chat",
        json={"question": "Which values are missing"},
    )
    assert result.status_code == 200
    body = result.json()
    assert body["provider"]["status"] == "not_required"

    # Ask about average salary — must compute from actual non-null values
    result2 = client.post(
        f"/api/runs/{run_id}/analysis/chat",
        json={"question": "average of salary"},
    )
    assert result2.status_code == 200
    body2 = result2.json()
    # Salaries: 95000, 85000, (missing), 78000, 102000 -> avg of 4 = 90000
    assert "90000" in body2["answer"] or "90,000" in body2["answer"]


def test_dataset_c_employee_prepare_impute_missing(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    dataset_id, _, _ = _upload_and_analyze(client, "employees.csv", EMPLOYEE_CSV)

    workflow = client.post(
        f"/api/datasets/{dataset_id}/workflows",
        json={
            "mode": "prepare_for_ml",
            "preparation": {"impute_median": ["salary", "rating"]},
        },
    )
    run_id = workflow.json()["run_id"]
    pending = client.get(f"/api/runs/{run_id}").json()
    assert pending["status"] == "needs_approval"
    approved = client.post(f"/api/runs/{run_id}/approval", json={"decision": "approved"})
    assert approved.status_code == 200
    run = _wait_for_completion(client, run_id)
    assert run["status"] == "completed"

    prepared = client.get(f"/api/runs/{run_id}/prepared-preview")
    assert prepared.status_code == 200
    body = prepared.json()
    assert body["total_rows"] == 5
    # After imputation, missing cells should not increase for imputed columns
    orig_missing = body["comparison"]["original"]["missing_cells"]
    prep_missing = body["comparison"]["prepared"]["missing_cells"]
    assert prep_missing <= orig_missing


# ===========================================================================
# Test: Dataset isolation — different datasets never mix
# ===========================================================================
def test_dataset_isolation_across_uploads(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)

    # Upload sales dataset
    sales_upload = client.post(
        "/api/datasets",
        files={"file": ("sales.csv", io.BytesIO(SALES_CSV), "text/csv")},
    )
    sales_id = sales_upload.json()["id"]

    # Upload product dataset (different structure)
    product_upload = client.post(
        "/api/datasets",
        files={"file": ("products.csv", io.BytesIO(PRODUCT_CSV), "text/csv")},
    )
    product_id = product_upload.json()["id"]
    assert sales_id != product_id

    # Each dataset lists its own correct metadata
    sales_meta = client.get(f"/api/datasets/{sales_id}").json()
    product_meta = client.get(f"/api/datasets/{product_id}").json()
    assert sales_meta["metadata"]["column_count"] == 5
    assert product_meta["metadata"]["column_count"] == 6
    assert sales_meta["filename"] == "sales.csv"
    assert product_meta["filename"] == "products.csv"

    # Preview for each dataset shows only its own rows
    sales_preview = client.get(f"/api/datasets/{sales_id}/preview").json()
    product_preview = client.get(f"/api/datasets/{product_id}/preview").json()
    assert sales_preview["total_rows"] == 5
    assert product_preview["total_rows"] == 5
    assert "date" in sales_preview["columns"]
    assert "date" not in product_preview["columns"]
    assert "product_id" in product_preview["columns"]
    assert "product_id" not in sales_preview["columns"]

    # List shows both datasets, correctly identified
    listed = client.get("/api/datasets").json()
    ids = {d["id"] for d in listed["datasets"]}
    assert sales_id in ids and product_id in ids


# ===========================================================================
# Test: Unsupported format handling
# ===========================================================================
def test_unsupported_format_clear_error(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    response = client.post(
        "/api/datasets",
        files={"file": ("image.png", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "unsupported_format"
    assert "Unsupported dataset format" in detail["message"]


# ===========================================================================
# Test: Analysis chat with direct frame (unit-level) for each fixture type
# ===========================================================================
def test_answer_question_dataset_a_sales(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame(
        {
            "date": ["2024-01-10", "2024-02-05", "2024-03-01"],
            "revenue": [120.5, 200.0, 150.75],
            "region": ["North", "East", "West"],
        }
    )
    result = answer_question("run-a", "ds-a", frame, "Show monthly revenue")
    assert result["chart"]["type"] == "line"
    assert result["evidence"][1]["value"] == 200.0  # Feb total = 200


def test_answer_question_dataset_b_product_no_date(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame(
        {
            "product_id": ["P-101", "P-102", "P-103"],
            "category": ["Electronics", "Electronics", "Home"],
            "price": ["29.99", "45.50", "12.00"],
        }
    )
    # No date column — summary question should work on actual data
    result = answer_question("run-b", "ds-b", frame, "What is the category column summary?")
    assert result["evidence"][0]["value"] == "category"
    assert result["evidence"][1]["value"] == "object"

    # Trend question on no-date dataset should fail with DatumError
    from backend.app.errors import DatumError
    with pytest.raises(DatumError, match="No suitable date or time column"):
        answer_question("run-b", "ds-b", frame, "Show monthly price trend")


def test_answer_question_dataset_c_employee_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "salary": [95000, 85000, None],
            "department": ["Engineering", None, "Engineering"],
        }
    )
    result = answer_question("run-c", "ds-c", frame, "average of salary")
    # Average of 95000 and 85000 = 90000 (missing excluded)
    assert "90000" in result["answer"] or "90,000" in result["answer"]
