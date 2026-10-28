from io import BytesIO

from fastapi.testclient import TestClient

from backend.app.main import app


def test_health():
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_and_preview(tmp_path, monkeypatch):
    from backend.app import main

    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(main.settings, "prepared_dir", tmp_path / "prepared")
    monkeypatch.setattr(main.settings, "report_dir", tmp_path / "reports")
    for directory in (
        main.settings.upload_dir,
        main.settings.prepared_dir,
        main.settings.report_dir,
    ):
        directory.mkdir()
    response = TestClient(app).post(
        "/api/datasets", files={"file": ("sample.csv", BytesIO(b"x,y\n1,a\n2,b\n"), "text/csv")}
    )
    assert response.status_code == 201
    dataset_id = response.json()["id"]
    preview = TestClient(app).get(f"/api/datasets/{dataset_id}/preview")
    assert preview.status_code == 200
    assert len(preview.json()["rows"]) == 2
    datasets = TestClient(app).get("/api/datasets")
    assert datasets.status_code == 200
    assert datasets.json()["datasets"][0]["id"] == dataset_id


def test_prepare_workflow_approval_and_resume(tmp_path, monkeypatch):
    from backend.app import main

    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(main.settings, "prepared_dir", tmp_path / "prepared")
    monkeypatch.setattr(main.settings, "report_dir", tmp_path / "reports")
    for directory in (
        main.settings.upload_dir,
        main.settings.prepared_dir,
        main.settings.report_dir,
    ):
        directory.mkdir()
    client = TestClient(app)
    upload = client.post(
        "/api/datasets",
        files={"file": ("approval.csv", BytesIO(b"age\n1\n\n3\n"), "text/csv")},
    )
    dataset_id = upload.json()["id"]
    workflow = client.post(
        f"/api/datasets/{dataset_id}/workflows",
        json={"mode": "prepare_for_ml", "preparation": {"impute_median": ["age"]}},
    )
    run_id = workflow.json()["run_id"]
    pending = client.get(f"/api/runs/{run_id}").json()
    assert pending["status"] == "needs_approval"
    assert client.get(f"/api/runs/{run_id}/preparation-plan").json()["steps"]
    approved = client.post(f"/api/runs/{run_id}/approval", json={"decision": "approved"})
    assert approved.status_code == 200
    completed = client.get(f"/api/runs/{run_id}").json()
    assert completed["status"] == "completed"
    assert completed["result"]["transformations"][0]["operation"] == "impute_median"
    prepared = client.get(f"/api/runs/{run_id}/prepared-preview")
    assert prepared.status_code == 200
    prepared_body = prepared.json()
    assert prepared_body["total_rows"] == 3
    assert prepared_body["total_columns"] == 1
    assert prepared_body["comparison"]["original"]["missing_cells"] == 1
    assert prepared_body["comparison"]["prepared"]["missing_cells"] == 0
    assert prepared_body["rows"][1]["age"] == 2
    assert client.get(f"/api/runs/{run_id}/prepared-preview?offset=-1").status_code == 422


def test_analysis_chat_is_run_scoped_and_deterministic(tmp_path, monkeypatch):
    from backend.app import main

    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(main.settings, "prepared_dir", tmp_path / "prepared")
    monkeypatch.setattr(main.settings, "report_dir", tmp_path / "reports")
    for directory in (
        main.settings.upload_dir,
        main.settings.prepared_dir,
        main.settings.report_dir,
    ):
        directory.mkdir()
    client = TestClient(app)
    upload = client.post(
        "/api/datasets",
        files={
            "file": (
                "analysis.csv",
                BytesIO(b"date,revenue,region\n2024-01-01,10,North\n2024-02-01,25,North\n"),
                "text/csv",
            )
        },
    )
    dataset_id = upload.json()["id"]
    workflow = client.post(
        f"/api/datasets/{dataset_id}/workflows",
        json={"mode": "analyze_only"},
    )
    run_id = workflow.json()["run_id"]
    result = client.post(
        f"/api/runs/{run_id}/analysis/chat",
        json={"question": "Show monthly revenue"},
    )
    assert result.status_code == 200
    body = result.json()
    assert body["provider"]["status"] == "not_required"
    assert body["chart"]["type"] == "line"
    assert body["evidence"][1]["value"] == 25.0
    mismatched = client.post(
        f"/api/runs/{run_id}/analysis/chat",
        json={
            "question": "What are the rows?",
            "context": {"run_id": "different-run", "dataset_id": dataset_id},
        },
    )
    assert mismatched.status_code == 400


def test_analysis_charts_include_dataset_context(tmp_path, monkeypatch):
    from backend.app import main

    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(main.settings, "prepared_dir", tmp_path / "prepared")
    monkeypatch.setattr(main.settings, "report_dir", tmp_path / "reports")
    for directory in (
        main.settings.upload_dir,
        main.settings.prepared_dir,
        main.settings.report_dir,
    ):
        directory.mkdir()
    client = TestClient(app)
    upload = client.post(
        "/api/datasets",
        files={
            "file": (
                "context.csv",
                BytesIO(b"date,revenue,region\n2024-01-01,10,North\n2024-02-01,25,South\n"),
                "text/csv",
            )
        },
    )
    dataset_id = upload.json()["id"]
    workflow = client.post(
        f"/api/datasets/{dataset_id}/workflows",
        json={"mode": "analyze_only"},
    )
    run_id = workflow.json()["run_id"]
    run = client.get(f"/api/runs/{run_id}").json()
    for _ in range(20):
        if run["status"] == "completed":
            break
        import time

        time.sleep(0.1)
        run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "completed"
    chart = run["result"]["statistics"]["charts"][0]
    context = chart["dataset_context"]
    assert context["dataset_name"] == "context.csv"
    assert context["rows"] == 2
    assert context["source_columns"] == chart["columns"]


def test_schema_detects_column_kinds(tmp_path, monkeypatch):
    from backend.app import main

    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(main.settings, "prepared_dir", tmp_path / "prepared")
    monkeypatch.setattr(main.settings, "report_dir", tmp_path / "reports")
    for directory in (
        main.settings.upload_dir,
        main.settings.prepared_dir,
        main.settings.report_dir,
    ):
        directory.mkdir()

    client = TestClient(app)
    upload = client.post(
        "/api/datasets",
        files={
            "file": (
                "types.csv",
                BytesIO(
                    b"order_id,order_date,quantity,price,region\n"
                    b"1001,2024-01-15,5,29.99,North\n"
                    b"1002,2024-02-20,3,19.50,South\n"
                    b"1003,2024-03-10,8,45.00,North\n"
                ),
                "text/csv",
            )
        },
    )
    dataset_id = upload.json()["id"]

    schema = client.get(f"/api/datasets/{dataset_id}/schema")
    assert schema.status_code == 200
    kinds = {col["name"]: col["kind"] for col in schema.json()["schema"]}
    assert kinds["order_id"] == "numeric", "order_id should be numeric"
    assert kinds["order_date"] == "datetime", "order_date should be datetime"
    assert kinds["quantity"] == "numeric", "quantity should be numeric"
    assert kinds["price"] == "numeric", "price should be numeric"
    assert kinds["region"] == "categorical", "region should be categorical"


def test_preview_pagination_and_search(tmp_path, monkeypatch):
    from backend.app import main

    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(main.settings, "prepared_dir", tmp_path / "prepared")
    monkeypatch.setattr(main.settings, "report_dir", tmp_path / "reports")
    for directory in (
        main.settings.upload_dir,
        main.settings.prepared_dir,
        main.settings.report_dir,
    ):
        directory.mkdir()

    client = TestClient(app)
    upload = client.post(
        "/api/datasets",
        files={
            "file": (
                "pagination.csv",
                BytesIO(
                    b"x,y\n1,a\n2,b\n3,c\n4,d\n5,e\n6,f\n7,g\n8,h\n9,i\n10,j\n"
                ),
                "text/csv",
            )
        },
    )
    dataset_id = upload.json()["id"]

    # Page 1 with limit 3
    p1 = client.get(f"/api/datasets/{dataset_id}/preview?offset=0&limit=3")
    assert p1.status_code == 200
    body = p1.json()
    assert body["total_rows"] == 10
    assert body["returned_rows"] == 3
    assert body["offset"] == 0
    assert body["limit"] == 3
    assert len(body["rows"]) == 3

    # Page 2
    p2 = client.get(f"/api/datasets/{dataset_id}/preview?offset=3&limit=3")
    assert p2.status_code == 200
    assert len(p2.json()["rows"]) == 3

    # Search
    ps = client.get(f"/api/datasets/{dataset_id}/preview?search=b")
    assert ps.status_code == 200
    search_body = ps.json()
    assert search_body["total_rows"] == 1  # Only row "2,b" contains "b"
    assert search_body["returned_rows"] == 1


def test_values_endpoint_for_filters(tmp_path, monkeypatch):
    from backend.app import main

    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(main.settings, "prepared_dir", tmp_path / "prepared")
    monkeypatch.setattr(main.settings, "report_dir", tmp_path / "reports")
    for directory in (
        main.settings.upload_dir,
        main.settings.prepared_dir,
        main.settings.report_dir,
    ):
        directory.mkdir()

    client = TestClient(app)
    upload = client.post(
        "/api/datasets",
        files={
            "file": (
                "values.csv",
                BytesIO(b"region,quantity\nNorth,5\nSouth,3\nNorth,8\nEast,2\nWest,6\n"),
                "text/csv",
            )
        },
    )
    dataset_id = upload.json()["id"]

    values = client.get(f"/api/datasets/{dataset_id}/values?column=region")
    assert values.status_code == 200
    body = values.json()
    assert body["column"] == "region"
    assert "North" in body["values"]
    assert "South" in body["values"]
    assert "East" in body["values"]
    assert "West" in body["values"]
    assert body["total_unique"] == 4

    # Numeric column values
    values_num = client.get(f"/api/datasets/{dataset_id}/values?column=quantity")
    assert values_num.status_code == 200
    assert values_num.json()["total_unique"] == 5


def test_chart_build_line_chart(tmp_path, monkeypatch):
    from backend.app import main

    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(main.settings, "prepared_dir", tmp_path / "prepared")
    monkeypatch.setattr(main.settings, "report_dir", tmp_path / "reports")
    for directory in (
        main.settings.upload_dir,
        main.settings.prepared_dir,
        main.settings.report_dir,
    ):
        directory.mkdir()

    client = TestClient(app)
    upload = client.post(
        "/api/datasets",
        files={
            "file": (
                "line.csv",
                BytesIO(
                    b"order_date,quantity\n"
                    b"2024-03-10,8\n"
                    b"2024-01-15,5\n"
                    b"2024-02-20,3\n"
                    b"2024-04-05,2\n"
                    b"2024-05-12,6\n"
                ),
                "text/csv",
            )
        },
    )
    dataset_id = upload.json()["id"]

    # Build a line chart with datetime X and numeric Y
    response = client.post(
        "/api/charts/build",
        json={
            "dataset_id": dataset_id,
            "chart_type": "line",
            "x_axis": "order_date",
            "y_axis": "quantity",
            "aggregation": None,
        },
    )
    assert response.status_code == 200
    chart = response.json()["chart"]
    assert chart["type"] == "line"
    assert chart["columns"] == ["order_date", "quantity"]
    assert len(chart["data"]) >= 5  # Line chart fills in missing dates
    # Verify data is sorted chronologically
    dates = [row["x"] for row in chart["data"]]
    assert dates == sorted(dates), f"Line chart data should be sorted by date, got {dates}"
    # Verify our actual data points are present
    data_map = {row["x"]: row["y"] for row in chart["data"]}
    assert data_map.get("2024-01-15") == 5.0
    assert data_map.get("2024-02-20") == 3.0
    assert data_map.get("2024-03-10") == 8.0

    # Invalid chart type should return 400
    bad = client.post(
        "/api/charts/build",
        json={"dataset_id": dataset_id, "chart_type": "invalid_type"},
    )
    assert bad.status_code == 400
    assert bad.json()["detail"]["code"] == "invalid_chart_config"

    # Missing required X axis for line chart
    missing_x = client.post(
        "/api/charts/build",
        json={
            "dataset_id": dataset_id,
            "chart_type": "line",
            "y_axis": "quantity",
        },
    )
    assert missing_x.status_code == 400
