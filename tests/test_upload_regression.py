import io

from fastapi.testclient import TestClient

from backend.app import main


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


def test_upload_valid_csv_success_and_workspace_refresh(tmp_path, monkeypatch):
    """Opening the upload interface, selecting a valid CSV, uploading, and
    refreshing the workspace (re-listing datasets) must keep the app usable."""
    client = _make_client(tmp_path, monkeypatch)
    response = client.post(
        "/api/datasets",
        files={"file": ("sample.csv", io.BytesIO(b"x,y\n1,a\n2,b\n"), "text/csv")},
    )
    assert response.status_code == 201
    body = response.json()
    dataset_id = body["id"]
    assert body["filename"] == "sample.csv"
    assert body["size_bytes"] > 0
    assert body["metadata"]["row_count"] == 2

    # Preview is available right after upload (workspace inspect step).
    preview = client.get(f"/api/datasets/{dataset_id}/preview")
    assert preview.status_code == 200
    assert preview.json()["rows"][0] == {"x": 1, "y": "a"}

    # Refreshing the workspace still lists the uploaded dataset.
    listed = client.get("/api/datasets")
    assert listed.status_code == 200
    ids = [dataset["id"] for dataset in listed.json()["datasets"]]
    assert dataset_id in ids


def test_upload_jsonl_supported_format(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    response = client.post(
        "/api/datasets",
        files={"file": ("data.jsonl", io.BytesIO(b'{"a": 1}\n{"a": 2}\n'), "application/jsonl")},
    )
    assert response.status_code == 201
    assert response.json()["metadata"]["row_count"] == 2


def test_upload_unsupported_format_shows_actionable_error(tmp_path, monkeypatch):
    """An unsupported file must fail with a structured error, not a blank page."""
    client = _make_client(tmp_path, monkeypatch)
    response = client.post(
        "/api/datasets",
        files={"file": ("image.png", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "unsupported_format"
    assert "Unsupported dataset format" in detail["message"]


def test_upload_file_too_large_shows_actionable_error(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    oversize = b"0" * (main.settings.max_upload_size_mb * 1024 * 1024 + 1)
    response = client.post(
        "/api/datasets",
        files={"file": ("big.csv", io.BytesIO(oversize), "text/csv")},
    )
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert detail["code"] == "file_too_large"
    assert "size limit" in detail["message"]
