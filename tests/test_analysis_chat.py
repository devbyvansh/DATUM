import pandas as pd
import pytest

from backend.app.analysis_chat import answer_question
from backend.app.errors import DatumError


def test_monthly_trend_uses_actual_values_and_returns_chart(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame(
        {
            "date": ["2024-02-01", "2024-01-01", "2024-02-15"],
            "revenue": [20, 10, 5],
            "region": ["North", "North", "South"],
        }
    )
    result = answer_question("run-1", "dataset-1", frame, "Show monthly revenue")
    assert result["evidence"][0]["value"] == "2024-02"
    assert result["evidence"][1]["value"] == 25.0
    assert result["chart"]["type"] == "line"
    assert result["chart"]["data"][-1]["y"] == 25.0
    assert result["chart"]["dataset_context"]["aggregation"] == "sum by period"
    assert result["context"]["metric"] == "revenue"


def test_unspecified_trend_uses_safe_monthly_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame(
        {"date": ["2024-01-01", "2024-02-01"], "revenue": [10, 20]}
    )
    result = answer_question("run-1", "dataset-1", frame, "tell me about trends")
    assert result["chart"]["type"] == "line"
    assert result["context"]["granularity"] == "M"


def test_follow_up_filter_and_previous_period():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"],
            "sales": [10, 20, 100, 50],
            "region": ["North", "North", "South", "South"],
        }
    )
    first = answer_question("run-2", "dataset-2", frame, "Show monthly sales")
    follow_up = answer_question(
        "run-2",
        "dataset-2",
        frame,
        "Compare it with the previous month for North",
        first["context"],
    )
    assert follow_up["context"]["filters"] == {"region": "North"}
    assert follow_up["evidence"][0]["value"] == 20.0
    assert follow_up["evidence"][1]["value"] == 10.0
    assert follow_up["evidence"][2]["value"] == 100.0


def test_context_cannot_cross_runs():
    with pytest.raises(DatumError, match="different run"):
        answer_question(
            "run-2",
            "dataset-2",
            pd.DataFrame({"value": [1, 2]}),
            "What are the rows?",
            {"run_id": "run-1", "dataset_id": "dataset-2"},
        )


def test_column_summary_and_missing_column_error():
    frame = pd.DataFrame({"region": ["North", "South"], "sales": [10, 20]})
    summary = answer_question("run-3", "dataset-3", frame, "What is the region column summary?")
    assert summary["evidence"][0]["value"] == "region"
    with pytest.raises(DatumError, match="missing_column"):
        answer_question(
            "run-3",
            "dataset-3",
            frame,
            "What is the column missing_column summary?",
        )
