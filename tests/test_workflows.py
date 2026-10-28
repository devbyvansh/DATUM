from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from backend.app.ai_provider import ProviderClient
from backend.app.config import Settings
from backend.app.feature_registry import assess_feature_options
from backend.app.workflows.graphs import DatumGraphs


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "db.sqlite",
        upload_dir=tmp_path / "uploads",
        prepared_dir=tmp_path / "prepared",
        report_dir=tmp_path / "reports",
        llm_provider="none",
    )


def test_analyze_only_graph_never_prepares(tmp_path):
    settings = make_settings(tmp_path)
    settings.upload_dir.mkdir()
    source = settings.upload_dir / "source.csv"
    pd.DataFrame({"age": [1, 2, 3], "group": ["a", "b", "a"]}).to_csv(source, index=False)
    graphs = DatumGraphs(settings)
    state = graphs.analyze_only.invoke(
        {
            "run_id": "analyze",
            "dataset_id": "d1",
            "source_path": str(source),
            "mode": "analyze_only",
        },
        config=graphs.config("analyze"),
    )
    assert state["stage"] == "final"
    assert "transformations" not in state["report"]
    assert "transformations" not in state["stage_status"]


def test_prepare_graph_interrupts_and_resumes(tmp_path):
    settings = make_settings(tmp_path)
    settings.upload_dir.mkdir()
    settings.prepared_dir.mkdir()
    source = settings.upload_dir / "source.csv"
    pd.DataFrame({"age": [1, None, 3]}).to_csv(source, index=False)
    graphs = DatumGraphs(settings)
    initial = graphs.prepare_for_ml.invoke(
        {
            "run_id": "prepare",
            "dataset_id": "d1",
            "source_path": str(source),
            "mode": "prepare_for_ml",
            "preparation_config": {"impute_median": ["age"]},
        },
        config=graphs.config("prepare"),
    )
    assert "__interrupt__" in initial
    resumed = graphs.prepare_for_ml.invoke(
        __import__("langgraph.types", fromlist=["Command"]).Command(resume="approved"),
        config=graphs.config("prepare"),
    )
    assert resumed["stage"] == "final"
    assert resumed["transformations"][0]["operation"] == "impute_median"
    assert pd.read_csv(resumed["working_path"])["age"].isna().sum() == 0


def test_provider_without_credentials_is_explicit():
    class Output(BaseModel):
        value: int

    result = ProviderClient(Settings(llm_provider="none")).invoke({}, Output)
    assert result.status == "unavailable"


def test_feature_registry_records_safe_and_deferred_decisions():
    decisions = assess_feature_options(
        pd.DataFrame(
            {
                "amount": [1.0, None, 3.0],
                "group": ["a", "b", "a"],
                "customer_id": ["x", "y", "z"],
            }
        )
    )
    by_technique = {item["technique"]: item for item in decisions}
    assert by_technique["impute_median"]["status"] == "selected"
    assert by_technique["one_hot_encode"]["status"] == "selected"
    assert by_technique["target_encoding"]["status"] == "requires_target"
    assert by_technique["resampling"]["status"] == "requires_user_input"
    assert by_technique["categorical_encoding"]["status"] == "skipped"
