from pathlib import Path

import httpx
from pydantic import BaseModel

from backend.app.ai_provider import ProviderClient
from backend.app.config import Settings


class Output(BaseModel):
    value: int


def test_local_provider_does_not_require_api_key(monkeypatch):
    settings = Settings(
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:11434/v1",
        llm_model="local-model",
        llm_api_key="",
        llm_max_retries=0,
    )
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"value": 7}'}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = ProviderClient(settings).invoke({"evidence": 7}, Output)
    assert result.status == "completed"
    assert result.output == {"value": 7}
    assert "Authorization" not in calls[0][1]["headers"]


def test_provider_rejects_permanent_auth_error(monkeypatch):
    settings = Settings(
        llm_provider="openai_compatible",
        llm_base_url="http://provider/v1",
        llm_model="model",
        llm_api_key="secret",
        llm_max_retries=3,
    )
    calls = 0

    def fake_post(url, **kwargs):
        nonlocal calls
        calls += 1
        return httpx.Response(401, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    result = ProviderClient(settings).invoke({}, Output)
    assert result.status == "failed"
    assert calls == 1
    assert "secret" not in (result.error or "")


def test_checkpoint_survives_graph_reinitialization(tmp_path: Path):
    import pandas as pd
    from langgraph.types import Command

    from backend.app.workflows.graphs import DatumGraphs

    settings = Settings(
        database_path=tmp_path / "db.sqlite",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite",
        upload_dir=tmp_path / "uploads",
        prepared_dir=tmp_path / "prepared",
        report_dir=tmp_path / "reports",
    )
    settings.upload_dir.mkdir()
    settings.prepared_dir.mkdir()
    source = settings.upload_dir / "data.csv"
    pd.DataFrame({"age": [1, None, 3]}).to_csv(source, index=False)
    first = DatumGraphs(settings)
    paused = first.prepare_for_ml.invoke(
        {
            "run_id": "persistent-run",
            "dataset_id": "dataset",
            "source_path": str(source),
            "mode": "prepare_for_ml",
            "preparation_config": {"impute_median": ["age"]},
        },
        config=first.config("persistent-run"),
    )
    assert "__interrupt__" in paused
    first.close()
    second = DatumGraphs(settings)
    completed = second.prepare_for_ml.invoke(
        Command(resume="approved"), config=second.config("persistent-run")
    )
    second.close()
    assert completed["stage"] == "final"
    assert completed["approval"] == "approved"
