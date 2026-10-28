import pandas as pd
import pytest

from backend.app.data_processing import (
    load_dataframe,
    prepare_dataframe,
    profile_dataframe,
    quality_report,
    statistics,
)
from backend.app.errors import DatumError


def test_profile_and_quality_report():
    frame = pd.DataFrame({"age": [10, None, 10], "group": ["a", "b", "b"], "constant": [1, 1, 1]})
    profile = profile_dataframe(frame)
    assert profile["row_count"] == 3
    assert (
        next(column for column in profile["columns"] if column["name"] == "age")["missing_count"]
        == 1
    )
    report = quality_report(frame)
    assert report["duplicate_row_count"] == 0
    assert {finding["check_id"] for finding in report["findings"]} >= {
        "missing_values",
        "constant_column",
    }


def test_duplicate_and_median_imputation():
    source = pd.DataFrame({"value": [1.0, None, 1.0], "label": ["a", "b", "a"]})
    prepared, records = prepare_dataframe(
        source, {"drop_duplicates": True, "impute_median": ["value"]}
    )
    assert source.isna().sum()["value"] == 1
    assert prepared["value"].isna().sum() == 0
    assert len(prepared) == 2
    assert [record["operation"] for record in records] == ["drop_duplicates", "impute_median"]


def test_statistics_handles_no_numeric_columns():
    result = statistics(pd.DataFrame({"name": ["a", "b"]}))
    assert result["descriptive_statistics"] == {}
    assert result["warnings"]


def test_invalid_file_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("a,b\n1,2", encoding="utf-8")
    with pytest.raises(DatumError, match="Unsupported"):
        load_dataframe(path)
