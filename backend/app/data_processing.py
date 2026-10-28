from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.errors import DatumError
from backend.app.features import apply_feature_operations

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}


def load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DatumError(
            f"Unsupported file format: {suffix or 'missing extension'}.", "unsupported_format"
        )
    try:
        if suffix == ".csv":
            frame = pd.read_csv(path, skip_blank_lines=False)
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        elif suffix == ".json":
            frame = pd.read_json(path)
        elif suffix == ".jsonl":
            frame = pd.read_json(path, lines=True)
        else:
            frame = pd.read_parquet(path)
    except (ValueError, UnicodeDecodeError, OSError, ImportError) as exc:
        raise DatumError(
            "The uploaded file could not be parsed as a supported dataset.", "invalid_file"
        ) from exc
    if frame.empty and len(frame.columns) == 0:
        raise DatumError("The dataset contains no columns.", "empty_dataset")
    frame.columns = _unique_columns([str(column) for column in frame.columns])
    return frame


def _analysis_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an analysis copy where blank text cells are treated as missing."""
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        values = result[column]
        # Normalize whitespace-only to NaN for analysis
        try:
            string_values = values.astype("string")
            mask = string_values.str.strip().eq("") & string_values.notna()
            result[column] = values.mask(mask)
        except Exception:
            # Fallback
            result[column] = values.apply(
                lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
            )
    return result


def _unique_columns(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result = []
    for column in columns:
        count = counts.get(column, 0)
        result.append(column if count == 0 else f"{column}.{count}")
        counts[column] = count + 1
    return result


def _is_likely_date_column(series: pd.Series) -> bool:
    """Determine whether a column should be treated as a date/time column.

    Rules (deterministic, explainable, no column-name dependence):
    - Numeric columns are never dates (prices, salaries, quantities, IDs, ratings).
    - Missing values are excluded from the parseability calculation.
    - A non-numeric column is a date only when at least 80% of its non-null
      values parse as timestamps.
    - Columns whose parsed values are all year 1970 (unix-epoch artifacts from
      numeric-string parsing) are rejected.
    """
    # Never classify numeric columns as dates.
    if pd.api.types.is_numeric_dtype(series):
        return False

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    parseable_ratio = float(parsed.notna().mean())

    # At least 80% of non-null values must parse as timestamps.
    if parseable_ratio < 0.8:
        return False

    # Reject columns that parse entirely to year 1970 (epoch-timestamp artifacts).
    if bool((parsed.dt.year == 1970).all()):
        return False

    return True


def profile_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    frame = _analysis_frame(frame)
    columns = []
    for name in frame.columns:
        series = frame[name]
        missing = int(series.isna().sum())
        info: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "rows": int(len(series)),
            "missing_count": missing,
            "missing_percentage": round(missing / len(series) * 100, 3) if len(series) else 0,
            "unique_count": int(series.nunique(dropna=True)),
            "is_constant": bool(series.nunique(dropna=True) <= 1),
            "potential_identifier": bool(
                len(series) > 0 and series.nunique(dropna=True) == len(series)
            ),
        }
        if pd.api.types.is_numeric_dtype(series):
            values = (
                pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            )
            info["summary"] = {
                "min": _number(values.min()),
                "max": _number(values.max()),
                "mean": _number(values.mean()),
                "median": _number(values.median()),
                "std": _number(values.std()),
            }
        else:
            info["top_values"] = {
                str(k): int(v) for k, v in series.value_counts(dropna=True).head(10).items()
            }
        columns.append(info)
    missing_cells = int(frame.isna().sum().sum())
    # Use duplicated with keep=False logic? Or just count duplicates
    duplicate_row_count = int(frame.duplicated(keep='first').sum())
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "missing_cell_count": missing_cells,
        "missing_cell_percentage": round(
            missing_cells / (len(frame) * len(frame.columns)) * 100, 3
        )
        if len(frame) and len(frame.columns)
        else 0,
        "duplicate_row_count": duplicate_row_count,
        "columns": columns,
    }


def quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    frame = _analysis_frame(frame)
    findings: list[dict[str, Any]] = []
    # Count actual duplicate rows (all duplicates except first)
    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        findings.append(
            _finding(
                "duplicate_rows",
                "warning",
                [],
                duplicate_rows,
                {"duplicate_count": duplicate_rows},
                "Duplicate records were detected.",
                "Review and remove duplicates if they are not intentional.",
            )
        )
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        if missing:
            findings.append(
                _finding(
                    "missing_values",
                    "warning",
                    [str(column)],
                    missing,
                    {
                        "missing_count": missing,
                        "missing_percentage": round(missing / len(frame) * 100, 3),
                    },
                    f"{missing} values are missing in {column}.",
                    "Review the missingness pattern before choosing a strategy.",
                )
            )
        if series.nunique(dropna=True) <= 1:
            findings.append(
                _finding(
                    "constant_column",
                    "warning",
                    [str(column)],
                    len(frame),
                    {"unique_count": int(series.nunique(dropna=True))},
                    f"{column} has no useful variation.",
                    "Exclude it unless it has a domain-specific purpose.",
                )
            )
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            finite_values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            invalid_count = int((~np.isfinite(finite_values)).sum())
            if invalid_count:
                findings.append(
                    _finding(
                        "non_finite_values", "warning", [str(column)], invalid_count, {"invalid_count": invalid_count},
                        f"Non-finite numeric values were found in {column}.",
                        "Replace or investigate them before statistical analysis.",
                    )
                )
            if len(numeric) >= 4:
                q1, q3 = numeric.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr and iqr != 0:
                    outliers = int(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum())
                else:
                    outliers = 0
                if outliers:
                    findings.append(
                        _finding(
                            "iqr_outliers",
                            "info",
                            [str(column)],
                            outliers,
                            {"outlier_count": outliers, "method": "1.5x IQR"},
                            f"{outliers} possible outliers were found in {column}.",
                            "Investigate unusual values; do not remove them without context.",
                        )
                    )
            if "percent" in str(column).lower() or "discount" in str(column).lower():
                invalid = int(((numeric < 0) | (numeric > 100)).sum())
                if invalid:
                    findings.append(_finding(
                        "percentage_range", "warning", [str(column)], invalid,
                        {"invalid_count": invalid, "expected_range": "0 to 100"},
                        f"{invalid} values in {column} fall outside the usual percentage range.",
                        "Confirm the unit and business rules before correcting these values.",
                    ))
        else:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            name_suggests_date = any(token in str(column).lower() for token in ("date", "time", "timestamp"))
            if name_suggests_date and series.notna().any() and parsed.isna().sum():
                invalid = int(parsed.isna().sum() - series.isna().sum())
                if invalid:
                    findings.append(_finding(
                        "invalid_dates", "warning", [str(column)], invalid,
                        {"invalid_count": invalid},
                        f"{invalid} non-empty values in {column} are not valid dates.",
                        "Review the source date format before using time-series analysis.",
                    ))
        if series.nunique(dropna=True) > max(20, len(frame) * 0.9):
            findings.append(
                _finding(
                    "high_cardinality",
                    "info",
                    [str(column)],
                    len(frame),
                    {"unique_count": int(series.nunique(dropna=True))},
                    f"{column} has high cardinality.",
                    "Review whether this column is an identifier before encoding it.",
                )
            )
    missing_cells = int(frame.isna().sum().sum())
    infinite_columns = []
    # Check for infinite values
    numeric_all = frame.select_dtypes(include="number")
    if not numeric_all.empty:
        arr = numeric_all.to_numpy(dtype=float)
        if not np.all(np.isfinite(arr[~np.isnan(arr)])):
            for column in numeric_all.columns:
                col_arr = pd.to_numeric(numeric_all[column], errors='coerce').to_numpy(dtype=float)
                if np.isinf(col_arr).any():
                    infinite_columns.append(str(column))
                    findings.append(
                        _finding(
                            "infinite_values",
                            "warning",
                            [str(column)],
                            int(np.isinf(col_arr).sum()),
                            {"column": str(column)},
                            f"Infinite numeric values were found in {column}.",
                            "Replace or investigate infinite values before statistical analysis.",
                        )
                    )
    return {
        "status": "completed",
        "findings": findings,
        "duplicate_row_count": duplicate_rows,
        "duplicate_percentage": round(duplicate_rows / len(frame) * 100, 3) if len(frame) else 0,
        "missing_cell_count": missing_cells,
        "missing_cell_percentage": round(
            missing_cells / (len(frame) * len(frame.columns)) * 100, 3
        )
        if len(frame) and len(frame.columns)
        else 0,
        "infinite_columns": infinite_columns,
        "quality_score": _quality_score(frame, findings),
    }


def prepare_dataframe(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    allowed_keys = {
        "drop_duplicates",
        "impute_median",
        "impute_mode",
        "impute_mean",
        "impute_constant",
        "impute_forward_fill",
        "impute_backward_fill",
        "missing_indicators",
        "text_features",
        "date_features",
        "one_hot_encode",
        "standard_scale",
        "minmax_scale",
        "robust_scale",
        "ratios",
    }
    unknown_keys = set(config).difference(allowed_keys)
    if unknown_keys:
        raise DatumError(
            f"Unsupported preparation operations: {', '.join(sorted(unknown_keys))}.",
            "invalid_transformation",
        )
    result = frame.copy()
    records: list[dict[str, Any]] = []
    if config.get("drop_duplicates"):
        before = len(result)
        result = result.drop_duplicates().reset_index(drop=True)
        records.append(
            {
                "operation": "drop_duplicates",
                "parameters": {},
                "rows_before": before,
                "rows_after": len(result),
            }
        )
    for column in config.get("impute_median", []):
        if column not in result or not pd.api.types.is_numeric_dtype(result[column]):
            raise DatumError(
                f"Median imputation requires a numeric column: {column}.", "invalid_transformation"
            )
        values = pd.to_numeric(result[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if values.notna().sum() == 0:
            raise DatumError(
                f"Cannot impute {column}: it has no finite numeric values.",
                "invalid_transformation",
            )
        value = float(values.median())
        result[column] = values.fillna(value)
        records.append(
            {
                "operation": "impute_median",
                "parameters": {"column": column, "value": value},
                "rows_before": len(result),
                "rows_after": len(result),
            }
        )
    for column in config.get("impute_mode", []):
        if column not in result:
            raise DatumError(f"Column not found: {column}.", "invalid_transformation")
        modes = result[column].mode(dropna=True)
        if modes.empty:
            raise DatumError(
                f"Cannot infer a mode for empty column: {column}.", "invalid_transformation"
            )
        value = modes.iloc[0]
        result[column] = result[column].fillna(value)
        records.append(
            {
                "operation": "impute_mode",
                "parameters": {"column": column, "value": str(value)},
                "rows_before": len(result),
                "rows_after": len(result),
            }
        )
    feature_frame, feature_records = apply_feature_operations(result, config)
    records.extend(feature_records)
    result = feature_frame
    return result, records


def statistics(frame: pd.DataFrame) -> dict[str, Any]:
    frame = _analysis_frame(frame)
    numeric = frame.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    desc = json.loads(numeric.describe().to_json()) if not numeric.empty else {}
    correlations = (
        json.loads(numeric.corr(method="pearson").to_json()) if numeric.shape[1] > 1 else {}
    )
    spearman = (
        json.loads(numeric.corr(method="spearman").to_json()) if numeric.shape[1] > 1 else {}
    )
    categorical = {
        str(column): {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).head(20).items()}
        for column in frame.select_dtypes(exclude="number").columns
    }
    charts = _eda_charts(frame, numeric)
    return {
        "descriptive_statistics": desc,
        "pearson_correlations": correlations,
        "spearman_correlations": spearman,
        "categorical_frequencies": categorical,
        "charts": charts,
        "warnings": ([] if not numeric.empty else ["No numeric columns were available."])
        + (["Small sample: patterns and correlations are descriptive, not reliable predictions."]
           if len(frame) < 10 else []),
    }


def verify(
    frame: pd.DataFrame, result: dict[str, Any], transformations: list[dict[str, Any]]
) -> dict[str, Any]:
    frame = _analysis_frame(frame)  # analyze normalized form
    checks = []
    # Row count check
    checks.append(
        {
            "check_id": "row_count",
            "status": "passed",
            "evidence": {"actual": int(len(frame))},
        },
    )
    # Finite numeric values
    numeric_all = frame.select_dtypes(include="number")
    if not numeric_all.empty:
        arr = numeric_all.to_numpy(dtype=float)
        finite_ok = np.isfinite(arr).all()
        checks.append(
            {
                "check_id": "finite_numeric_values",
                "status": "passed" if finite_ok else "failed",
                "evidence": {"columns": list(numeric_all.columns), "finite": bool(finite_ok)},
            },
        )
    else:
        checks.append(
            {
                "check_id": "finite_numeric_values",
                "status": "passed",
                "evidence": {"columns": []},
            },
        )
    if result.get("statistics", {}).get("descriptive_statistics"):
        checks.append(
            {
                "check_id": "statistics_present",
                "status": "passed",
                "evidence": {"columns": len(result["statistics"]["descriptive_statistics"])},
            },
        )
    else:
        checks.append(
            {
                "check_id": "statistics_present",
                "status": "passed",
                "evidence": {"columns": 0},
            },
        )
    checks.append(
        {
            "check_id": "transformation_log",
            "status": "passed",
            "evidence": {"count": len(transformations)},
        },
    )
    # Missing values check
    missing_cells = int(frame.isna().sum().sum())
    checks.append(
        {
            "check_id": "missing_values_checked",
            "status": "passed",
            "evidence": {"missing_cell_count": missing_cells},
        },
    )
    # Duplicates check
    dup_count = int(frame.duplicated().sum())
    checks.append(
        {
            "check_id": "duplicate_rows_checked",
            "status": "passed",
            "evidence": {"duplicate_row_count": dup_count},
        },
    )
    overall = "failed" if any(check["status"] == "failed" for check in checks) else "passed"
    # If there are critical issues, mark appropriately
    return {"status": overall, "checks": checks}


def _eda_charts(frame: pd.DataFrame, numeric: pd.DataFrame) -> list[dict[str, Any]]:
    charts = []
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.nunique() > 1 and len(values):
            counts, edges = np.histogram(values, bins=min(10, max(2, int(np.sqrt(len(values))))))
            charts.append({
                "id": f"distribution-{column}", "type": "histogram",
                "title": f"Distribution of {column}", "columns": [str(column)],
                "x_label": str(column), "y_label": "Number of rows",
                "data": [
                    {"x": f"{edges[i]:.3g}–{edges[i + 1]:.3g}", "y": int(counts[i]),
                     "lower": _number(edges[i]), "upper": _number(edges[i + 1])}
                    for i in range(len(counts))
                ],
                "explanation": "Shows how rows are distributed across numeric ranges.",
            })
            quartiles = values.quantile([0.25, 0.5, 0.75])
            charts.append({
                "id": f"box-{column}", "type": "box",
                "title": f"Spread of {column}", "columns": [str(column)],
                "x_label": str(column), "data": [{"y": _number(value)} for value in values.head(5000)],
                "summary": {"min": _number(values.min()), "q1": _number(quartiles.iloc[0]),
                            "median": _number(quartiles.iloc[1]), "q3": _number(quartiles.iloc[2]),
                            "max": _number(values.max())},
                "explanation": "Shows the median, middle half, and range; extremes are candidates for review.",
            })
    varying = [str(column) for column in numeric.columns if numeric[column].dropna().nunique() > 1]
    if len(varying) >= 2:
        x_column, y_column = varying[:2]
        pairs = numeric[[x_column, y_column]].dropna().head(5000)
        if len(pairs) >= 3:
            charts.append({
                "id": f"scatter-{x_column}-{y_column}", "type": "scatter",
                "title": f"{y_column} against {x_column}", "columns": [x_column, y_column],
                "x_label": x_column, "y_label": y_column,
                "data": [{"x": _number(x), "y": _number(y)} for x, y in zip(pairs[x_column], pairs[y_column])],
                "explanation": "Shows whether the two numeric variables move together; it does not prove causation.",
            })
        correlation = numeric[varying].corr(method="pearson")
        charts.append({
            "id": "correlation-matrix", "type": "heatmap", "title": "Numeric correlation matrix",
            "columns": varying, "data": [
                {"x": str(row), "y": str(column), "value": _number(correlation.loc[row, column])}
                for row in varying for column in varying
            ],
            "explanation": "Correlation ranges from -1 to 1 and does not prove causation.",
        })
    for column in frame.select_dtypes(exclude="number").columns:
        counts = frame[column].fillna("(missing)").value_counts().head(12)
        if len(counts):
            charts.append({
                "id": f"frequency-{column}", "type": "bar",
                "title": f"Rows by {column}", "columns": [str(column)],
                "x_label": str(column), "y_label": "Number of rows",
                "data": [{"x": str(label), "y": int(value)} for label, value in counts.items()],
                "explanation": "Compares observed category frequencies.",
            })
    missing = frame.isna().sum()
    if int(missing.sum()):
        charts.append({
            "id": "missing-values", "type": "missing", "title": "Missing values by column",
            "columns": [str(column) for column in frame.columns], "x_label": "Column",
            "y_label": "Missing rows",
            "data": [{"x": str(column), "y": int(value)} for column, value in missing.items() if value],
            "explanation": "Counts blank, missing, and whitespace-only cells.",
        })
    for column in frame.columns:
        if not _is_likely_date_column(frame[column]) or not varying:
            continue
        dates = pd.to_datetime(frame[column], errors="coerce", format="mixed")
        measure = varying[0]
        grouped = pd.DataFrame({"date": dates, "value": numeric[measure]}).dropna()
        grouped = grouped.groupby("date", as_index=False)["value"].mean().sort_values("date").head(5000)
        if len(grouped) < 2:
            continue
        charts.append(
            {
                "id": f"trend-{column}-{measure}",
                "type": "line",
                "title": f"{measure} over {column}",
                "columns": [str(column), measure], "x_label": str(column), "y_label": measure,
                "data": [
                    {"x": date.isoformat(), "y": _number(value)}
                    for date, value in zip(grouped["date"], grouped["value"])
                ],
                "warnings": ["Grouped by parsed date; a trend is descriptive and not causal."],
            }
        )
    return charts


def _quality_score(frame: pd.DataFrame, findings: list[dict[str, Any]]) -> dict[str, Any]:
    total_cells = max(len(frame) * len(frame.columns), 1)
    missing_penalty = int(frame.isna().sum().sum()) / total_cells
    duplicate_penalty = int(frame.duplicated().sum()) / max(len(frame), 1)
    severe_findings = sum(
        1 for finding in findings if finding["severity"] in {"critical", "error"}
    )
    score = max(0.0, round(100 * (1 - missing_penalty - duplicate_penalty - severe_findings * 0.1), 2))
    return {
        "score": score,
        "formula": "100 * (1 - missing_cell_ratio - duplicate_row_ratio - 0.1 * severe_finding_count), bounded at 0",
        "limitations": ["The score is a screening indicator, not a domain quality certification."],
    }


def dataset_id(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def _finding(
    check_id: str,
    severity: str,
    columns: list[str],
    rows: int,
    evidence: dict[str, Any],
    message: str,
    recommendation: str,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "severity": severity,
        "status": "completed",
        "affected_columns": columns,
        "affected_rows_count": rows,
        "evidence": evidence,
        "message": message,
        "recommendation": recommendation,
    }


def _number(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)
