from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app import charts as chart_utils
from backend.app.data_processing import _is_likely_date_column
from backend.app.errors import DatumError


@dataclass
class AnalysisContext:
    run_id: str
    dataset_id: str
    metric: str | None = None
    date_column: str | None = None
    granularity: str = "M"
    filters: dict[str, str] | None = None
    chart_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "metric": self.metric,
            "date_column": self.date_column,
            "granularity": self.granularity,
            "filters": self.filters or {},
            "chart_id": self.chart_id,
        }


def answer_question(
    run_id: str,
    dataset_id: str,
    frame: pd.DataFrame,
    question: str,
    supplied_context: dict[str, Any] | None = None,
    chart_output_dir: Path | None = None,
) -> dict[str, Any]:
    context = _context(run_id, dataset_id, supplied_context)
    if not question.strip():
        raise DatumError("Ask a question about the dataset.", "invalid_question")
    # Work with full source dataset - charts are optional context only
    filtered = _apply_filter(frame, context)
    lowered = question.lower()
    _validate_explicit_column(question, frame)
    _update_filter_from_question(context, frame, question)
    filtered = _apply_filter(frame, context)

    # Positional/structural questions
    if "first row and first column" in lowered or ("first row" in lowered and "first column" in lowered):
        return _first_cell_answer(context, filtered)
    if re.search(r"\bfirst\s+row\b", lowered) and "column" not in lowered:
        return _first_row_answer(context, filtered)
    if re.search(r"\bfirst\s+column\b", lowered):
        return _first_column_answer(context, filtered)
    if re.search(r"show\s+first\s+(\d+)\s+rows?", lowered):
        match = re.search(r"show\s+first\s+(\d+)\s+rows?", lowered)
        n = int(match.group(1)) if match else 5
        return _show_rows_answer(context, filtered, n)
    if re.search(r"first\s+(\d+)\s+rows?", lowered):
        match = re.search(r"first\s+(\d+)\s+rows?", lowered)
        n = int(match.group(1)) if match else 5
        return _show_rows_answer(context, filtered, n)
    if "column names" in lowered or ("what are the columns" in lowered):
        return _columns_answer(context, filtered)
    if "how many rows and columns" in lowered or ("rows and columns" in lowered and "how many" in lowered):
        return _shape_answer(context, filtered)
    if re.search(r"how many rows", lowered) and "column" not in lowered:
        return _shape_answer(context, filtered)
    if re.search(r"how many columns", lowered):
        return _shape_answer(context, filtered)
    if "average of" in lowered:
        return _average_answer(context, filtered, question)
    if "which category has the highest count" in lowered or "highest count" in lowered:
        return _highest_count_answer(context, filtered, question)
    if "which values are missing" in lowered or "missing values" in lowered:
        return _missing_values_detail_answer(context, filtered)
    if "unique values of" in lowered:
        return _unique_values_answer(context, filtered, question)
    if "compare" in lowered and ("and" in lowered):
        return _compare_answer(context, filtered, question)
    if "find correlations among numeric columns" in lowered or "correlations among" in lowered:
        return _correlations(context, filtered)

    if any(token in lowered for token in ("overview", "schema", "dataset size")) and not re.search(r"how many", lowered):
        return _overview_answer(context, filtered)
    column_summary = _column_summary(context, filtered, question)
    if column_summary is not None:
        return column_summary
    if "correlat" in lowered and "among" not in lowered:
        return _correlations(context, filtered)
    if "previous month" in lowered or "previous period" in lowered:
        return _period_comparison(context, filtered)
    if any(
        token in lowered
        for token in (
            "monthly",
            "weekly",
            "daily",
            "quarterly",
            "yearly",
            "trend",
            "over time",
            "month",
            "quarter",
            "show.*trend",
        )
    ) or ("trend" in lowered):
        return _trend_answer(run_id, context, filtered, question, chart_output_dir)
    if "top" in lowered or "highest" in lowered or "largest" in lowered:
        return _ranking(
            run_id,
            context,
            filtered,
            question,
            descending=True,
            chart_output_dir=chart_output_dir,
        )
    if "lowest" in lowered or "smallest" in lowered:
        return _ranking(
            run_id,
            context,
            filtered,
            question,
            descending=False,
            chart_output_dir=chart_output_dir,
        )
    if ("missing" in lowered and "values are missing" not in lowered) or "unusual" in lowered or "outlier" in lowered:
        return _quality_answer(context, filtered)
    if "explain" in lowered and context.chart_id:
        return _chart_explanation(context)
    return _summary_answer(context, filtered)


def _column_summary(
    context: AnalysisContext, frame: pd.DataFrame, question: str
) -> dict[str, Any] | None:
    lowered = question.lower()
    column = next(
        (str(name) for name in frame.columns if str(name).lower() in lowered),
        None,
    )
    if column is None or not any(
        token in lowered for token in ("summary", "describe", "distribution", "about")
    ):
        return None
    series = frame[column]
    evidence: list[dict[str, Any]] = [
        {"label": "Column", "value": column},
        {"label": "Data type", "value": str(series.dtype)},
        {"label": "Missing values", "value": int(series.isna().sum())},
        {"label": "Unique values", "value": int(series.nunique(dropna=True))},
    ]
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        evidence.extend(
            [
                {"label": "Mean", "value": _json_number(numeric.mean())},
                {"label": "Median", "value": _json_number(numeric.median())},
                {"label": "Minimum", "value": _json_number(numeric.min())},
                {"label": "Maximum", "value": _json_number(numeric.max())},
            ]
        )
    else:
        evidence.append(
            {"label": "Most frequent values", "value": ", ".join(
                str(value) for value in series.value_counts(dropna=True).head(5).index
            )}
        )
    return {
        "answer": f"Here is the deterministic summary for the `{column}` column.",
        "evidence": evidence,
        "limitations": ["Summary statistics exclude non-finite numeric values."],
        "context": context.as_dict(),
    }


def _overview_answer(context: AnalysisContext, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "answer": f"This dataset has {len(frame)} rows and {len(frame.columns)} columns.",
        "evidence": [
            {"label": "Rows", "value": int(len(frame))},
            {"label": "Columns", "value": int(len(frame.columns))},
            {"label": "Column names", "value": ", ".join(str(column) for column in frame.columns)},
            {
                "label": "Data types",
                "value": "; ".join(f"{column}: {dtype}" for column, dtype in frame.dtypes.items()),
            },
        ],
        "limitations": ["Types are inferred by Pandas from the persisted source file."],
        "context": context.as_dict(),
    }


def _json_number(value: Any) -> float | None:
    return None if pd.isna(value) or not np.isfinite(value) else float(value)


def _validate_explicit_column(question: str, frame: pd.DataFrame) -> None:
    available_columns = [str(column) for column in frame.columns]
    lowered = question.lower()
    if any(column.lower() in lowered for column in available_columns):
        return
    match = re.search(r"\bcolumn\s+['\"]?([A-Za-z_][\w .-]*)['\"]?", question, re.IGNORECASE)
    if not match:
        return
    requested = re.split(
        r"\s+(?:summary|distribution|statistics|stats|value|values)\b",
        match.group(1).strip().rstrip("?."),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    if requested not in set(available_columns):
        available = ", ".join(available_columns[:10])
        suffix = " ..." if len(frame.columns) > 10 else ""
        raise DatumError(
            f"Column '{requested}' was not found. Available columns: {available}{suffix}.",
            "column_not_found",
        )


def _context(run_id: str, dataset_id: str, value: dict[str, Any] | None) -> AnalysisContext:
    value = value or {}
    if value.get("run_id") not in {None, run_id} or value.get("dataset_id") not in {
        None,
        dataset_id,
    }:
        raise DatumError("Analysis context belongs to a different run.", "context_mismatch")
    filters = value.get("filters") or {}
    if not isinstance(filters, dict):
        raise DatumError("Analysis filters must be an object.", "invalid_context")
    return AnalysisContext(
        run_id=run_id,
        dataset_id=dataset_id,
        metric=value.get("metric"),
        date_column=value.get("date_column"),
        granularity=_normalize_granularity(value.get("granularity", "M")),
        filters={str(key): str(item) for key, item in filters.items()},
        chart_id=value.get("chart_id"),
    )


def _apply_filter(frame: pd.DataFrame, context: AnalysisContext) -> pd.DataFrame:
    result = frame
    for column, value in (context.filters or {}).items():
        if column not in result.columns:
            raise DatumError(f"Column not found: {column}.", "invalid_filter")
        result = result[result[column].astype("string") == value]
    return result


def _update_filter_from_question(
    context: AnalysisContext, frame: pd.DataFrame, question: str
) -> None:
    lowered = question.lower()
    for column in frame.select_dtypes(exclude="number").columns:
        values = {str(value): value for value in frame[column].dropna().unique()}
        for label, _value in values.items():
            if label.lower() in lowered and len(label) >= 2:
                context.filters = {column: label}
                return


def _numeric_column(frame: pd.DataFrame, question: str, context: AnalysisContext) -> str:
    numeric = list(frame.select_dtypes(include="number").columns)
    requested = next(
        (column for column in numeric if str(column).lower() in question.lower()), None
    )
    if requested:
        context.metric = str(requested)
        return str(requested)
    if context.metric in numeric:
        return str(context.metric)
    if not numeric:
        raise DatumError(
            "This question needs a numeric column, but none is available.", "unsupported_analysis"
        )
    context.metric = str(numeric[0])
    return str(numeric[0])


def _date_column(frame: pd.DataFrame, context: AnalysisContext) -> str:
    if context.date_column in frame.columns:
        return str(context.date_column)
    for column in frame.columns:
        if _is_likely_date_column(frame[column]):
            context.date_column = str(column)
            return str(column)
    raise DatumError(
        "No suitable date or time column was found for this analysis.", "no_time_dimension"
    )


def _granularity(question: str, context: AnalysisContext) -> str:
    for value, alias in (
        ("daily", "D"),
        ("weekly", "W"),
        ("monthly", "M"),
        ("quarterly", "Q"),
        ("yearly", "Y"),
    ):
        if value in question.lower():
            context.granularity = alias
    return context.granularity


def _normalize_granularity(value: Any) -> str:
    aliases = {
        "day": "D",
        "daily": "D",
        "week": "W",
        "weekly": "W",
        "month": "M",
        "monthly": "M",
        "quarter": "Q",
        "quarterly": "Q",
        "year": "Y",
        "yearly": "Y",
    }
    normalized = aliases.get(str(value).strip().lower(), str(value).strip().upper())
    if normalized not in {"D", "W", "M", "Q", "Y"}:
        raise DatumError(
            "Use daily, weekly, monthly, quarterly, or yearly granularity.",
            "invalid_granularity",
        )
    return normalized


def _trend_answer(
    run_id: str,
    context: AnalysisContext,
    frame: pd.DataFrame,
    question: str,
    chart_output_dir: Path | None,
) -> dict[str, Any]:
    date_column = _date_column(frame, context)
    metric = _numeric_column(frame, question, context)
    granularity = _granularity(question, context)
    dates = pd.to_datetime(frame[date_column], errors="coerce", format="mixed")
    values = pd.to_numeric(frame[metric], errors="coerce")
    points = pd.DataFrame({"date": dates, "value": values}).dropna()
    if len(points) < 2:
        raise DatumError("At least two valid dated observations are required.", "insufficient_data")
    periods = points["date"].dt.to_period(granularity)
    grouped = points.assign(period=periods).groupby("period", as_index=False)["value"].sum()
    grouped = grouped.sort_values("period")
    labels = grouped["period"].astype(str).tolist()
    series = grouped["value"].astype(float).tolist()
    change = _percentage_change(series[-2], series[-1]) if len(series) > 1 else None
    peak = int(np.argmax(series))
    chart = {
        "id": f"chat-trend-{metric}-{granularity}",
        "type": "line",
        "title": f"{metric} by {granularity}",
        "columns": [date_column, metric],
        "aggregation": "sum by period",
        "x_label": date_column,
        "y_label": metric,
        "data": [
            {"x": label, "y": float(value)}
            for label, value in zip(labels, series, strict=False)
        ],
        "explanation": "Values are summed by period from valid date and numeric observations.",
        "warnings": ["Trend is descriptive and does not establish causation."],
    }
    try:
        rendered = chart_utils.generate_chart_images(
            run_id, frame, [chart], output_dir=chart_output_dir
        )[0]
    except Exception:
        rendered = chart
    try:
        rendered = chart_utils.add_dataset_context(
            [rendered],
            dataset_name=context.dataset_id,
            row_count=len(frame),
            column_count=len(frame.columns),
        )[0]
    except Exception:
        pass
    context.chart_id = chart["id"]
    evidence = [
        {"label": "Peak period", "value": labels[peak]},
        {"label": "Peak value", "value": float(series[peak])},
        {"label": "Latest period", "value": labels[-1]},
        {"label": "Latest value", "value": float(series[-1])},
    ]
    if change is not None:
        evidence.append({"label": "Latest period change", "value": change, "unit": "%"})
    return {
        "answer": (
            f"{metric} peaks in {labels[peak]} at {series[peak]:g}. "
            f"The latest period is {labels[-1]} with {series[-1]:g}."
        ),
        "evidence": evidence,
        "limitations": ["Only observed periods are plotted; missing periods are not filled."],
        "chart": rendered,
        "context": context.as_dict(),
    }


def _ranking(
    run_id: str,
    context: AnalysisContext,
    frame: pd.DataFrame,
    question: str,
    descending: bool,
    chart_output_dir: Path | None,
) -> dict[str, Any]:
    metric = _numeric_column(frame, question, context)
    candidates = [
        column
        for column in frame.columns
        if column != metric and not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not candidates:
        raise DatumError(
            "A ranking needs a categorical column and a numeric measure.", "unsupported_analysis"
        )
    group = candidates[0]
    values = (
        frame.assign(_value=pd.to_numeric(frame[metric], errors="coerce"))
        .groupby(group)["_value"]
        .sum()
        .dropna()
    )
    values = values.sort_values(ascending=not descending).head(_top_n(question))
    if values.empty:
        raise DatumError(
            "No valid numeric observations were available for the ranking.", "insufficient_data"
        )
    chart = {
        "id": f"chat-ranking-{group}-{metric}",
        "type": "bar",
        "title": f"{metric} by {group}",
        "columns": [group, metric],
        "aggregation": "sum by category",
        "x_label": group,
        "y_label": metric,
        "data": [{"x": str(label), "y": float(value)} for label, value in values.items()],
        "explanation": f"Categories are ranked by the sum of {metric}.",
    }
    rendered = chart_utils.generate_chart_images(
        run_id, frame, [chart], output_dir=chart_output_dir
    )[0]
    rendered = chart_utils.add_dataset_context(
        [rendered],
        dataset_name=context.dataset_id,
        row_count=len(frame),
        column_count=len(frame.columns),
    )[0]
    context.chart_id = chart["id"]
    label = "highest" if descending else "lowest"
    return {
        "answer": f"The {label} {group} is {values.index[0]} with {values.iloc[0]:g} {metric}.",
        "evidence": [{"label": str(key), "value": float(value)} for key, value in values.items()],
        "limitations": [
            "Ranking uses the sum of the selected numeric column and excludes "
            "invalid numeric values."
        ],
        "chart": rendered,
        "context": context.as_dict(),
    }


def _period_comparison(context: AnalysisContext, frame: pd.DataFrame) -> dict[str, Any]:
    date_column = _date_column(frame, context)
    metric = _numeric_column(frame, context.metric or "", context)
    points = (
        pd.DataFrame(
            {
                "period": pd.to_datetime(
                    frame[date_column], errors="coerce", format="mixed"
                ).dt.to_period(context.granularity),
                "value": pd.to_numeric(frame[metric], errors="coerce"),
            }
        )
        .dropna()
        .groupby("period")["value"]
        .sum()
        .sort_index()
    )
    if len(points) < 2:
        raise DatumError(
            "At least two observed periods are required for comparison.", "insufficient_data"
        )
    current, previous = float(points.iloc[-1]), float(points.iloc[-2])
    change = _percentage_change(previous, current)
    evidence = [
        {"label": "Current period", "value": current},
        {"label": "Previous period", "value": previous},
    ]
    if change is not None:
        evidence.append({"label": "Change", "value": change, "unit": "%"})
    return {
        "answer": (
            f"{metric} changed from {previous:g} in {points.index[-2]} "
            f"to {current:g} in {points.index[-1]}."
        ),
        "evidence": evidence,
        "limitations": ["A percentage change is unavailable when the previous period is zero."],
        "context": context.as_dict(),
    }


def _correlations(context: AnalysisContext, frame: pd.DataFrame) -> dict[str, Any]:
    numeric = frame.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    matrix = numeric.corr(method="pearson")
    pairs = []
    for left_index, left in enumerate(matrix.columns):
        for right in matrix.columns[left_index + 1 :]:
            value = matrix.loc[left, right]
            if pd.notna(value):
                pairs.append((abs(float(value)), left, right, float(value)))
    pairs.sort(reverse=True)
    return {
        "answer": (
            "The strongest observed numeric relationships are listed below. "
            "Correlation does not prove causation."
        ),
        "evidence": [
            {"label": f"{left} and {right}", "value": value} for _, left, right, value in pairs[:5]
        ],
        "limitations": [
            "Pearson correlation uses available paired numeric observations and is descriptive."
        ],
        "context": context.as_dict(),
    }


def _quality_answer(context: AnalysisContext, frame: pd.DataFrame) -> dict[str, Any]:
    missing = int(frame.isna().sum().sum())
    numeric = frame.select_dtypes(include="number")
    non_finite = (
        int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()) if not numeric.empty else 0
    )
    return {
        "answer": (
            f"The current filtered data contains {missing} missing cells and "
            f"{non_finite} non-finite numeric values."
        ),
        "evidence": [
            {"label": "Rows inspected", "value": int(len(frame))},
            {"label": "Missing cells", "value": missing},
            {"label": "Non-finite numeric values", "value": non_finite},
        ],
        "limitations": [
            "Potential outliers require domain context and are not automatically treated as errors."
        ],
        "context": context.as_dict(),
    }


def _first_cell_answer(context: AnalysisContext, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise DatumError("The dataset has no rows.", "empty_dataset")
    first_row = frame.iloc[0]
    first_col = frame.columns[0]
    value = first_row[first_col]
    return {
        "answer": f"The value in the first row and first column is {format_value_answer(value)} (row 1, column '{first_col}').",
        "evidence": [
            {"label": "Row index", "value": 1},
            {"label": "Column", "value": str(first_col)},
            {"label": "Value", "value": format_value_answer(value)},
        ],
        "limitations": ["Row numbering excludes the header and uses the dataset's stable preview order."],
        "context": context.as_dict(),
    }


def _first_row_answer(context: AnalysisContext, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise DatumError("The dataset has no rows.", "empty_dataset")
    first_row = frame.iloc[0]
    evidence = [{"label": str(col), "value": format_value_answer(first_row[col])} for col in frame.columns]
    return {
        "answer": f"The first data row (row 1) contains values for all {len(frame.columns)} columns.",
        "evidence": evidence,
        "limitations": ["Row numbering excludes the header and uses the dataset's stable preview order."],
        "context": context.as_dict(),
    }


def _first_column_answer(context: AnalysisContext, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or len(frame.columns) == 0:
        raise DatumError("The dataset has no columns.", "empty_dataset")
    first_col = frame.columns[0]
    values = frame[first_col]
    evidence = [{"label": f"Row {i+1}", "value": format_value_answer(values.iloc[i])} for i in range(min(len(values), 10))]
    if len(values) > 10:
        evidence.append({"label": "...", "value": f"({len(values) - 10} more rows)"})
    return {
        "answer": f"The first column is '{first_col}' with {len(values)} values.",
        "evidence": evidence,
        "limitations": ["Only first 10 values are shown in evidence for brevity."],
        "context": context.as_dict(),
    }


def _show_rows_answer(context: AnalysisContext, frame: pd.DataFrame, n: int) -> dict[str, Any]:
    if frame.empty:
        raise DatumError("The dataset has no rows.", "empty_dataset")
    n = max(1, min(int(n), 100))
    shown = frame.head(n)
    evidence = []
    for idx, (_, row) in enumerate(shown.iterrows(), start=1):
        evidence.append({"label": f"Row {idx}", "value": "; ".join(f"{col}={format_value_answer(row[col])}" for col in frame.columns)})
    return {
        "answer": f"Showing first {len(shown)} rows (rows 1-{len(shown)}).",
        "evidence": evidence,
        "limitations": ["Row numbering excludes the header and uses the dataset's stable preview order."],
        "context": context.as_dict(),
    }


def _columns_answer(context: AnalysisContext, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "answer": f"The dataset has {len(frame.columns)} columns.",
        "evidence": [{"label": "Column names", "value": ", ".join(str(c) for c in frame.columns)}],
        "limitations": [],
        "context": context.as_dict(),
    }


def _shape_answer(context: AnalysisContext, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "answer": f"The dataset has {len(frame)} rows and {len(frame.columns)} columns.",
        "evidence": [
            {"label": "Rows", "value": int(len(frame))},
            {"label": "Columns", "value": int(len(frame.columns))},
        ],
        "limitations": [],
        "context": context.as_dict(),
    }


def _average_answer(context: AnalysisContext, frame: pd.DataFrame, question: str) -> dict[str, Any]:
    metric = _numeric_column(frame, question, context)
    series = pd.to_numeric(frame[metric], errors="coerce")
    avg = series.mean()
    return {
        "answer": f"The average of {metric} is {format_value_answer(avg)}.",
        "evidence": [
            {"label": "Column", "value": metric},
            {"label": "Average", "value": format_value_answer(avg)},
            {"label": "Count (non-null numeric)", "value": int(series.notna().sum())},
        ],
        "limitations": ["Non-numeric/non-finite values are excluded from the calculation."],
        "context": context.as_dict(),
    }


def _highest_count_answer(context: AnalysisContext, frame: pd.DataFrame, question: str) -> dict[str, Any]:
    candidates = [col for col in frame.columns if not pd.api.types.is_numeric_dtype(frame[col])]
    if not candidates:
        raise DatumError("No categorical column available for counting.", "unsupported_analysis")
    group = candidates[0]
    counts = frame[group].value_counts(dropna=False)
    if counts.empty:
        raise DatumError("No values available for counting.", "insufficient_data")
    top_cat = counts.index[0]
    return {
        "answer": f"The category with the highest count is '{format_value_answer(top_cat)}' with {int(counts.iloc[0])} occurrences.",
        "evidence": [
            {"label": "Category column", "value": group},
            {"label": str(top_cat), "value": int(counts.iloc[0])},
        ],
        "limitations": [],
        "context": context.as_dict(),
    }


def _missing_values_detail_answer(context: AnalysisContext, frame: pd.DataFrame) -> dict[str, Any]:
    missing = frame.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return {
            "answer": "There are no missing values in the dataset.",
            "evidence": [{"label": "Missing cells", "value": 0}],
            "limitations": [],
            "context": context.as_dict(),
        }
    evidence = [{"label": str(col), "value": int(missing[col])} for col in missing.index]
    return {
        "answer": f"Missing values found in {len(missing)} column(s) totaling {int(missing.sum())} cells.",
        "evidence": evidence,
        "limitations": ["Whitespace-only values may be treated as missing depending on normalization."],
        "context": context.as_dict(),
    }


def _unique_values_answer(context: AnalysisContext, frame: pd.DataFrame, question: str) -> dict[str, Any]:
    lowered = question.lower()
    column = next((str(c) for c in frame.columns if str(c).lower() in lowered), None)
    if column is None:
        # try to find after "of"
        match = re.search(r"unique values of\s+([A-Za-z_][\w .-]*)", question, re.IGNORECASE)
        if match:
            column = match.group(1).strip()
    if column not in frame.columns:
        raise DatumError(f"Column '{column}' was not found.", "column_not_found")
    uniques = frame[column].dropna().unique()
    evidence = [{"label": str(u), "value": "present"} for u in list(uniques)[:50]]
    return {
        "answer": f"Column '{column}' has {len(uniques)} unique values.",
        "evidence": evidence,
        "limitations": ["Only up to 50 unique values are shown in evidence."],
        "context": context.as_dict(),
    }


def _compare_answer(context: AnalysisContext, frame: pd.DataFrame, question: str) -> dict[str, Any]:
    # Try to find two categorical values
    lowered = question.lower()
    cat_cols = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    num_cols = list(frame.select_dtypes(include="number").columns)
    if not cat_cols or not num_cols:
        raise DatumError("Comparison requires a categorical and numeric column.", "unsupported_analysis")
    # simple comparison
    group = cat_cols[0]
    metric = num_cols[0]
    # find values mentioned
    vals = []
    for v in frame[group].dropna().unique():
        if str(v).lower() in lowered:
            vals.append(str(v))
    if len(vals) < 2:
        # just compare groups
        res = frame.groupby(group)[metric].agg(['count', 'sum', 'mean']).head(10)
        evidence = []
        for idx, row in res.iterrows():
            evidence.append({"label": f"{group}={idx}", "value": f"sum={row['sum']}, mean={row['mean']:.3g}"})
        return {
            "answer": f"Comparison of {metric} across {group} groups.",
            "evidence": evidence,
            "limitations": ["Showing group aggregates."],
            "context": context.as_dict(),
        }
    else:
        g1 = frame[frame[group].astype(str) == vals[0]]
        g2 = frame[frame[group].astype(str) == vals[1]]
        s1 = pd.to_numeric(g1[metric], errors='coerce')
        s2 = pd.to_numeric(g2[metric], errors='coerce')
        return {
            "answer": f"Comparing {metric} between {vals[0]} and {vals[1]}.",
            "evidence": [
                {"label": f"{vals[0]} count", "value": int(s1.notna().sum())},
                {"label": f"{vals[0]} mean", "value": float(s1.mean()) if pd.notna(s1.mean()) else None},
                {"label": f"{vals[0]} sum", "value": float(s1.sum()) if pd.notna(s1.sum()) else None},
                {"label": f"{vals[1]} count", "value": int(s2.notna().sum())},
                {"label": f"{vals[1]} mean", "value": float(s2.mean()) if pd.notna(s2.mean()) else None},
                {"label": f"{vals[1]} sum", "value": float(s2.sum()) if pd.notna(s2.sum()) else None},
            ],
            "limitations": [],
            "context": context.as_dict(),
        }


def format_value_answer(value: Any) -> str:
    if value is None or (isinstance(value, float) and (pd.isna(value) or not np.isfinite(value))):
        return "null"
    try:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        # avoid scientific notation for simple cases
        if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
            return str(value)
        return f"{value}"
    return str(value)


def _chart_explanation(context: AnalysisContext) -> dict[str, Any]:
    return {
        "answer": (
            "The selected graph is a descriptive view of the current analysis. "
            "Its title, columns, and aggregation describe the plotted evidence; "
            "it does not establish causation."
        ),
        "evidence": [{"label": "Chart", "value": context.chart_id}],
        "limitations": ["The selected chart metadata is not persisted across browser sessions."],
        "context": context.as_dict(),
    }


def _percentage_change(previous: float, current: float) -> float | None:
    return None if previous == 0 else round((current - previous) / abs(previous) * 100, 2)


def _top_n(question: str) -> int:
    match = re.search(r"\btop\s+(\d+)", question.lower())
    return max(1, min(int(match.group(1)), 50)) if match else 10
