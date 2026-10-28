from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def add_dataset_context(
    charts: list[dict[str, Any]],
    dataset_name: str,
    row_count: int,
    column_count: int,
) -> list[dict[str, Any]]:
    result = []
    for chart in charts:
        enriched = dict(chart)
        chart_type = str(chart.get("type", ""))
        aggregation = {
            "histogram": "count by numeric bin",
            "box": "distribution summary",
            "bar": "count by category",
            "missing": "missing row count",
            "scatter": "row-level values",
            "heatmap": "Pearson correlation",
            "line": "mean by parsed date",
        }.get(chart_type, "not specified")
        enriched["dataset_context"] = {
            "dataset_name": dataset_name,
            "rows": row_count,
            "columns": column_count,
            "source_columns": list(chart.get("columns", [])),
            "aggregation": chart.get("aggregation", aggregation),
            "filters": {},
            "record_scope": "all applicable records",
            "exclusions": "Invalid or missing values are excluded where required by the chart.",
        }
        result.append(enriched)
    return result


def generate_chart_images(
    run_id: str,
    frame: pd.DataFrame,
    charts: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    output_dir = output_dir or Path(__file__).resolve().parents[2] / "data" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, Any]] = []
    for chart in charts:
        rendered = dict(chart)
        try:
            filename = _filename(run_id, str(chart["id"]))
            _render_chart(frame, chart, output_dir / filename)
            rendered["image_url"] = f"/api/charts/{filename}"
        except (KeyError, TypeError, ValueError):
            rendered.setdefault("warnings", []).append(
                "An image could not be generated for this chart specification."
            )
        result.append(rendered)
    return result


def _filename(run_id: str, chart_id: str) -> str:
    safe_run = re.sub(r"[^A-Za-z0-9_-]", "_", run_id)
    safe_chart = re.sub(r"[^A-Za-z0-9_-]", "_", chart_id)
    return f"{safe_run}-{safe_chart}.png"


def _render_chart(frame: pd.DataFrame, chart: dict[str, Any], output: Path) -> None:
    chart_type = chart["type"]
    columns = chart.get("columns", [])
    data = chart.get("data", [])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if chart_type in {"histogram", "bar", "missing"}:
        labels = [str(point.get("x", point.get("value", ""))) for point in data]
        values = [float(point.get("y", point.get("value", 0)) or 0) for point in data]
        axis.bar(labels, values, color="#81b9de")
        axis.tick_params(axis="x", rotation=45)
    elif chart_type == "box":
        if not columns:
            raise ValueError("A box plot requires a source column.")
        values = pd.to_numeric(frame[columns[0]], errors="coerce").dropna()
        if values.empty:
            raise ValueError("A box plot requires numeric observations.")
        axis.boxplot(values, vert=False)
        axis.set_yticks([1], [columns[0]])
    elif chart_type == "scatter":
        x_column, y_column = columns[:2]
        values = frame[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if values.empty:
            raise ValueError("A scatter plot requires two numeric columns.")
        axis.scatter(values[x_column], values[y_column], color="#c4f05f")
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)
    elif chart_type == "line":
        if len(columns) < 2:
            raise ValueError("A line chart requires a date and numeric column.")
        if data:
            points = pd.DataFrame(
                {
                    "date": [point.get("x") for point in data],
                    "value": [point.get("y") for point in data],
                }
            )
            points["date"] = pd.to_datetime(points["date"], errors="coerce", format="mixed")
            points["value"] = pd.to_numeric(points["value"], errors="coerce")
            points = points.dropna()
        else:
            dates = pd.to_datetime(frame[columns[0]], errors="coerce", format="mixed")
            values = pd.to_numeric(frame[columns[1]], errors="coerce")
            points = pd.DataFrame({"date": dates, "value": values}).dropna()
        if len(points) < 2:
            raise ValueError("A line chart requires at least two valid observations.")
        points = points.groupby("date", as_index=False)["value"].mean().sort_values("date")
        axis.plot(points["date"], points["value"], marker="o", color="#c4f05f")
        axis.set_xlabel(columns[0])
        axis.set_ylabel(columns[1])
        figure.autofmt_xdate()
    elif chart_type == "heatmap":
        if not columns:
            raise ValueError("A heatmap requires numeric columns.")
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
        matrix = numeric.corr(method="pearson")
        image = axis.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
        axis.set_yticks(range(len(columns)), columns)
        figure.colorbar(image, ax=axis, label="Pearson correlation")
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    axis.set_title(str(chart.get("title", "DATUM chart")))
    axis.set_xlabel(str(chart.get("x_label", axis.get_xlabel())))
    axis.set_ylabel(str(chart.get("y_label", axis.get_ylabel())))
    figure.tight_layout()
    figure.savefig(output, dpi=120, format="png")
    plt.close(figure)