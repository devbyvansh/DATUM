from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.app import charts as chart_utils
from backend.app.analysis_chat import answer_question
from backend.app.config import get_settings
from backend.app.connectors.sql_database import (
    describe_table,
    list_tables,
    query_dataframe,
    query_frame,
    test_connection,
)
from backend.app.data_processing import (
    _is_likely_date_column,
    dataset_id,
    load_dataframe,
    profile_dataframe,
    quality_report,
)
from backend.app.database import Store, utc_now
from backend.app.errors import DatumError, not_found
from backend.app.notebook_generator import generate_notebook
from backend.app.workflows.graphs import DatumGraphs, resume_graph

settings = get_settings()
logger = logging.getLogger("datum.analysis")
for directory in (settings.upload_dir, settings.prepared_dir, settings.report_dir):
    directory.mkdir(parents=True, exist_ok=True)
store = Store(settings.database_path)
graphs = DatumGraphs(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    graphs.close()
    store.close()


app = FastAPI(
    title="DATUM Backend API",
    version="0.1.0",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "datum-backend"}


class WorkflowRequest(BaseModel):
    mode: str = Field(pattern="^(analyze_only|prepare_for_ml)$")
    preparation: dict = Field(default_factory=dict)


class ApprovalRequest(BaseModel):
    decision: str = Field(pattern="^(approved|rejected)$")


class SqlConnectionRequest(BaseModel):
    connection_url: str = Field(min_length=1, max_length=2048)
    schema: str | None = Field(default=None, max_length=128)


class SqlQueryRequest(SqlConnectionRequest):
    query: str = Field(min_length=1, max_length=10000)
    limit: int = Field(default=1000, ge=1, le=10000)


class SqlDatasetRequest(SqlQueryRequest):
    filename: str = Field(default="sql_source.csv", min_length=1, max_length=128)


class SqlTableRequest(SqlConnectionRequest):
    table: str = Field(min_length=1, max_length=128)


@app.exception_handler(DatumError)
async def datum_error_handler(_, exc: DatumError):
    return JSONResponse(
        status_code=400, content={"detail": {"code": exc.code, "message": exc.message}}
    )


@app.get("/api/charts/{filename}")
def get_chart(filename: str):
    if Path(filename).name != filename or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(
            400,
            detail={"code": "invalid_path", "message": "Invalid chart filename."},
        )
    chart_path = (settings.report_dir.parent / "charts") / filename
    if not chart_path.is_file():
        raise HTTPException(404, detail={"code": "chart_not_found", "message": "Chart not found."})
    return FileResponse(chart_path, media_type="image/png")


@app.post("/api/sql/test")
def sql_test(request: SqlConnectionRequest) -> dict:
    return test_connection(request.connection_url)


@app.post("/api/sql/tables")
def sql_tables(request: SqlConnectionRequest) -> dict:
    return list_tables(request.connection_url, request.schema)


@app.post("/api/sql/table")
def sql_table(request: SqlTableRequest) -> dict:
    return describe_table(request.connection_url, request.table, request.schema)


@app.post("/api/sql/query")
def sql_query(request: SqlQueryRequest) -> dict:
    return query_dataframe(
        request.connection_url,
        request.query,
        min(request.limit, settings.max_sql_rows),
    )


@app.post("/api/sql/datasets", status_code=201)
def create_sql_dataset(request: SqlDatasetRequest) -> dict:
    frame = query_frame(
        request.connection_url,
        request.query,
        min(request.limit, settings.max_sql_rows),
    )
    content = frame.to_csv(index=False).encode("utf-8")
    identifier = dataset_id(content)
    path = settings.upload_dir / f"{identifier}.csv"
    path.write_bytes(content)
    metadata = profile_dataframe(frame)
    store.add_dataset(
        {
            "id": identifier,
            "filename": request.filename,
            "path": str(path),
            "size_bytes": len(content),
            "created_at": utc_now(),
            "metadata": {
                **metadata,
                "source_type": "sql",
                "findings": quality_report(frame).get("findings", []),
            },
        }
    )
    return {
        "id": identifier,
        "filename": request.filename,
        "size_bytes": len(content),
        "metadata": metadata,
        "source_type": "sql",
    }


@app.post("/api/datasets", status_code=201)
async def upload_dataset(file: UploadFile = File(...)) -> dict:  # noqa: B008
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}:
        raise HTTPException(
            400, detail={"code": "unsupported_format", "message": "Unsupported dataset format."}
        )
    content = await file.read(settings.max_upload_size_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            413,
            detail={
                "code": "file_too_large",
                "message": "The uploaded file exceeds the configured size limit.",
            },
        )
    identifier = dataset_id(content)
    path = settings.upload_dir / f"{identifier}{suffix}"
    path.write_bytes(content)
    try:
        frame = load_dataframe(path)
    except DatumError:
        path.unlink(missing_ok=True)
        raise
    metadata = profile_dataframe(frame)
    try:
        from backend.app.data_processing import quality_report
        qreport = quality_report(frame)
        metadata = {**metadata, "findings": qreport.get("findings", [])}
    except Exception:
        pass
    record = {
        "id": identifier,
        "filename": file.filename or path.name,
        "path": str(path),
        "size_bytes": len(content),
        "created_at": utc_now(),
        "metadata": metadata,
    }
    store.add_dataset(record)
    return {
        "id": identifier,
        "filename": record["filename"],
        "size_bytes": len(content),
        "metadata": metadata,
    }


@app.get("/api/datasets")
def list_datasets() -> dict[str, Any]:
    return {"datasets": store.list_datasets()}


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise not_found("Dataset")
    return {key: value for key, value in dataset.items() if key != "path"}


def _apply_filters_frame(frame: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    result = frame.copy()
    for f in filters:
        if not f:
            continue
        col = f.get("column") or f.get("field")
        op = f.get("operator") or f.get("op") or "equals"
        val = f.get("value")
        if col not in result.columns:
            raise DatumError(f"Unknown column: {col}", "invalid_filter")
        series = result[col]
        if op == "equals":
            result = result[series.astype(str) == str(val)]
        elif op == "not_equals":
            result = result[series.astype(str) != str(val)]
        elif op == "contains":
            result = result[series.astype(str).str.contains(str(val), case=False, na=False)]
        elif op == "is_empty":
            result = result[series.isna() | (series.astype(str).str.strip() == "")]
        elif op == "is_not_empty":
            result = result[series.notna() & (series.astype(str).str.strip() != "")]
        elif op == "greater_than":
            result = result[pd.to_numeric(series, errors='coerce') > float(val)]
        elif op == "less_than":
            result = result[pd.to_numeric(series, errors='coerce') < float(val)]
        elif op == "between":
            if isinstance(val, (list, tuple)) and len(val) == 2:
                nums = pd.to_numeric(series, errors='coerce')
                result = result[(nums >= float(val[0])) & (nums <= float(val[1]))]
        elif op == "before":
            dates = pd.to_datetime(series, errors='coerce')
            result = result[dates < pd.to_datetime(val)]
        elif op == "after":
            dates = pd.to_datetime(series, errors='coerce')
            result = result[dates > pd.to_datetime(val)]
        elif op == "date_range":
            if isinstance(val, (list, tuple)) and len(val) == 2:
                dates = pd.to_datetime(series, errors='coerce')
                result = result[(dates >= pd.to_datetime(val[0])) & (dates <= pd.to_datetime(val[1]))]
        else:
            raise DatumError(f"Unsupported filter operator: {op}", "invalid_filter")
    return result


def _build_chart_from_config(frame: pd.DataFrame, req: ChartBuildRequest, dataset: dict) -> dict:
    chart_type = req.chart_type.lower()
    filtered = _apply_filters_frame(frame, req.filters)
    if filtered.empty:
        raise DatumError("No data remains after applying filters.", "empty_filtered_result")
    
    agg = (req.aggregation or "count").lower()
    
    if chart_type == "bar":
        cat = req.category or req.x_axis
        metric = req.value or req.y_axis or req.metric
        if not cat or cat not in filtered.columns:
            raise DatumError("Bar chart requires a valid category/X column.", "invalid_chart_config")
        if agg == "count":
            counts = filtered[cat].value_counts().head(req.limit or 20)
            data = [{"x": str(k), "y": int(v)} for k, v in counts.items()]
            return {
                "id": f"built-bar-{cat}",
                "type": "bar",
                "title": f"Count of rows by {cat}",
                "columns": [cat],
                "aggregation": "count",
                "x_label": cat,
                "y_label": "Count",
                "data": data,
            }
        else:
            if not metric or metric not in filtered.columns:
                raise DatumError("Metric required for non-count aggregation.", "invalid_chart_config")
            if not pd.api.types.is_numeric_dtype(filtered[metric]):
                raise DatumError("Metric must be numeric for aggregation.", "invalid_chart_config")
            grouped = filtered.groupby(cat)[metric]
            if agg == "sum":
                res = grouped.sum()
            elif agg == "mean" or agg == "average":
                res = grouped.mean()
            elif agg == "median":
                res = grouped.median()
            elif agg == "min":
                res = grouped.min()
            elif agg == "max":
                res = grouped.max()
            elif agg == "std" or agg == "standard_deviation":
                res = grouped.std()
            else:
                raise DatumError(f"Unsupported aggregation: {agg}", "invalid_chart_config")
            res = res.sort_values(ascending=False).head(req.limit or 20)
            data = [{"x": str(k), "y": float(v) if pd.notna(v) else 0} for k, v in res.items()]
            return {
                "id": f"built-bar-{cat}-{metric}",
                "type": "bar",
                "title": f"{agg.title()} of {metric} by {cat}",
                "columns": [cat, metric],
                "aggregation": f"{agg} of {metric}",
                "x_label": cat,
                "y_label": f"{agg.title()}({metric})",
                "data": data,
            }
    elif chart_type == "line":
        x = req.x_axis or req.category
        y = req.y_axis or req.metric or req.value
        if not x or x not in filtered.columns or not y or y not in filtered.columns:
            raise DatumError("Line chart requires X and Y columns.", "invalid_chart_config")
        dates = pd.to_datetime(filtered[x], errors='coerce') if any(t in str(filtered[x].dtype).lower() for t in ['date','time']) or filtered[x].astype(str).str.match(r'\d{4}-\d{2}').any() else None
        if dates is not None and dates.notna().mean() > 0.5:
            df = pd.DataFrame({"date": dates, "y": pd.to_numeric(filtered[y], errors='coerce')}).dropna()
            if req.date_granularity:
                g = req.date_granularity.lower()
                if g == "day": p = df["date"].dt.to_period("D")
                elif g == "week": p = df["date"].dt.to_period("W")
                elif g == "month": p = df["date"].dt.to_period("M")
                elif g == "quarter": p = df["date"].dt.to_period("Q")
                elif g == "year": p = df["date"].dt.to_period("Y")
                else: p = df["date"].dt.to_period("M")
                grp = df.assign(p=p).groupby("p")["y"].sum().sort_index()
                data = [{"x": str(k), "y": float(v)} for k, v in grp.items()]
            else:
                grp = df.set_index("date")["y"].resample("D").sum().dropna()
                data = [{"x": str(k.date()) if hasattr(k, 'date') else str(k), "y": float(v)} for k, v in grp.items()]
            return {
                "id": f"built-line-{x}-{y}",
                "type": "line",
                "title": f"{y} over time",
                "columns": [x, y],
                "aggregation": f"sum by {req.date_granularity or 'day'}",
                "x_label": x,
                "y_label": y,
                "data": data,
            }
        else:
            # categorical x
            if pd.api.types.is_numeric_dtype(filtered[y]):
                grp = filtered.groupby(x)[y].sum().sort_index()
                data = [{"x": str(k), "y": float(v)} for k, v in grp.items()]
            else:
                data = [{"x": str(k), "y": int(v)} for k, v in filtered[x].value_counts().items()]
            return {
                "id": f"built-line-{x}-{y}",
                "type": "line",
                "title": f"{y} by {x}",
                "columns": [x, y],
                "aggregation": "sum",
                "x_label": x,
                "y_label": y,
                "data": data,
            }
    elif chart_type == "scatter":
        x = req.x_axis
        y = req.y_axis
        if not x or not y or x not in filtered.columns or y not in filtered.columns:
            raise DatumError("Scatter plot requires two numeric columns.", "invalid_chart_config")
        df = filtered[[x,y]].apply(pd.to_numeric, errors='coerce').dropna()
        data = [{"x": float(r[x]), "y": float(r[y])} for _, r in df.head(5000).iterrows()]
        return {
            "id": f"built-scatter-{x}-{y}",
            "type": "scatter",
            "title": f"{y} vs {x}",
            "columns": [x, y],
            "aggregation": "row-level values",
            "x_label": x,
            "y_label": y,
            "data": data,
        }
    elif chart_type == "histogram":
        col = req.value or req.metric or req.x_axis
        if not col or col not in filtered.columns:
            raise DatumError("Histogram requires a numeric field.", "invalid_chart_config")
        values = pd.to_numeric(filtered[col], errors='coerce').dropna()
        bins = req.bin_count or 10
        counts, edges = np.histogram(values, bins=min(max(2,bins), 50))
        data = [{"x": f"{edges[i]:.3g}-{edges[i+1]:.3g}", "y": int(counts[i])} for i in range(len(counts))]
        return {
            "id": f"built-hist-{col}",
            "type": "histogram",
            "title": f"Distribution of {col}",
            "columns": [col],
            "aggregation": "count by numeric bin",
            "x_label": col,
            "y_label": "Count",
            "data": data,
        }
    elif chart_type == "box" or chart_type == "boxplot":
        col = req.value or req.metric
        if not col or col not in filtered.columns:
            raise DatumError("Box plot requires a numeric field.", "invalid_chart_config")
        values = pd.to_numeric(filtered[col], errors='coerce').dropna()
        quartiles = values.quantile([0.25,0.5,0.75])
        return {
            "id": f"built-box-{col}",
            "type": "box",
            "title": f"Spread of {col}",
            "columns": [col],
            "aggregation": "distribution summary",
            "summary": {"min": float(values.min()), "q1": float(quartiles.iloc[0]), "median": float(quartiles.iloc[1]), "q3": float(quartiles.iloc[2]), "max": float(values.max())},
            "data": [{"y": float(v)} for v in values.head(5000)],
        }
    elif chart_type == "heatmap" or chart_type == "correlation_heatmap" or chart_type == "correlation":
        cols = req.columns or []
        if not cols:
            # pick numeric cols
            cols = list(filtered.select_dtypes(include='number').columns)[:10]
        if len(cols) < 2:
            raise DatumError("Correlation heatmap requires multiple numeric columns.", "invalid_chart_config")
        numeric = filtered[cols].apply(pd.to_numeric, errors='coerce')
        corr = numeric.corr(method=req.correlation_method or 'pearson')
        data = [{"x": str(r), "y": str(c), "value": float(corr.loc[r,c]) if pd.notna(corr.loc[r,c]) else 0} for r in corr.index for c in corr.columns]
        return {
            "id": "built-heatmap",
            "type": "heatmap",
            "title": "Numeric correlation matrix",
            "columns": list(cols),
            "aggregation": f"{req.correlation_method or 'pearson'} correlation",
            "data": data,
        }
    elif chart_type == "pie" or chart_type == "donut":
        cat = req.category or req.x_axis
        if not cat or cat not in filtered.columns:
            raise DatumError("Pie chart requires a category column.", "invalid_chart_config")
        counts = filtered[cat].value_counts().head(req.limit or 10)
        data = [{"x": str(k), "y": int(v)} for k, v in counts.items()]
        return {
            "id": f"built-pie-{cat}",
            "type": "bar",  # frontend can render as pie if needed, use bar structure
            "title": f"Distribution of {cat}",
            "columns": [cat],
            "aggregation": "count",
            "data": data,
        }
    else:
        raise DatumError(f"Unsupported chart type: {chart_type}. Supported: bar, line, scatter, histogram, box, correlation heatmap, pie/donut", "invalid_chart_config")


@app.post("/api/datasets/{dataset_id}/workflows", status_code=202)
def create_workflow(
    dataset_id: str, request: WorkflowRequest, background_tasks: BackgroundTasks
) -> dict:
    if store.get_dataset(dataset_id) is None:
        raise not_found("Dataset")
    run_id = str(uuid.uuid4())
    now = utc_now()
    store.add_run(
        {
            "id": run_id,
            "dataset_id": dataset_id,
            "mode": request.mode,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
        }
    )
    background_tasks.add_task(
        _execute_workflow, run_id, dataset_id, request.mode, request.preparation
    )
    return {"run_id": run_id, "dataset_id": dataset_id, "mode": request.mode, "status": "queued"}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise not_found("Run")
    return run


@app.get("/api/runs/{run_id}/report")
def get_report(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise not_found("Run")
    if run["status"] not in {"completed", "rejected"}:
        raise HTTPException(
            409, detail={"code": "run_not_completed", "message": "The report is not available yet."}
        )
    return run["result"]


@app.get("/api/runs/{run_id}/preparation-plan")
def get_preparation_plan(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise not_found("Run")
    return (run["result"] or {}).get("preparation_plan", {"status": "not_available"})


@app.post("/api/runs/{run_id}/approval")
def approve_workflow(
    run_id: str, request: ApprovalRequest, background_tasks: BackgroundTasks
) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise not_found("Run")
    if run["status"] != "needs_approval":
        raise HTTPException(
            409,
            detail={
                "code": "approval_not_pending",
                "message": "This run is not waiting for approval.",
            },
        )
    store.update_run(run_id, "running", approval_status=request.decision)
    background_tasks.add_task(_resume_workflow, run_id, run["mode"], request.decision)
    return {"run_id": run_id, "status": "running", "approval": request.decision}


@app.get("/api/runs/{run_id}/statistics")
def get_statistics(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise not_found("Run")
    return (run["result"] or {}).get("statistics", {})


class ChartCustomizeRequest(BaseModel):
    selections: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    context: dict[str, Any] = Field(default_factory=dict)


class ChartBuildRequest(BaseModel):
    dataset_id: str | None = None
    chart_type: str
    x_axis: str | None = None
    y_axis: str | None = None
    category: str | None = None
    value: str | None = None
    metric: str | None = None
    columns: list[str] | None = None
    aggregation: str | None = "count"
    group_by: str | None = None
    series: str | None = None
    filters: list[dict[str, Any]] = Field(default_factory=list)
    limit: int | None = None
    sort: str | None = None
    date_granularity: str | None = None
    bin_count: int | None = 10
    correlation_method: str | None = "pearson"


@app.post("/api/runs/{run_id}/analysis/chat")
def analysis_chat(run_id: str, request: AnalysisChatRequest) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise not_found("Run")
    if run["mode"] != "analyze_only" or run["status"] != "completed":
        raise HTTPException(
            409,
            detail={
                "code": "analysis_unavailable",
                "message": "Conversational analysis is available only for completed Analyze runs.",
            },
        )
    dataset = store.get_dataset(run["dataset_id"])
    if dataset is None:
        raise not_found("Dataset")
    frame = load_dataframe(Path(dataset["path"]))
    try:
        result = answer_question(
            run_id,
            run["dataset_id"],
            frame,
            request.question,
            request.context,
            settings.report_dir.parent / "charts",
        )
    except DatumError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        logger.exception("Deterministic analysis failed for run %s: %s", run_id, exc)
        raise HTTPException(
            500,
            detail={
                "code": "analysis_failed",
                "message": "The requested analysis could not be calculated from this dataset.",
            },
        ) from exc
    if result.get("chart"):
        try:
            result["chart"] = chart_utils.add_dataset_context(
                [result["chart"]],
                dataset["filename"],
                len(frame),
                len(frame.columns),
            )[0]
            result["chart"]["dataset_context"]["filters"] = result.get("context", {}).get("filters", {})
        except Exception:
            pass
    result["provider"] = {
        "status": "not_required",
        "message": "Calculated deterministically from the dataset.",
    }
    return result


@app.post("/api/runs/{run_id}/charts/customize")
def customize_charts(run_id: str, request: ChartCustomizeRequest) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise not_found("Run")
    dataset = store.get_dataset(run["dataset_id"])
    if dataset is None:
        raise not_found("Dataset")
    frame = load_dataframe(Path(dataset["path"]))
    # Prepare reports use the prepared output for report-specific statistics.
    working_path = (run.get("result") or {}).get("prepared_path")
    if working_path and Path(working_path).exists():
        try:
            frame = load_dataframe(Path(working_path))
        except Exception:
            pass
    charts_out = []
    from backend.app.data_processing import _eda_charts
    # generate base
    numeric = frame.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    base_charts = _eda_charts(frame, numeric)
    # filter by selections if provided
    selected = request.selections
    if selected:
        allowed_ids = set()
        for sel in selected:
            cid = sel.get("id")
            if cid:
                allowed_ids.add(cid)
            # also allow by type/columns combinations
        if allowed_ids:
            charts_out = [c for c in base_charts if c.get("id") in allowed_ids]
        else:
            charts_out = base_charts[:20]
    else:
        charts_out = base_charts
    try:
        charts_out = chart_utils.generate_chart_images(
            run_id, frame, charts_out, settings.report_dir.parent / "charts"
        )
    except Exception:
        pass
    return {
        "charts": chart_utils.add_dataset_context(
            charts_out,
            dataset["filename"],
            len(frame),
            len(frame.columns),
        )
    }


@app.get("/api/runs/{run_id}/download")
def download_prepared(run_id: str):
    run = store.get_run(run_id)
    if run is None:
        raise not_found("Run")
    output = settings.prepared_dir / f"{run_id}.csv"
    if not output.exists():
        raise HTTPException(
            404,
            detail={
                "code": "no_prepared_dataset",
                "message": "This run did not create a prepared dataset.",
            },
        )
    return FileResponse(output, filename=output.name, media_type="text/csv")


@app.get("/api/runs/{run_id}/prepared-preview")
def prepared_preview(
    run_id: str,
    offset: int = 0,
    limit: int = 10,
    column_offset: int = 0,
    column_limit: int = 8,
):
    run = store.get_run(run_id)
    if run is None:
        raise not_found("Run")
    if run["mode"] != "prepare_for_ml" or run["status"] != "completed":
        raise HTTPException(
            409,
            detail={
                "code": "prepared_dataset_unavailable",
                "message": (
                    "A prepared dataset is available only after a completed "
                    "Prepare for ML run."
                ),
            },
        )
    if offset < 0 or offset > 1_000_000 or limit < 1 or limit > 100:
        raise HTTPException(
            422,
            detail={
                "code": "invalid_pagination",
                "message": "Use offset >= 0 and limit between 1 and 100.",
            },
        )
    if column_offset < 0 or column_offset > 100_000 or column_limit < 1 or column_limit > 50:
        raise HTTPException(
            422,
            detail={
                "code": "invalid_columns",
                "message": "Use column_offset >= 0 and column_limit between 1 and 50.",
            },
        )
    output = settings.prepared_dir / f"{run_id}.csv"
    if not output.is_file():
        raise HTTPException(
            404,
            detail={
                "code": "no_prepared_dataset",
                "message": "The prepared dataset for this run is unavailable.",
            },
        )
    dataset = store.get_dataset(run["dataset_id"])
    if dataset is None:
        raise not_found("Dataset")
    source = load_dataframe(Path(dataset["path"]))
    prepared = load_dataframe(output)
    all_columns = [str(column) for column in prepared.columns]
    visible_columns = all_columns[column_offset : column_offset + column_limit]
    rows = json.loads(
        prepared.iloc[offset : offset + limit][visible_columns].to_json(
            orient="records", date_format="iso"
        )
    )
    source_profile = profile_dataframe(source)
    prepared_profile = profile_dataframe(prepared)
    source_quality = quality_report(source)
    prepared_quality = quality_report(prepared)
    source_numeric = source.select_dtypes(include="number")
    prepared_numeric = prepared.select_dtypes(include="number")
    source_non_finite = int(
        (~np.isfinite(source_numeric.to_numpy(dtype=float))).sum()
    ) if not source_numeric.empty else 0
    prepared_non_finite = int(
        (~np.isfinite(prepared_numeric.to_numpy(dtype=float))).sum()
    ) if not prepared_numeric.empty else 0
    transformations = (run["result"] or {}).get("transformations", [])
    generated_by = {
        column: [
            item.get("operation", "transformation")
            for item in transformations
            if column in (item.get("parameters", {}).get("output_columns", []) or [])
            or column == item.get("parameters", {}).get("output_column")
        ]
        for column in all_columns
        if column not in source.columns
    }
    return {
        "run_id": run_id,
        "filename": output.name,
        "offset": offset,
        "limit": limit,
        "column_offset": column_offset,
        "column_limit": column_limit,
        "total_rows": int(len(prepared)),
        "total_columns": len(all_columns),
        "columns": [
            {
                "name": str(column["name"]),
                "dtype": str(column["dtype"]),
                "new": str(column["name"]) not in source.columns,
            }
            for column in prepared_profile["columns"]
            if str(column["name"]) in visible_columns
        ],
        "all_columns": all_columns,
        "rows": rows,
        "new_columns": [column for column in all_columns if column not in source.columns],
        "generated_by": generated_by,
        "transformations": transformations,
        "verification": (run["result"] or {}).get("verification", {}),
        "quality_warnings": [
            finding.get("message", "Quality warning")
            for finding in prepared_quality.get("findings", [])
            if finding.get("severity") in {"warning", "error"}
        ],
        "quality_scores": {
            "original": source_quality.get("quality_score"),
            "prepared": prepared_quality.get("quality_score"),
        },
        "non_finite_values": {
            "original": source_non_finite,
            "prepared": prepared_non_finite,
        },
        "comparison": {
            "original": _comparison_metrics(source, source_profile, source_quality),
            "prepared": _comparison_metrics(prepared, prepared_profile, prepared_quality),
        },
    }


@app.get("/api/runs/{run_id}/notebook")
def download_notebook(run_id: str):
    run = store.get_run(run_id)
    if run is None:
        raise not_found("Run")
    notebook = (run["result"] or {}).get("notebook", {})
    output_name = notebook.get("filename")
    if not output_name:
        raise HTTPException(
            404,
            detail={
                "code": "no_notebook",
                "message": "A notebook is not available for this run.",
            },
        )
    output = settings.report_dir / output_name
    if not output.is_file() or output.parent != settings.report_dir:
        raise HTTPException(
            404,
            detail={
                "code": "notebook_unavailable",
                "message": "The notebook artifact is unavailable.",
            },
        )
    return FileResponse(
        output,
        filename=output.name,
        media_type="application/x-ipynb+json",
    )


def _comparison_metrics(frame, profile: dict, quality: dict) -> dict[str, int | float]:
    numeric_columns = int(len(frame.select_dtypes(include="number").columns))
    duplicate_rows = int(frame.duplicated().sum())
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_cells": int(profile.get("missing_cell_count", 0)),
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_columns,
        "missing_cell_percentage": float(profile.get("missing_cell_percentage", 0)),
        "duplicate_percentage": round(duplicate_rows / len(frame) * 100, 3) if len(frame) else 0,
    }


def _execute_workflow(run_id: str, dataset_id_value: str, mode: str, preparation: dict) -> None:
    try:
        dataset = store.get_dataset(dataset_id_value)
        if dataset is None:
            raise DatumError("Dataset was deleted before the workflow started.", "not_found")
        store.update_run(run_id, "running", stage="validate")
        state = {
            "run_id": run_id,
            "dataset_id": dataset_id_value,
            "source_path": dataset["path"],
            "dataset_name": dataset["filename"],
            "mode": mode,
            "preparation_config": preparation,
        }
        compiled = graphs.prepare_for_ml if mode == "prepare_for_ml" else graphs.analyze_only
        result = compiled.invoke(state, config=graphs.config(run_id))
        if "__interrupt__" in result:
            interrupt_value = result["__interrupt__"][0].value
            plan = interrupt_value.get("plan", {})
            store.update_run(
                run_id,
                "needs_approval",
                result={
                    "preparation_plan": plan,
                    "mode": mode,
                    "run_id": run_id,
                    "dataset_id": dataset_id_value,
                },
                stage="approval",
                approval_status="pending",
            )
            return
        _complete_graph_run(run_id, result)
    except (
        DatumError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        OverflowError,
    ) as exc:
        current = store.get_run(run_id) or {}
        stage = current.get("stage", "failed")
        stage_status = {**current.get("stage_status", {}), stage: "failed"}
        store.update_run(
            run_id,
            "failed",
            error=str(exc),
            stage=stage,
            stage_status=stage_status,
        )


def _resume_workflow(run_id: str, mode: str, decision: str) -> None:
    try:
        result = resume_graph(graphs, run_id, mode, decision)
        if "__interrupt__" in result:
            raise RuntimeError("Workflow requested an unexpected second approval.")
        _complete_graph_run(run_id, result)
    except (
        DatumError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        OverflowError,
    ) as exc:
        current = store.get_run(run_id) or {}
        stage = current.get("stage", "failed")
        stage_status = {**current.get("stage_status", {}), stage: "failed"}
        store.update_run(
            run_id,
            "failed",
            error=str(exc),
            stage=stage,
            stage_status=stage_status,
        )


def _complete_graph_run(run_id: str, state: dict) -> None:
    report = state.get("report", {})
    report.pop("prepared_path", None)
    notebook_name = f"{run_id}.ipynb"
    generate_notebook(
        output_path=settings.report_dir / notebook_name,
        source_path=Path(state["source_path"]),
        dataset_id=state["dataset_id"],
        mode=state["mode"],
        profile=state.get("profile", {}),
        quality=state.get("quality", {}),
        statistics=state.get("statistics", {}),
        transformations=state.get("transformations", []),
        verification=state.get("verification", {}),
        preparation_plan=state.get("preparation_plan"),
        approval=state.get("approval"),
    )
    report["notebook"] = {
        "available": True,
        "filename": notebook_name,
        "download": f"/api/runs/{run_id}/notebook",
    }
    report_path = settings.report_dir / f"{run_id}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    store.update_run(
        run_id,
        "rejected" if state.get("approval") == "rejected" else "completed",
        report,
        stage=state.get("stage", "final"),
        stage_status=state.get("stage_status", {}),
        approval_status=state.get("approval"),
    )
# Additional endpoints for enhanced chart building and dataset schema
@app.get("/api/datasets/{dataset_id}/schema")
def dataset_schema(dataset_id: str) -> dict:
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise not_found("Dataset")
    frame = load_dataframe(Path(dataset["path"]))
    columns = []
    for col in frame.columns:
        series = frame[col]
        col_info = {
            "name": str(col),
            "dtype": str(series.dtype),
            "nullable": bool(series.isna().any()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        try:
            is_num = pd.api.types.is_numeric_dtype(series)
            dtype_str = str(series.dtype)
            if is_num or dtype_str.startswith(("int", "float", "complex", "uint")):
                col_info["kind"] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(series) or str(series.dtype).startswith("datetime"):
                col_info["kind"] = "datetime"
            else:
                col_info["kind"] = "datetime" if _is_likely_date_column(series) else "categorical"
            if col_info["kind"] == "categorical":
                try:
                    col_info["sample_values"] = [str(v) for v in series.dropna().unique()[:20]]
                except Exception:
                    pass
        except Exception:
            col_info["kind"] = "categorical"
        columns.append(col_info)
    return {
        "dataset_id": dataset_id,
        "filename": dataset["filename"],
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "schema": columns,
    }


@app.get("/api/datasets/{dataset_id}/preview")
def preview_dataset(
    dataset_id: str,
    limit: int = 20,
    offset: int = 0,
    search: str | None = None,
    sort_by: str | None = None,
    sort_order: str = "asc",
) -> dict:
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise not_found("Dataset")
    frame = load_dataframe(Path(dataset["path"]))
    if search:
        mask = frame.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        frame = frame[mask]
    total_rows = int(len(frame))
    if sort_by:
        if sort_by not in frame.columns:
            raise not_found(f"Sort column '{sort_by}' not found")
        ascending = sort_order.lower() != "desc"
        frame = frame.sort_values(by=sort_by, ascending=ascending, kind="mergesort")
    page = frame.iloc[offset:offset + limit]
    return {
        "dataset_id": dataset_id,
        "columns": list(frame.columns),
        "rows": json.loads(page.to_json(orient="records")),
        "total_rows": total_rows,
        "returned_rows": int(len(page)),
        "offset": int(offset),
        "limit": int(limit),
    }


@app.get("/api/datasets/{dataset_id}/values")
def dataset_values(dataset_id: str, column: str, limit: int = 50) -> dict:
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise not_found("Dataset")
    frame = load_dataframe(Path(dataset["path"]))
    if column not in frame.columns:
        raise not_found(f"Column '{column}' not found")
    series = frame[column].dropna()
    values = [str(v) for v in series.unique()[:limit]]
    return {
        "dataset_id": dataset_id,
        "column": column,
        "values": values,
        "total_unique": int(frame[column].nunique(dropna=True)),
    }


def _apply_filters_frame(frame: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    result = frame.copy()
    for f in filters or []:
        if not f:
            continue
        col = f.get("column") or f.get("field") or f.get("name")
        op = f.get("operator") or f.get("op") or "equals"
        val = f.get("value")
        if col not in result.columns:
            raise DatumError(f"Unknown column: {col}", "invalid_filter")
        series = result[col]
        try:
            if op == "equals":
                result = result[series.astype(str) == str(val)]
            elif op == "not_equals":
                result = result[series.astype(str) != str(val)]
            elif op == "contains":
                result = result[series.astype(str).str.contains(str(val), case=False, na=False)]
            elif op == "is_empty":
                result = result[series.isna() | (series.astype(str).str.strip() == "")]
            elif op == "is_not_empty":
                result = result[series.notna() & (series.astype(str).str.strip() != "")]
            elif op == "greater_than":
                result = result[pd.to_numeric(series, errors='coerce') > float(val)]
            elif op == "less_than":
                result = result[pd.to_numeric(series, errors='coerce') < float(val)]
            elif op == "between":
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    nums = pd.to_numeric(series, errors='coerce')
                    result = result[(nums >= float(val[0])) & (nums <= float(val[1]))]
            elif op == "before":
                dates = pd.to_datetime(series, errors='coerce')
                result = result[dates < pd.to_datetime(val)]
            elif op == "after":
                dates = pd.to_datetime(series, errors='coerce')
                result = result[dates > pd.to_datetime(val)]
            elif op == "date_range":
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    dates = pd.to_datetime(series, errors='coerce')
                    result = result[(dates >= pd.to_datetime(val[0])) & (dates <= pd.to_datetime(val[1]))]
            else:
                raise DatumError(f"Unsupported filter operator: {op}", "invalid_filter")
        except DatumError:
            raise
        except Exception as e:
            raise DatumError(f"Filter application failed: {e}", "invalid_filter")
    return result

@app.post('/api/charts/build')
def build_chart(request: ChartBuildRequest) -> dict:
    dataset_id = request.dataset_id
    if not dataset_id:
        datasets = store.list_datasets()
        if not datasets:
            raise not_found('Dataset')
        if isinstance(datasets, list):
            dataset_id = datasets[-1]['id']
        elif isinstance(datasets, dict) and datasets.get('datasets'):
            dataset_id = datasets['datasets'][-1]['id']
        else:
            raise not_found('Dataset')
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise not_found('Dataset')
    frame = load_dataframe(Path(dataset['path']))
    try:
        chart = _build_chart_from_config(frame, request, dataset)
    except DatumError:
        raise
    except Exception as e:
        raise HTTPException(400, detail={'code': 'chart_build_failed', 'message': str(e)})
    run_id = f'build-{dataset_id[:8]}'
    try:
        charts_out = chart_utils.generate_chart_images(
            run_id, frame, [chart], settings.report_dir.parent / 'charts'
        )
        chart = charts_out[0]
    except Exception:
        pass
    try:
        chart = chart_utils.add_dataset_context(
            [chart], dataset['filename'], len(frame), len(frame.columns)
        )[0]
        ctx_filters = {}
        for f in (request.filters or []):
            if f and f.get('column'):
                ctx_filters[str(f.get('column'))] = str(f.get('value'))
        chart['dataset_context']['filters'] = ctx_filters
    except Exception:
        pass
    return {'chart': chart, 'dataset': {'id': dataset_id, 'filename': dataset['filename'], 'rows': len(frame), 'columns': len(frame.columns)}}
