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
            if pd.api.types.is_numeric_dtype(series):
                col_info["kind"] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(series) or str(series.dtype).startswith("datetime"):
                col_info["kind"] = "datetime"
            else:
                try:
                    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
                    if parsed.notna().mean() > 0.8:
                        col_info["kind"] = "datetime"
                    else:
                        col_info["kind"] = "categorical"
                except Exception:
                    col_info["kind"] = "categorical"
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


@app.get("/api/datasets/{dataset_id}/values")
def dataset_values(dataset_id: str, column: str, limit: int = 50) -> dict:
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise not_found("Dataset")
    frame = load_dataframe(Path(dataset["path"]))
    if column not in frame.columns:
        raise HTTPException(400, detail={"code": "column_not_found", "message": f"Column {column} not found"})
    try:
        values = frame[column].dropna().value_counts().head(limit).index.tolist()
        return {"values": [str(v) for v in values]}
    except Exception as e:
        raise HTTPException(400, detail={"code": "error", "message": str(e)})


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
