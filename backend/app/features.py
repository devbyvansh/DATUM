from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from backend.app.errors import DatumError


def apply_feature_operations(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = frame.copy()
    records: list[dict[str, Any]] = []
    result, indicator_records = _add_missing_indicators(result, config)
    records.extend(indicator_records)
    result, imputation_records = _impute_values(result, config)
    records.extend(imputation_records)
    result, text_records = _text_features(result, config)
    records.extend(text_records)
    result, date_records = _date_features(result, config)
    records.extend(date_records)
    result, encoded_records = _one_hot_encode(result, config)
    records.extend(encoded_records)
    result, scaled_records = _scale_numeric(result, config)
    records.extend(scaled_records)
    result, arithmetic_records = _arithmetic_features(result, config)
    records.extend(arithmetic_records)
    _validate_output(result)
    return result, records


def _impute_values(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = frame.copy()
    records = []
    for operation, method in (
        ("impute_mean", "mean"),
        ("impute_constant", "constant"),
        ("impute_forward_fill", "ffill"),
        ("impute_backward_fill", "bfill"),
    ):
        for item in config.get(operation, []):
            column = item if isinstance(item, str) else item.get("column")
            if column not in result:
                raise DatumError(f"Column not found: {column}.", "invalid_feature_operation")
            before = int(result[column].isna().sum())
            if method == "mean":
                if not pd.api.types.is_numeric_dtype(result[column]):
                    raise DatumError(
                        f"Mean imputation requires a numeric column: {column}.",
                        "invalid_feature_operation",
                    )
                values = pd.to_numeric(result[column], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                if values.notna().sum() == 0:
                    raise DatumError(
                        f"Cannot impute {column}: it has no finite numeric values.",
                        "invalid_feature_operation",
                    )
                result[column] = values.fillna(values.mean())
            elif method == "constant":
                value = item.get("value") if isinstance(item, dict) else 0
                result[column] = result[column].fillna(value)
            else:
                result[column] = getattr(result[column], method)()
            records.append(_record(operation, [column], {"filled_count": before}))
    return result, records


def _add_missing_indicators(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = frame.copy()
    records = []
    for column in config.get("missing_indicators", []):
        if column not in result:
            raise DatumError(f"Column not found: {column}.", "invalid_feature_operation")
        name = f"{column}__missing"
        _ensure_new_column(result, name)
        result[name] = result[column].isna().astype("int8")
        records.append(_record("missing_indicator", [column], {"output_column": name}))
    return result, records


def _text_features(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = frame.copy()
    records = []
    for item in config.get("text_features", []):
        column = item if isinstance(item, str) else item.get("column")
        if column not in result:
            raise DatumError(f"Column not found: {column}.", "invalid_feature_operation")
        text = result[column].fillna("").astype(str).str.strip()
        outputs = {
            f"{column}__text_length": text.str.len(),
            f"{column}__word_count": text.str.split().str.len(),
        }
        for name, values in outputs.items():
            _ensure_new_column(result, name)
            result[name] = values
        records.append(_record("text_features", [column], {"output_columns": list(outputs)}))
    return result, records


def _date_features(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = frame.copy()
    records = []
    components = ("year", "quarter", "month", "day", "dayofweek", "hour")
    for item in config.get("date_features", []):
        column = item if isinstance(item, str) else item.get("column")
        requested = item.get("components", components) if isinstance(item, dict) else components
        if column not in result:
            raise DatumError(f"Column not found: {column}.", "invalid_feature_operation")
        dates = pd.to_datetime(result[column], errors="coerce", format="mixed")
        values = {
            "year": dates.dt.year,
            "quarter": dates.dt.quarter,
            "month": dates.dt.month,
            "day": dates.dt.day,
            "dayofweek": dates.dt.dayofweek,
            "hour": dates.dt.hour,
        }
        output_columns = []
        for component in requested:
            if component not in values:
                raise DatumError(f"Unsupported date component: {component}.", "invalid_feature_operation")
            name = f"{column}__{component}"
            _ensure_new_column(result, name)
            result[name] = values[component].fillna(-1)
            output_columns.append(name)
        records.append(_record("date_features", [column], {"output_columns": output_columns}))
    return result, records


def _one_hot_encode(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = frame.copy()
    records = []
    for column in config.get("one_hot_encode", []):
        if column not in result:
            raise DatumError(f"Column not found: {column}.", "invalid_feature_operation")
        values = result[column].fillna("__missing__").astype(str)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype="int8")
        encoded = encoder.fit_transform(values.to_frame())
        names = [f"{column}__{category}" for category in encoder.categories_[0]]
        for name, values_for_column in zip(names, encoded.T):
            _ensure_new_column(result, name)
            result[name] = values_for_column
        records.append(_record("one_hot_encode", [column], {"output_columns": names}))
    return result, records


def _scale_numeric(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = frame.copy()
    records = []
    methods = {
        "standard_scale": StandardScaler,
        "minmax_scale": MinMaxScaler,
        "robust_scale": RobustScaler,
    }
    for operation, scaler_type in methods.items():
        for column in config.get(operation, []):
            if column not in result or not pd.api.types.is_numeric_dtype(result[column]):
                raise DatumError(f"Scaling requires a numeric column: {column}.", "invalid_feature_operation")
            values = result[column].to_numpy(dtype=float).reshape(-1, 1)
            if not np.isfinite(values[~np.isnan(values)]).all():
                raise DatumError(f"Column contains invalid numeric values: {column}.", "invalid_feature_operation")
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                raise DatumError(
                    f"Scaling cannot be applied to {column}: it has no finite values.",
                    "invalid_feature_operation",
                )
            median = float(np.median(finite_values))
            filled = np.nan_to_num(values, nan=median)
            if np.nanmax(filled) == np.nanmin(filled):
                raise DatumError(
                    f"Scaling was skipped for {column}: the column has no variation.",
                    "constant_feature",
                )
            result[column] = scaler_type().fit_transform(filled).ravel()
            records.append(_record(operation, [column], {"fitted_on_rows": len(result)}))
    return result, records


def _arithmetic_features(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = frame.copy()
    records = []
    for item in config.get("ratios", []):
        numerator, denominator, output = item["numerator"], item["denominator"], item["output"]
        if numerator not in result or denominator not in result:
            raise DatumError("Ratio columns were not found.", "invalid_feature_operation")
        denominator_values = pd.to_numeric(result[denominator], errors="coerce")
        zero_count = int((denominator_values == 0).sum())
        if zero_count:
            records.append(
                _record(
                    "ratio",
                    [numerator, denominator],
                    {
                        "output_column": output,
                        "status": "skipped",
                        "reason": f"{zero_count} denominator values were zero.",
                    },
                )
            )
            continue
        _ensure_new_column(result, output)
        result[output] = result[numerator] / denominator_values
        records.append(_record("ratio", [numerator, denominator], {"output_column": output}))
    return result, records


def _ensure_new_column(frame: pd.DataFrame, name: str) -> None:
    if name in frame.columns:
        raise DatumError(f"Output column already exists: {name}.", "feature_collision")


def _record(operation: str, columns: list[str], parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": operation,
        "columns": columns,
        "parameters": parameters,
    }


def _validate_output(frame: pd.DataFrame) -> None:
    numeric = frame.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise DatumError("A feature operation produced invalid numeric values.", "invalid_feature_output")
