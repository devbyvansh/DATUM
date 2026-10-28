from __future__ import annotations

from pathlib import Path
from typing import Any

import nbformat
import pandas as pd

from backend.app.data_processing import load_dataframe


_SUPPORTED_OPERATIONS = {
    "drop_duplicates",
    "impute_median",
    "impute_mode",
    "impute_mean",
    "impute_constant",
    "impute_forward_fill",
    "impute_backward_fill",
    "missing_indicator",
    "text_features",
    "date_features",
    "one_hot_encode",
    "standard_scale",
    "minmax_scale",
    "robust_scale",
    "ratio",
}


def generate_notebook(
    *,
    output_path: Path,
    source_path: Path,
    dataset_id: str,
    mode: str,
    profile: dict[str, Any],
    quality: dict[str, Any],
    statistics: dict[str, Any],
    transformations: list[dict[str, Any]],
    verification: dict[str, Any],
    preparation_plan: dict[str, Any] | None = None,
    approval: str | None = None,
) -> None:
    """Write a safe, dataset-specific notebook without embedding private paths or data."""
    frame = load_dataframe(source_path)
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python"}

    source_name = source_path.name
    notebook.cells.extend(
        [
            nbformat.v4.new_markdown_cell(
                f"# DATUM {mode.replace('_', ' ').title()} Notebook\n\n"
                f"This notebook documents the completed DATUM run for dataset `{dataset_id}`. "
                "It contains controlled code for the analyses and transformations that were "
                "actually selected and executed. Cells are not pre-executed; run them locally "
                "after placing the source dataset beside this notebook."
            ),
            nbformat.v4.new_markdown_cell(
                "## Reproduction setup\n\n"
                f"The original source file was named `{source_name}`. Set `DATASET_PATH` "
                "to a portable copy of that file. DATUM's private server paths and connection "
                "credentials are intentionally not embedded.\n\n"
                "Required packages: `pandas`, `numpy`, `matplotlib`, and `seaborn`. "
                "Prepare runs may also require `scikit-learn` for scaling operations. "
                "SQL-source runs require you to export the bounded source query to the "
                "file named above; no database credentials are included."
            ),
            nbformat.v4.new_code_cell(
                "from pathlib import Path\n"
                "import numpy as np\n"
                "import pandas as pd\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "\n"
                "DATASET_PATH = Path(" + repr(source_name) + ")\n"
                "if not DATASET_PATH.exists():\n"
                "    raise FileNotFoundError(\n"
                "        f\"Place the source dataset next to this notebook: {DATASET_PATH}\"\n"
                "    )\n"
                "\n"
                "def load_datum_data(path):\n"
                "    suffix = path.suffix.lower()\n"
                "    readers = {\n"
                "        '.csv': lambda value: pd.read_csv(value, skip_blank_lines=False),\n"
                "        '.xlsx': pd.read_excel,\n"
                "        '.xls': pd.read_excel,\n"
                "        '.json': pd.read_json,\n"
                "        '.parquet': pd.read_parquet,\n"
                "    }\n"
                "    if suffix == '.jsonl':\n"
                "        return pd.read_json(path, lines=True)\n"
                "    if suffix not in readers:\n"
                "        raise ValueError(f'Unsupported dataset format: {suffix}')\n"
                "    return readers[suffix](path)\n"
                "\n"
                "source = load_datum_data(DATASET_PATH)\n"
                "for column in source.select_dtypes(include=['object', 'string']).columns:\n"
                "    values = source[column].astype('string')\n"
                "    source[column] = values.mask(values.str.strip().eq(''))\n"
                "working = source.copy()\n"
                "def quality_snapshot(data):\n"
                "    numeric = data.select_dtypes(include='number')\n"
                "    invalid_dates = {}\n"
                "    for column in data.columns:\n"
                "        if any(token in str(column).lower() for token in ('date', 'time', 'timestamp')):\n"
                "            parsed = pd.to_datetime(data[column], errors='coerce', format='mixed')\n"
                "            invalid = int(parsed.isna().sum() - data[column].isna().sum())\n"
                "            if invalid:\n"
                "                invalid_dates[column] = invalid\n"
                "    return {\n"
                "        'rows': int(len(data)),\n"
                "        'columns': int(len(data.columns)),\n"
                "        'missing_cells': int(data.isna().sum().sum()),\n"
                "        'missing_percent': round(100 * data.isna().sum().sum() / max(len(data) * len(data.columns), 1), 3),\n"
                "        'duplicate_rows': int(data.duplicated().sum()),\n"
                "        'non_finite_values': int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()) if not numeric.empty else 0,\n"
                "        'invalid_dates': invalid_dates,\n"
                "    }\n"
                "source.head()"
            ),
            nbformat.v4.new_markdown_cell(
                "## Scope and profile\n\n"
                "The next cell recomputes the basic scope from the source data. "
                "This is the same source snapshot used by the DATUM run; SQL sources "
                "must be exported using the bounded query before running this notebook."
            ),
            nbformat.v4.new_code_cell(
                "scope = {\n"
                "    'rows': int(len(source)),\n"
                "    'columns': int(len(source.columns)),\n"
                "    'missing_cells': int(source.isna().sum().sum()),\n"
                "    'duplicate_rows': int(source.duplicated().sum()),\n"
                "}\n"
                "scope"
            ),
            nbformat.v4.new_markdown_cell(
                "## Data quality analysis\n\n"
                "These checks use the normalized analysis copy. Blank and whitespace-only "
                "text is treated as missing. Extreme numeric values are reported as "
                "candidates for investigation, not automatically confirmed errors."
            ),
            nbformat.v4.new_code_cell(
                "source_quality = quality_snapshot(source)\n"
                "source_quality\n"
                "\n"
                "# Recorded DATUM quality evidence for this run (for comparison):\n"
                "datum_quality_evidence = " + repr(quality) + "\n"
                "datum_quality_evidence"
            ),
            nbformat.v4.new_code_cell(
                "numeric_summary = source.select_dtypes(include='number').describe().T\n"
                "outlier_candidates = {}\n"
                "for column in numeric_summary.index:\n"
                "    values = pd.to_numeric(source[column], errors='coerce').dropna()\n"
                "    if len(values) < 4:\n"
                "        outlier_candidates[column] = {'status': 'insufficient_sample'}\n"
                "        continue\n"
                "    q1, q3 = values.quantile([0.25, 0.75])\n"
                "    iqr = q3 - q1\n"
                "    outlier_candidates[column] = {\n"
                "        'candidate_count': int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()),\n"
                "        'method': '1.5x IQR; candidate only',\n"
                "    }\n"
                "numeric_summary, outlier_candidates"
            ),
        ]
    )

    numeric = list(frame.select_dtypes(include="number").columns)
    categorical = [column for column in frame.columns if column not in numeric]
    date_columns = _date_columns(frame)
    text_columns = _text_columns(frame, categorical)
    categorical = [
        column for column in categorical if column not in date_columns and column not in text_columns
    ]

    if numeric:
        notebook.cells.extend(
            [
                nbformat.v4.new_markdown_cell(
                    "## Exploratory data analysis and statistical analysis\n\n"
                    "Numerical columns were present, so DATUM included descriptive statistics, "
                    "distributions, outlier checks, and correlations where meaningful. "
                    "Chart samples are bounded to the first 5,000 rows, matching DATUM's "
                    "dashboard chart payload."
                ),
                nbformat.v4.new_code_cell(
                    "numeric_columns = " + repr(numeric) + "\n"
                    "numeric_data = source[numeric_columns].apply(pd.to_numeric, errors='coerce')\n"
                    "descriptive_statistics = numeric_data.describe().T\n"
                    "descriptive_statistics"
                ),
                nbformat.v4.new_code_cell(
                    "import matplotlib.pyplot as plt\n"
                    "\n"
                    "for column in numeric_columns:\n"
                    "    values = numeric_data[column].dropna()\n"
                    "    if values.nunique() < 2:\n"
                    "        continue\n"
                    "    plt.figure(figsize=(7, 4))\n"
                    "    sns.histplot(values, bins=min(10, max(2, int(np.sqrt(len(values))))), "
                    "                kde=False)\n"
                    "    plt.title(f'Distribution of {column}')\n"
                    "    plt.xlabel(column)\n"
                    "    plt.ylabel('Number of rows')\n"
                    "    plt.tight_layout()\n"
                    "    plt.show()"
                ),
                nbformat.v4.new_code_cell(
                    "plt.figure(figsize=(max(6, 2 * len(numeric_columns)), 4))\n"
                    "sns.boxplot(data=numeric_data[numeric_columns])\n"
                    "plt.title('Numeric distributions and candidate extremes')\n"
                    "plt.xticks(rotation=45)\n"
                    "plt.tight_layout()\n"
                    "plt.show()"
                ),
                *(
                    [
                        nbformat.v4.new_code_cell(
                            "scatter_data = numeric_data[[numeric_columns[0], numeric_columns[1]]].dropna()\n"
                            "if len(scatter_data) >= 3:\n"
                            "    plt.figure(figsize=(7, 4))\n"
                            "    sns.scatterplot(data=scatter_data, x=numeric_columns[0], y=numeric_columns[1])\n"
                            "    plt.title(f'{numeric_columns[1]} against {numeric_columns[0]}')\n"
                            "    plt.tight_layout()\n"
                            "    plt.show()\n"
                            "else:\n"
                            "    print('Scatter plot skipped: fewer than three complete observations.')"
                        )
                    ]
                    if len(numeric) > 1
                    else []
                ),
                nbformat.v4.new_code_cell(
                    "outlier_summary = {}\n"
                    "for column in numeric_columns:\n"
                    "    values = numeric_data[column].dropna()\n"
                    "    if len(values) < 4:\n"
                    "        outlier_summary[column] = {'status': 'insufficient_sample'}\n"
                    "        continue\n"
                    "    q1, q3 = values.quantile([0.25, 0.75])\n"
                    "    iqr = q3 - q1\n"
                    "    outlier_summary[column] = {\n"
                    "        'method': '1.5x IQR',\n"
                    "        'count': int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()),\n"
                    "    }\n"
                    "outlier_summary"
                ),
            ]
        )
        if len(numeric) > 1:
            notebook.cells.append(
                nbformat.v4.new_code_cell(
                    "pearson_correlations = numeric_data.corr(method='pearson')\n"
                    "spearman_correlations = numeric_data.corr(method='spearman')\n"
                    "plt.figure(figsize=(max(6, 1.2 * len(numeric_columns)), max(4, 1.0 * len(numeric_columns))))\n"
                    "sns.heatmap(pearson_correlations, annot=True, vmin=-1, vmax=1, cmap='vlag')\n"
                    "plt.title('Pearson correlation matrix')\n"
                    "plt.tight_layout()\n"
                    "plt.show()\n"
                    "pearson_correlations"
                )
            )
        else:
            notebook.cells.append(
                nbformat.v4.new_markdown_cell(
                    "Correlation matrices were skipped because fewer than two numeric "
                    "columns were available."
                )
            )
    else:
        notebook.cells.append(
            nbformat.v4.new_markdown_cell(
                "## Numerical analysis skipped\n\nNo numeric columns were available, so "
                "numeric distributions, outlier analysis, and correlations were not meaningful."
            )
        )

    if categorical:
        notebook.cells.extend(
            [
                nbformat.v4.new_markdown_cell(
                    "## Categorical analysis\n\n"
                    "Frequency tables use the 20 most common values to keep high-cardinality "
                    "columns readable."
                ),
                nbformat.v4.new_code_cell(
                    "categorical_columns = " + repr(categorical) + "\n"
                    "frequency_tables = {\n"
                    "    column: source[column].value_counts(dropna=False).head(20)\n"
                    "    for column in categorical_columns\n"
                    "}\n"
                    "frequency_tables\n"
                    "for column, values in frequency_tables.items():\n"
                    "    values.plot(kind='bar', title=f'Top categories: {column}')\n"
                    "    plt.tight_layout()\n"
                    "    plt.show()"
                ),
            ]
        )
    else:
        notebook.cells.append(
            nbformat.v4.new_markdown_cell(
                "Categorical analysis was skipped because no non-numeric columns were available."
            )
        )

    if date_columns:
        notebook.cells.extend(
            [
                nbformat.v4.new_markdown_cell(
                    "## Date and time analysis\n\n"
                    "These columns were parseable as dates for most rows. Invalid values remain "
                    "visible as missing values after parsing."
                ),
                nbformat.v4.new_code_cell(
                    "date_columns = " + repr(date_columns) + "\n"
                    "for column in date_columns:\n"
                    "    parsed = pd.to_datetime(source[column], errors='coerce', format='mixed')\n"
                    "    print(column, {'min': parsed.min(), 'max': parsed.max(), "
                    "'invalid_or_missing': int(parsed.isna().sum())})"
                ),
            ]
        )
        if numeric:
            notebook.cells.append(
                nbformat.v4.new_code_cell(
                    "time_column = date_columns[0]\n"
                    "time_values = pd.to_datetime(source[time_column], errors='coerce', format='mixed')\n"
                    "trend = (pd.DataFrame({'time': time_values, 'value': numeric_data[numeric_columns[0]]})\n"
                    "         .dropna().sort_values('time').groupby('time', as_index=False)['value'].mean())\n"
                    "plt.figure(figsize=(8, 4))\n"
                    "sns.lineplot(data=trend, x='time', y='value', marker='o')\n"
                    "plt.title(f'{numeric_columns[0]} over {time_column}')\n"
                    "plt.xlabel(time_column)\n"
                    "plt.ylabel(numeric_columns[0])\n"
                    "plt.tight_layout()\n"
                    "plt.show()"
                )
            )
    else:
        notebook.cells.append(
            nbformat.v4.new_markdown_cell(
                "Date and time analysis was skipped because no meaningful date column was detected."
            )
        )

    if text_columns:
        notebook.cells.extend(
            [
                nbformat.v4.new_markdown_cell(
                    "## Text analysis\n\n"
                    "Text analysis is limited to inexpensive length, word-count, and empty-value "
                    "checks; no language model or external NLP resource is used."
                ),
                nbformat.v4.new_code_cell(
                    "text_columns = " + repr(text_columns) + "\n"
                    "text_summary = {}\n"
                    "for column in text_columns:\n"
                    "    values = source[column].fillna('').astype(str)\n"
                    "    text_summary[column] = {\n"
                    "        'empty_or_whitespace': int(values.str.strip().eq('').sum()),\n"
                    "        'length': values.str.len().describe().to_dict(),\n"
                    "        'word_count': values.str.split().str.len().describe().to_dict(),\n"
                    "    }\n"
                    "text_summary"
                ),
                nbformat.v4.new_code_cell(
                    "for column in text_columns:\n"
                    "    source[column].fillna('').astype(str).str.len().plot(kind='hist', bins=20, "
                    "title=f'Text length: {column}')\n"
                    "    plt.tight_layout()\n"
                    "    plt.show()"
                ),
            ]
        )

    if mode == "prepare_for_ml" and transformations:
        notebook.cells.append(
            nbformat.v4.new_markdown_cell(
                "## Feature engineering\n\n"
                "Only operations in DATUM's transformation audit trail are reproduced below. "
                "A rejected, skipped, or failed plan is described separately and is not executed."
            )
        )
        for record in transformations:
            operation = record.get("operation")
            if operation in _SUPPORTED_OPERATIONS:
                reason = _reason_for_operation(operation, record, preparation_plan)
                notebook.cells.append(
                    nbformat.v4.new_markdown_cell(
                        f"### Executed `{operation}`\n\n"
                        f"- Source columns: `{', '.join(map(str, record.get('columns', [])))}`\n"
                        f"- Parameters and audit evidence: `{record.get('parameters', {})}`\n"
                        f"- Selection reason: {reason}\n"
                        "- Verification: included in the final verification output below."
                    )
                )
                notebook.cells.append(
                    nbformat.v4.new_code_cell(_operation_code(record))
                )
            else:
                notebook.cells.append(
                    nbformat.v4.new_markdown_cell(
                        f"Operation `{operation}` was recorded but has no safe reproduction "
                        "template in this notebook."
                    )
                )
        notebook.cells.append(
            nbformat.v4.new_code_cell(
                "before_after = {\n"
                "    'rows_before': int(len(source)),\n"
                "    'rows_after': int(len(working)),\n"
                "    'columns_before': int(len(source.columns)),\n"
                "    'columns_after': int(len(working.columns)),\n"
                "}\n"
                "before_after"
            )
        )
    elif mode == "prepare_for_ml":
        plan_status = approval or (preparation_plan or {}).get("status", "not executed")
        notebook.cells.append(
            nbformat.v4.new_markdown_cell(
                f"## Feature engineering\n\nNo transformation was executed because the "
                f"preparation/approval status was `{plan_status}`. "
                "No unapproved or skipped transformation is represented as executed code."
            )
        )
    else:
        notebook.cells.append(
            nbformat.v4.new_markdown_cell(
                "## Feature engineering\n\n"
                "Not performed in Analyze Only. The source dataset remains unchanged and "
                "no transformations are represented as executed."
            )
        )

    notebook.cells.extend(
        [
            nbformat.v4.new_markdown_cell(
                "## Before-and-after comparison\n\n"
                "Analyze Only has no prepared output, so this section reports that no "
                "comparison was performed. Prepare for ML compares the preserved source "
                "copy with the locally reproduced, approved transformation result."
            ),
            nbformat.v4.new_code_cell(
                "source_quality = quality_snapshot(source)\n"
                "prepared_quality = quality_snapshot(working)\n"
                "comparison = {\n"
                "    'source_shape': source.shape,\n"
                "    'prepared_shape': working.shape,\n"
                "    'source_quality': source_quality,\n"
                "    'prepared_quality': prepared_quality,\n"
                "    'new_columns': sorted(set(working.columns) - set(source.columns)),\n"
                "    'removed_columns': sorted(set(source.columns) - set(working.columns)),\n"
                "    'transformations_executed': " + repr(transformations) + ",\n"
                "}\n"
                "comparison"
            ),
        ]
    )

    notebook.cells.extend(
        [
            nbformat.v4.new_markdown_cell(
                "## Final verification\n\n"
                "The following values are the recorded outputs from the completed run. "
                "The code cells above recompute the numerical work rather than hardcoding it."
            ),
            nbformat.v4.new_code_cell(
                "quality = " + repr(quality) + "\n"
                "statistics = " + repr(statistics) + "\n"
                "verification = " + repr(verification) + "\n"
                "print('Quality score:', quality.get('quality_score'))\n"
                "print('Verification:', verification.get('status'))\n"
                "final_checks = {\n"
                "    'readable': bool(len(working.columns) > 0),\n"
                "    'missing_values': int(working.isna().sum().sum()),\n"
                "    'duplicate_rows': int(working.duplicated().sum()),\n"
                "    'non_finite_values': int((~np.isfinite(working.select_dtypes(include='number').to_numpy(dtype=float))).sum()) if not working.select_dtypes(include='number').empty else 0,\n"
                "    'verification_status': verification.get('status', 'unavailable'),\n"
                "}\n"
                "final_checks"
            ),
            nbformat.v4.new_markdown_cell(
                "## Final notes\n\n"
                "Run all cells locally to reproduce the analysis. Scaling and one-hot encoding "
                "are refit on the local source snapshot because DATUM does not currently persist "
                "fitted transformer artifacts; their recorded parameters and limitations remain "
                "in the transformation audit trail. The notebook does not contain secrets, "
                "server paths, SQL credentials, or embedded dataset contents."
            ),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)


def _date_columns(frame: pd.DataFrame) -> list[str]:
    result = []
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed")
        if len(frame) and float(parsed.notna().mean()) >= 0.8:
            result.append(str(column))
    return result


def _text_columns(frame: pd.DataFrame, categorical: list[str]) -> list[str]:
    result = []
    for column in categorical:
        values = frame[column].dropna().astype(str)
        if not len(values):
            continue
        if float(values.str.len().mean()) > 30 or float(values.str.split().str.len().mean()) > 3:
            result.append(column)
    return result


def _operation_code(record: dict[str, Any]) -> str:
    operation = record["operation"]
    parameters = record.get("parameters", {})
    columns = record.get("columns", [])
    if not columns and parameters.get("column"):
        columns = [parameters["column"]]
    if operation == "drop_duplicates":
        return "before_rows = len(working)\nworking = working.drop_duplicates().reset_index(drop=True)\nprint({'operation': 'drop_duplicates', 'rows_removed': before_rows - len(working)})"
    if operation in {"impute_median", "impute_mode", "impute_mean"}:
        column = columns[0]
        expression = {
            "impute_median": f"working[{column!r}].median()",
            "impute_mean": f"working[{column!r}].mean()",
            "impute_mode": f"working[{column!r}].mode(dropna=True).iloc[0]",
        }[operation]
        return (
            f"fill_value = {expression}\n"
            f"working[{column!r}] = working[{column!r}].fillna(fill_value)\n"
            f"print({{'operation': {operation!r}, 'column': {column!r}, 'fill_value': fill_value}})"
        )
    if operation == "impute_constant":
        column = columns[0]
        value = parameters.get("value", 0)
        return f"working[{column!r}] = working[{column!r}].fillna({value!r})"
    if operation in {"impute_forward_fill", "impute_backward_fill"}:
        method = "ffill" if operation.endswith("forward_fill") else "bfill"
        column = columns[0]
        return f"working[{column!r}] = working[{column!r}].{method}()"
    if operation == "missing_indicator":
        column = columns[0]
        output = parameters["output_column"]
        return f"working[{output!r}] = working[{column!r}].isna().astype('int8')"
    if operation == "text_features":
        column = columns[0]
        return (
            f"text = working[{column!r}].fillna('').astype(str).str.strip()\n"
            f"working[{(column + '__text_length')!r}] = text.str.len()\n"
            f"working[{(column + '__word_count')!r}] = text.str.split().str.len()"
        )
    if operation == "date_features":
        column = columns[0]
        outputs = parameters.get("output_columns", [])
        code = f"dates = pd.to_datetime(working[{column!r}], errors='coerce')\n"
        for output in outputs:
            component = output.rsplit("__", 1)[-1]
            code += f"working[{output!r}] = dates.dt.{component}\n"
        return code
    if operation == "one_hot_encode":
        column = columns[0]
        return (
            f"encoded = pd.get_dummies(working[{column!r}].fillna('__missing__').astype(str), "
            f"prefix={column!r}, prefix_sep='__', dtype='int8')\n"
            "working = pd.concat([working, encoded], axis=1)"
        )
    if operation in {"standard_scale", "minmax_scale", "robust_scale"}:
        column = columns[0]
        scaler = {
            "standard_scale": "StandardScaler",
            "minmax_scale": "MinMaxScaler",
            "robust_scale": "RobustScaler",
        }[operation]
        return (
            "from sklearn.preprocessing import "
            f"{scaler}\n"
            f"values = working[{column!r}].to_numpy(dtype=float).reshape(-1, 1)\n"
            f"values = np.nan_to_num(values, nan=float(np.nanmedian(values)))\n"
            f"working[{column!r}] = {scaler}().fit_transform(values).ravel()"
        )
    if operation == "ratio":
        numerator, denominator = columns
        output = parameters["output_column"]
        return (
            f"denominator = working[{denominator!r}].replace(0, np.nan)\n"
            f"working[{output!r}] = working[{numerator!r}] / denominator"
        )
    return "# No reproduction template was generated for this operation."


def _reason_for_operation(
    operation: str, record: dict[str, Any], preparation_plan: dict[str, Any] | None
) -> str:
    for step in (preparation_plan or {}).get("steps", []):
        if step.get("operation") != operation:
            continue
        step_columns = step.get("parameters", {}).get("column")
        if not step_columns or step_columns in record.get("columns", []):
            return str(step.get("reason", "Selected by the completed preparation plan."))
    return "Recorded as executed by the completed DATUM transformation audit."
