from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TechniqueSpec:
    identifier: str
    category: str
    leakage_risk: str
    fit_scope: str
    required_inputs: tuple[str, ...]


FEATURE_REGISTRY: dict[str, TechniqueSpec] = {
    "impute_median": TechniqueSpec(
        "impute_median", "missing_values", "low", "dataset-scope", ("numeric_column",)
    ),
    "missing_indicator": TechniqueSpec(
        "missing_indicator", "missing_values", "low", "stateless", ("column",)
    ),
    "date_features": TechniqueSpec(
        "date_features", "date_time", "low", "stateless", ("date_column",)
    ),
    "text_features": TechniqueSpec(
        "text_features", "text", "low", "stateless", ("text_column",)
    ),
    "one_hot_encode": TechniqueSpec(
        "one_hot_encode", "categorical", "medium", "dataset-scope", ("categorical_column",)
    ),
    "standard_scale": TechniqueSpec(
        "standard_scale", "numeric_transform", "medium", "dataset-scope", ("numeric_column",)
    ),
    "drop_duplicates": TechniqueSpec(
        "drop_duplicates", "row_quality", "none", "stateless", ("dataframe",)
    ),
    "target_encoding": TechniqueSpec(
        "target_encoding", "categorical", "high", "training-only", ("target", "split")
    ),
    "resampling": TechniqueSpec(
        "resampling", "class_imbalance", "high", "training-only", ("target", "split")
    ),
    "dimensionality_reduction": TechniqueSpec(
        "dimensionality_reduction", "feature_selection", "high", "training-only", ("split",)
    ),
}


@dataclass(frozen=True)
class FeatureDecision:
    technique: str
    category: str
    status: str
    reason: str
    columns: list[str]
    parameters: dict[str, Any]
    generated_features: list[str]
    leakage_risk: str
    fit_scope: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "technique": self.technique,
            "category": self.category,
            "status": self.status,
            "reason": self.reason,
            "columns": self.columns,
            "parameters": self.parameters,
            "generated_features": self.generated_features,
            "leakage_risk": self.leakage_risk,
            "fit_scope": self.fit_scope,
        }


def assess_feature_options(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return explainable, split-free decisions for the safe preparation registry.

    Learned operations are explicitly marked as dataset-scope because the general
    preparation workflow has no configured training partition.
    """
    decisions: list[FeatureDecision] = []
    numeric = list(frame.select_dtypes(include="number").columns)
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.isna().any() and values.notna().any():
            decisions.append(
                FeatureDecision(
                    "impute_median",
                    "missing_values",
                    "selected",
                    "Finite numeric observations support median imputation.",
                    [str(column)],
                    {"column": str(column)},
                    [],
                    "low; label-independent statistic fit on the current dataset",
                    "dataset-scope",
                )
            )
        elif values.isna().all():
            decisions.append(
                FeatureDecision(
                    "impute_median",
                    "missing_values",
                    "skipped",
                    "The column has no finite observations from which to calculate a median.",
                    [str(column)],
                    {"column": str(column)},
                    [],
                    "none",
                    "not applicable",
                )
            )
        if values.nunique(dropna=True) <= 1:
            decisions.append(
                FeatureDecision(
                    "scaling",
                    "numeric_transform",
                    "skipped",
                    "Constant or unusable numeric columns are not scaled.",
                    [str(column)],
                    {},
                    [],
                    "none",
                    "not applicable",
                )
            )

    for column in frame.select_dtypes(exclude="number").columns:
        name = str(column)
        series = frame[column]
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        if len(series) and parsed.notna().mean() >= 0.8:
            components = ["year", "quarter", "month", "dayofweek"]
            decisions.append(
                FeatureDecision(
                    "date_features",
                    "date_time",
                    "selected",
                    "Most non-empty values parse as dates, so calendar components are applicable.",
                    [name],
                    {"column": name, "components": components},
                    [f"{name}__{component}" for component in components],
                    "low; derived without target information",
                    "stateless",
                )
            )
            continue
        cardinality = int(series.nunique(dropna=True))
        identifier_like = any(
            token in name.lower() for token in ("id", "uuid", "key", "identifier")
        )
        if identifier_like or (len(series) and cardinality / len(series) > 0.8):
            decisions.append(
                FeatureDecision(
                    "categorical_encoding",
                    "categorical",
                    "skipped",
                    "Identifier-like or high-cardinality values are not one-hot encoded.",
                    [name],
                    {"cardinality": cardinality},
                    [],
                    "medium; high-dimensional and identifier leakage risk",
                    "not applicable",
                )
            )
        elif 2 <= cardinality <= 10:
            decisions.append(
                FeatureDecision(
                    "one_hot_encode",
                    "categorical",
                    "selected",
                    "Low-cardinality categories can be encoded without target information.",
                    [name],
                    {"column": name},
                    [],
                    "medium; category vocabulary is fit on the current dataset",
                    "dataset-scope",
                )
            )
        else:
            decisions.append(
                FeatureDecision(
                    "categorical_encoding",
                    "categorical",
                    "skipped",
                    "Cardinality is outside the safe automatic one-hot range.",
                    [name],
                    {"cardinality": cardinality},
                    [],
                    "medium",
                    "not applicable",
                )
            )

    decisions.extend(
        [
            FeatureDecision(
                "target_encoding",
                "categorical",
                "requires_target",
                "Target-dependent encoding requires an explicit target and training partition.",
                [],
                {},
                [],
                "high; target leakage risk",
                "training-only",
            ),
            FeatureDecision(
                "resampling",
                "class_imbalance",
                "requires_user_input",
                "Resampling is classification-specific and must run inside a training fold.",
                [],
                {},
                [],
                "high; validation leakage risk",
                "training-only",
            ),
            FeatureDecision(
                "dimensionality_reduction",
                "feature_selection",
                "skipped",
                "PCA or feature selection is not applied without a configured training pipeline.",
                [],
                {},
                [],
                "high; fitted model leakage risk",
                "training-only",
            ),
        ]
    )
    return [decision.as_dict() for decision in decisions]


def add_executed_decisions(
    decisions: list[dict[str, Any]], steps: list[Any]
) -> list[dict[str, Any]]:
    """Make the audit explicit when the planner selected an existing operation."""
    known = {
        (item["technique"], tuple(item.get("columns", [])))
        for item in decisions
        if item.get("status") == "selected"
    }
    for step in steps:
        column = step.parameters.get("column")
        columns = [column] if column else []
        key = (step.operation, tuple(columns))
        if key in known:
            continue
        decisions.append(
            FeatureDecision(
                technique=step.operation,
                category="executed_preparation",
                status="selected",
                reason=step.reason,
                columns=columns,
                parameters=dict(step.parameters),
                generated_features=list(step.expected_output_columns),
                leakage_risk="review required for dataset-scope fitting",
                fit_scope="dataset-scope",
            ).as_dict()
        )
        known.add(key)
    return decisions
