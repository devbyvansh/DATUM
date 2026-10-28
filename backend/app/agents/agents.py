from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.app.ai_provider import ProviderClient
from backend.app.feature_registry import add_executed_decisions, assess_feature_options
from backend.app.schemas.agent import (
    AgentFinding,
    EvidenceReference,
    PreparationAgentOutput,
    PreparationStep,
    QualityAgentOutput,
    StatisticalAgentOutput,
    VerificationAgentOutput,
)


def quality_agent(
    quality: dict[str, Any], profile: dict[str, Any], provider: ProviderClient
) -> QualityAgentOutput:
    result = provider.invoke(
        {
            "quality": quality,
            "profile": profile,
            "instructions": "Interpret only supplied evidence.",
        },
        QualityAgentOutput,
    )
    if result.status != "unavailable":
        return (
            QualityAgentOutput.model_validate(result.output)
            if result.status == "completed"
            else QualityAgentOutput(status="failed")
        )
    findings = [
        AgentFinding(
            message=item["message"],
            recommendation=item["recommendation"],
            evidence=[EvidenceReference(source="quality", key="evidence", value=item["evidence"])],
        )
        for item in quality.get("findings", [])
    ]
    return QualityAgentOutput(status="unavailable", findings=findings)


def preparation_agent(
    config: dict[str, Any],
    quality: dict[str, Any],
    provider: ProviderClient,
    frame: pd.DataFrame | None = None,
) -> PreparationAgentOutput:
    if not config and frame is not None:
        return _automatic_preparation_plan(frame, quality)
    result = provider.invoke(
        {"requested_operations": config, "quality": quality},
        PreparationAgentOutput,
    )
    if result.status == "completed":
        output = PreparationAgentOutput.model_validate(result.output)
        if frame is not None and not output.decisions:
            output.decisions = assess_feature_options(frame)
        return output
    steps: list[PreparationStep] = []
    if config.get("drop_duplicates"):
        steps.append(
            PreparationStep(
                operation="drop_duplicates",
                reason="Quality checks found or requested duplicate removal.",
            )
        )
    for column in config.get("impute_median", []):
        steps.append(
            PreparationStep(
                operation="impute_median",
                parameters={"column": column},
                reason="Median imputation was explicitly requested.",
            )
        )
    for column in config.get("impute_mode", []):
        steps.append(
            PreparationStep(
                operation="impute_mode",
                parameters={"column": column},
                reason="Mode imputation was explicitly requested.",
            )
        )
    operation_parameters = {
        "impute_mean": config.get("impute_mean", []),
        "impute_constant": config.get("impute_constant", []),
        "impute_forward_fill": config.get("impute_forward_fill", []),
        "impute_backward_fill": config.get("impute_backward_fill", []),
        "missing_indicator": config.get("missing_indicators", []),
        "text_features": config.get("text_features", []),
        "date_features": config.get("date_features", []),
        "one_hot_encode": config.get("one_hot_encode", []),
        "standard_scale": config.get("standard_scale", []),
        "minmax_scale": config.get("minmax_scale", []),
        "robust_scale": config.get("robust_scale", []),
        "ratio": config.get("ratios", []),
    }
    for operation, parameters in operation_parameters.items():
        for parameter in parameters:
            column = parameter if isinstance(parameter, str) else parameter.get("column", "")
            steps.append(
                PreparationStep(
                    operation=operation,
                    parameters=parameter if isinstance(parameter, dict) else {"column": column},
                    reason=f"{operation} was explicitly requested.",
                )
            )
    decisions = assess_feature_options(frame) if frame is not None else []
    return PreparationAgentOutput(
        status="unavailable",
        steps=steps,
        decisions=add_executed_decisions(decisions, steps),
    )


def _automatic_preparation_plan(
    frame: pd.DataFrame, quality: dict[str, Any]
) -> PreparationAgentOutput:
    steps: list[PreparationStep] = []
    decisions = assess_feature_options(frame)
    if int(frame.duplicated().sum()):
        steps.append(
            PreparationStep(
                operation="drop_duplicates",
                reason="Exact duplicate rows were detected.",
                expected_effect="Remove repeated records while preserving the first occurrence.",
                safety_checks=["Compare row counts before and after the operation."],
            )
        )

    date_columns: set[str] = set()
    text_columns: set[str] = set()
    identifier_columns: set[str] = set()
    for item in quality.get("findings", []):
        if item.get("check_id") == "high_cardinality":
            identifier_columns.update(item.get("columns", []))

    for column in frame.columns:
        name = str(column)
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if finite.notna().any() and finite.isna().any():
                steps.append(
                    PreparationStep(
                        operation="impute_median",
                        parameters={"column": name},
                        reason=f"Fill missing values in {name} with its observed median.",
                        expected_effect=(
                            "Replace missing or non-finite numeric values without changing "
                            "row count."
                        ),
                        safety_checks=["Use only finite values to calculate the median."],
                    )
                )
                steps.append(
                    PreparationStep(
                        operation="missing_indicator",
                        parameters={"column": name},
                        reason=f"Keep a flag showing which {name} values were missing.",
                        expected_effect=(
                            f"Add {name}__missing with 1 for originally missing values."
                        ),
                        expected_output_columns=[f"{name}__missing"],
                    )
                )
            if (
                finite.notna().sum() > 0
                and finite.nunique(dropna=True) > 1
                and name not in identifier_columns
                and not np.isinf(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)).any()
            ):
                steps.append(
                    PreparationStep(
                        operation="standard_scale",
                        parameters={"column": name},
                        reason=f"Scale variable {name}, which has usable numeric variation.",
                        expected_effect=(
                            "Center and scale the numeric column for preparation workflows."
                        ),
                        safety_checks=["Skip columns with no finite variation."],
                    )
                )
            continue

        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        if len(series) and parsed.notna().mean() >= 0.8:
            date_columns.add(name)
            components = ["year", "quarter", "month", "dayofweek"]
            if parsed.dt.hour.fillna(0).ne(0).any():
                components.append("hour")
            outputs = [f"{name}__{component}" for component in components]
            steps.append(
                PreparationStep(
                    operation="date_features",
                    parameters={"column": name, "components": components},
                    reason=f"Most values in {name} parse as dates.",
                    expected_effect=(
                        "Add calendar features while preserving the original date column."
                    ),
                    safety_checks=["Invalid dates receive an explicit -1 feature value."],
                    expected_output_columns=outputs,
                )
            )
            continue

        values = series.dropna().astype(str)
        average_words = float(values.str.split().str.len().mean()) if len(values) else 0
        average_length = float(values.str.len().mean()) if len(values) else 0
        text_name = any(
            token in name.lower()
            for token in ("text", "description", "comment", "notes", "message", "review")
        )
        if text_name or average_words > 3 or average_length > 30:
            text_columns.add(name)
            steps.append(
                PreparationStep(
                    operation="text_features",
                    parameters={"column": name},
                    reason=f"{name} contains free-form text suitable for simple length features.",
                    expected_effect="Add text length and word-count columns without external NLP.",
                    expected_output_columns=[f"{name}__text_length", f"{name}__word_count"],
                )
            )
            continue

        cardinality = int(series.nunique(dropna=True))
        identifier_name = any(
            token in name.lower() for token in ("id", "identifier", "uuid", "key", "code")
        )
        if series.isna().any() and cardinality:
            steps.append(
                PreparationStep(
                    operation="impute_mode",
                    parameters={"column": name},
                    reason=(
                        f"Fill missing categories in {name} with its most common observed value."
                    ),
                    expected_effect="Replace missing category values without changing row count.",
                )
            )
        if 2 <= cardinality <= 10 and name not in identifier_columns and not identifier_name:
            steps.append(
                PreparationStep(
                    operation="one_hot_encode",
                    parameters={"column": name},
                    reason=f"Encode low-cardinality categories in {name} for ML preparation.",
                    expected_effect="Add one binary column per observed category.",
                    safety_checks=["Skip identifier-like and high-cardinality columns."],
                )
            )

    return PreparationAgentOutput(
        status="unavailable",
        steps=steps,
        decisions=add_executed_decisions(decisions, steps),
    )


def statistical_agent(
    statistics: dict[str, Any], provider: ProviderClient
) -> StatisticalAgentOutput:
    result = provider.invoke({"statistics": statistics}, StatisticalAgentOutput)
    if result.status == "completed":
        return StatisticalAgentOutput.model_validate(result.output)
    charts = statistics.get("charts", [])
    findings = [
        AgentFinding(
            message=(
                "Calculated statistics are available for "
                f"{len(statistics.get('descriptive_statistics', {}))} numeric columns."
            ),
            recommendation="Review distributions and limitations before drawing conclusions.",
            evidence=[
                EvidenceReference(
                    source="statistics",
                    key="descriptive_statistics",
                    value=statistics.get("descriptive_statistics", {}),
                )
            ],
        )
    ]
    return StatisticalAgentOutput(
        status=result.status,
        findings=findings,
        selected_chart_ids=[chart["id"] for chart in charts],
    )


def verification_agent(
    verification: dict[str, Any], provider: ProviderClient
) -> VerificationAgentOutput:
    result = provider.invoke({"verification": verification}, VerificationAgentOutput)
    if result.status == "completed":
        return VerificationAgentOutput.model_validate(result.output)
    status = "passed" if verification.get("status") == "passed" else "failed"
    return VerificationAgentOutput(status=status)
