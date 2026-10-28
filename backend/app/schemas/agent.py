from typing import Literal

from pydantic import BaseModel, Field


class EvidenceReference(BaseModel):
    source: Literal["quality", "profile", "statistics", "verification"]
    key: str
    value: object


class AgentFinding(BaseModel):
    message: str
    recommendation: str
    evidence: list[EvidenceReference] = Field(default_factory=list)


class QualityAgentOutput(BaseModel):
    status: Literal["completed", "unavailable", "failed"]
    findings: list[AgentFinding] = Field(default_factory=list)


class PreparationStep(BaseModel):
    operation: Literal[
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
    ]
    parameters: dict = Field(default_factory=dict)
    reason: str
    requires_approval: bool = True
    expected_effect: str = ""
    safety_checks: list[str] = Field(default_factory=list)
    expected_output_columns: list[str] = Field(default_factory=list)


class PreparationAgentOutput(BaseModel):
    status: Literal["completed", "unavailable", "failed"]
    steps: list[PreparationStep] = Field(default_factory=list)
    decisions: list[dict] = Field(default_factory=list)


class StatisticalAgentOutput(BaseModel):
    status: Literal["completed", "unavailable", "failed"]
    findings: list[AgentFinding] = Field(default_factory=list)
    selected_chart_ids: list[str] = Field(default_factory=list)


class VerificationAgentOutput(BaseModel):
    status: Literal["passed", "warning", "failed"]
    findings: list[AgentFinding] = Field(default_factory=list)


class ProviderResult(BaseModel):
    status: Literal["completed", "unavailable", "failed"]
    model: str | None = None
    output: dict = Field(default_factory=dict)
    error: str | None = None
