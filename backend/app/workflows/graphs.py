from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from backend.app import charts as chart_utils
from backend.app.agents.agents import (
    preparation_agent,
    quality_agent,
    statistical_agent,
    verification_agent,
)
from backend.app.ai_provider import ProviderClient
from backend.app.config import Settings
from backend.app.data_processing import (
    load_dataframe,
    prepare_dataframe,
    profile_dataframe,
    quality_report,
    statistics,
    verify,
)


class WorkflowState(TypedDict, total=False):
    run_id: str
    dataset_id: str
    source_path: str
    dataset_name: str
    mode: str
    preparation_config: dict[str, Any]
    stage: str
    stage_status: dict[str, str]
    profile: dict[str, Any]
    quality: dict[str, Any]
    preparation_plan: dict[str, Any]
    approval: str
    transformations: list[dict[str, Any]]
    working_path: str
    statistics: dict[str, Any]
    verification: dict[str, Any]
    quality_agent: dict[str, Any]
    preparation_agent: dict[str, Any]
    feature_decisions: list[dict[str, Any]]
    statistical_agent: dict[str, Any]
    verification_agent: dict[str, Any]
    report: dict[str, Any]
    error: str


class DatumGraphs:
    def __init__(self, settings: Settings, provider: ProviderClient | None = None) -> None:
        self.settings = settings
        self.provider = provider or ProviderClient(settings)
        settings.checkpoint_database_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(
            settings.checkpoint_database_path, check_same_thread=False
        )
        self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.checkpointer.setup()
        self.analyze_only = self._build_analyze_only().compile(checkpointer=self.checkpointer)
        self.prepare_for_ml = self._build_prepare().compile(checkpointer=self.checkpointer)

    def close(self) -> None:
        self._checkpoint_connection.close()

    def config(self, run_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": run_id}}

    def _common_nodes(self) -> dict[str, Any]:
        def status(state: WorkflowState, name: str, value: str) -> dict[str, str]:
            return {**state.get("stage_status", {}), name: value}

        def validate(state: WorkflowState) -> dict[str, Any]:
            frame = load_dataframe(Path(state["source_path"]))
            return {
                "stage": "validate",
                "stage_status": status(state, "validate", "completed"),
                "profile": profile_dataframe(frame),
            }

        def quality(state: WorkflowState) -> dict[str, Any]:
            frame = load_dataframe(Path(state["source_path"]))
            report = quality_report(frame)
            return {
                "stage": "quality",
                "stage_status": status(state, "quality", "completed"),
                "quality": report,
                "quality_agent": quality_agent(
                    report, state["profile"], self.provider
                ).model_dump(),
            }

        def analysis(state: WorkflowState) -> dict[str, Any]:
            path = Path(state.get("working_path", state["source_path"]))
            frame = load_dataframe(path)
            report = statistics(frame)
            try:
                enhanced_charts = chart_utils.generate_chart_images(
                    state["run_id"],
                    frame,
                    report.get("charts", []),
                    self.settings.report_dir.parent / "charts",
                )
                report["charts"] = chart_utils.add_dataset_context(
                    enhanced_charts,
                    state.get("dataset_name", Path(state["source_path"]).name),
                    len(frame),
                    len(frame.columns),
                )
            except Exception:
                pass
            return {
                "stage": "analysis",
                "stage_status": status(state, "analysis", "completed"),
                "statistics": report,
                "statistical_agent": statistical_agent(report, self.provider).model_dump(),
            }

        def verification(state: WorkflowState) -> dict[str, Any]:
            path = Path(state.get("working_path", state["source_path"]))
            frame = load_dataframe(path)
            result = verify(
                frame, {"statistics": state["statistics"]}, state.get("transformations", [])
            )
            agent = verification_agent(result, self.provider)
            return {
                "stage": "verification",
                "stage_status": status(state, "verification", agent.status),
                "verification": result,
                "verification_agent": agent.model_dump(),
            }

        def final(state: WorkflowState) -> dict[str, Any]:
            report = {
                key: state[key]
                for key in (
                    "run_id",
                    "dataset_id",
                    "mode",
                    "profile",
                    "quality",
                    "preparation_plan",
                    "approval",
                    "transformations",
                    "statistics",
                    "verification",
                    "quality_agent",
                    "preparation_agent",
                    "feature_decisions",
                    "statistical_agent",
                    "verification_agent",
                )
                if key in state
            }
            agent_statuses = [
                state.get(name, {}).get("status")
                for name in (
                    "quality_agent",
                    "preparation_agent",
                    "statistical_agent",
                    "verification_agent",
                )
                if name in state
            ]
            if "failed" in agent_statuses:
                ai_status = "failed"
            elif "completed" in agent_statuses:
                ai_status = "completed"
            else:
                ai_status = "unavailable"
            report["ai"] = {"status": ai_status}
            if state.get("working_path") and state["mode"] == "prepare_for_ml":
                report["prepared_path"] = state["working_path"]
            if state.get("approval") == "rejected":
                report["outcome"] = "rejected"
            return {
                "stage": "final",
                "stage_status": status(state, "final", "completed"),
                "report": report,
            }

        return {
            "validate": validate,
            "quality": quality,
            "analysis": analysis,
            "verification": verification,
            "final": final,
        }

    def _build_analyze_only(self) -> StateGraph:
        graph = StateGraph(WorkflowState)
        for name, node in self._common_nodes().items():
            graph.add_node(name, node)
        graph.add_edge(START, "validate")
        graph.add_edge("validate", "quality")
        graph.add_edge("quality", "analysis")
        graph.add_edge("analysis", "verification")
        graph.add_edge("verification", "final")
        graph.add_edge("final", END)
        return graph

    def _build_prepare(self) -> StateGraph:
        graph = StateGraph(WorkflowState)
        nodes = self._common_nodes()
        for name, node in nodes.items():
            graph.add_node(name, node)

        def plan(state: WorkflowState) -> dict[str, Any]:
            frame = load_dataframe(Path(state["source_path"]))
            output = preparation_agent(
                state.get("preparation_config", {}),
                state["quality"],
                self.provider,
                frame,
            )
            return {
                "stage": "preparation_plan",
                "stage_status": {**state.get("stage_status", {}), "preparation_plan": "completed"},
                "preparation_plan": output.model_dump(),
                "preparation_agent": output.model_dump(),
                "feature_decisions": output.decisions,
            }

        def approval(state: WorkflowState) -> dict[str, Any]:
            plan_value = state.get("preparation_plan", {})
            if not plan_value.get("steps"):
                return {
                    "approval": "not_required",
                    "stage_status": {**state.get("stage_status", {}), "approval": "skipped"},
                }
            decision = interrupt({"type": "preparation_approval", "plan": plan_value})
            decision_value = (
                decision if isinstance(decision, str) else decision.get("decision", "rejected")
            )
            if decision_value not in {"approved", "rejected"}:
                return {
                    "approval": "rejected",
                    "stage_status": {**state.get("stage_status", {}), "approval": "rejected"},
                }
            return {
                "approval": decision_value,
                "stage_status": {**state.get("stage_status", {}), "approval": decision_value},
            }

        def route_approval(state: WorkflowState) -> str:
            return (
                "transformations"
                if state.get("approval") in {"approved", "not_required"}
                else "final"
            )

        def transformations(state: WorkflowState) -> dict[str, Any]:
            source = load_dataframe(Path(state["source_path"]))
            working, records = prepare_dataframe(
                source, _config_from_plan(state.get("preparation_plan", {}))
            )
            path = self.settings.prepared_dir / f"{state['run_id']}.csv"
            working.to_csv(path, index=False)
            return {
                "stage": "transformations",
                "stage_status": {**state.get("stage_status", {}), "transformations": "completed"},
                "transformations": records,
                "working_path": str(path),
            }

        def reprofile(state: WorkflowState) -> dict[str, Any]:
            return {
                "stage": "reprofile",
                "stage_status": {**state.get("stage_status", {}), "reprofile": "completed"},
                "profile": profile_dataframe(load_dataframe(Path(state["working_path"]))),
            }

        graph.add_node("plan", plan)
        graph.add_node("approval", approval)
        graph.add_node("transformations", transformations)
        graph.add_node("reprofile", reprofile)
        graph.add_edge(START, "validate")
        graph.add_edge("validate", "quality")
        graph.add_edge("quality", "plan")
        graph.add_edge("plan", "approval")
        graph.add_conditional_edges(
            "approval", route_approval, {"transformations": "transformations", "final": "final"}
        )
        graph.add_edge("transformations", "reprofile")
        graph.add_edge("reprofile", "analysis")
        graph.add_edge("analysis", "verification")
        graph.add_edge("verification", "final")
        graph.add_edge("final", END)
        return graph


def _config_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for step in plan.get("steps", []):
        operation = step["operation"]
        parameters = step.get("parameters", {})
        if operation == "drop_duplicates":
            config["drop_duplicates"] = True
            continue
        key = {
            "missing_indicator": "missing_indicators",
            "ratio": "ratios",
        }.get(operation, operation)
        if operation in {"impute_median", "impute_mode", "impute_mean"}:
            config.setdefault(key, []).append(parameters.get("column"))
        elif operation in {"impute_constant", "impute_forward_fill", "impute_backward_fill"}:
            config.setdefault(key, []).append(parameters)
        elif operation in {
            "text_features",
            "one_hot_encode",
            "standard_scale",
            "minmax_scale",
            "robust_scale",
            "missing_indicator",
        }:
            config.setdefault(key, []).append(
                parameters.get("column") if "column" in parameters else parameters
            )
        elif operation == "date_features":
            config.setdefault(key, []).append(parameters)
        elif operation in {"ratio"}:
            config.setdefault(key, []).append(parameters)
    return config


def resume_graph(graphs: DatumGraphs, run_id: str, mode: str, decision: str) -> dict[str, Any]:
    compiled = graphs.prepare_for_ml if mode == "prepare_for_ml" else graphs.analyze_only
    result = compiled.invoke(Command(resume=decision), config=graphs.config(run_id))
    return result
