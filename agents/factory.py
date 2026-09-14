"""Factory for the three-agent ARIA retrieval workflow."""

from __future__ import annotations

from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from agents.builder import BuilderAgent
from agents.controller import ControllerAgent
from agents.instructor import InstructorAgent
from agents.tracing import ProcessTracer
from utils.aria import AriaPage
from store import RDFKnowledgeGraphStore


ModelInput = str | BaseChatModel


def _model(value: ModelInput, temperature: float) -> BaseChatModel:
    if isinstance(value, BaseChatModel):
        return value
    return init_chat_model(value, temperature=temperature)


def create_tri_agent(
    model: ModelInput = "ollama:llama3.2",
    *,
    instructor_model: ModelInput | None = None,
    controller_model: ModelInput | None = None,
    builder_model: ModelInput | None = None,
    temperature: float = 0,
    max_controller_actions: int = 5,
    max_retrieval_rounds: int = 3,
    controller_snapshot_max_chars: int = 30000,
    builder_snapshot_max_chars: int = 60000,
    aria_page: AriaPage | None = None,
    graph_store: RDFKnowledgeGraphStore | None = None,
    trace: bool = False,
    trace_path: str | Path | None = None,
    trace_console: bool = True,
) -> InstructorAgent:
    """Create the Instructor with its Controller and Builder collaborators."""
    shared = _model(model, temperature)
    instructor_llm = _model(instructor_model, temperature) if instructor_model else shared
    controller_llm = _model(controller_model, temperature) if controller_model else shared
    builder_llm = _model(builder_model, temperature) if builder_model else shared
    tracer = ProcessTracer(trace, path=trace_path, console=trace_console)

    controller = ControllerAgent(
        controller_llm,
        aria_page=aria_page,
        max_actions=max_controller_actions,
        snapshot_max_chars=controller_snapshot_max_chars,
        tracer=tracer,
    )
    builder = BuilderAgent(
        builder_llm,
        snapshot_max_chars=builder_snapshot_max_chars,
        graph_store=graph_store,
        tracer=tracer,
    )
    return InstructorAgent(
        instructor_llm,
        controller,
        builder,
        tracer=tracer,
        max_retrieval_rounds=max_retrieval_rounds,
    )
