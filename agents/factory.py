"""Factory for the three-agent ARIA retrieval workflow."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from agents.builder import BuilderAgent
from agents.controller import ControllerAgent
from agents.instructor import InstructorAgent
from agents.tracing import ProcessTracer
from agents.usage import ModelUsageTracker
from utils.aria import AriaPage
from store import RDFKnowledgeGraphStore
from store.fact_vector_store import FactVectorIndex
from utils.structured_data import StructuredDataExtractor


ModelInput = str | BaseChatModel
EmbeddingInput = str | Embeddings


def _model(
    value: ModelInput,
    temperature: float,
    ollama_keep_alive: str | int | None,
) -> BaseChatModel:
    if isinstance(value, BaseChatModel):
        return value
    options = {"temperature": temperature}
    if value.startswith("ollama:") and ollama_keep_alive is not None:
        options["keep_alive"] = ollama_keep_alive
    return init_chat_model(value, **options)


def _preload(models: list[BaseChatModel], tracer: ProcessTracer) -> None:
    """Warm each distinct provider/model pair with one minimal request."""
    loaded: set[tuple[object, object]] = set()
    for model in models:
        model_name = getattr(model, "model", None) or id(model)
        key = (type(model), str(model_name))
        if key in loaded:
            continue
        loaded.add(key)
        tracer.emit("models", "preload_started", model=str(model_name))
        started = perf_counter()
        model.invoke("Reply with only: OK")
        tracer.emit(
            "models",
            "preload_completed",
            model=str(model_name),
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )


def create_tri_agent(
    model: ModelInput = "ollama:llama3.2",
    *,
    instructor_model: ModelInput | None = None,
    controller_model: ModelInput | None = None,
    builder_model: ModelInput | None = None,
    graph_embedding_model: EmbeddingInput = "ollama:nomic-embed-text",
    temperature: float = 0,
    max_controller_actions: int = 5,
    controller_action_timeout: float = 5,
    controller_navigation_timeout: float = 15,
    max_retrieval_rounds: int = 3,
    max_graph_query_steps: int = 4,
    graph_query_result_limit: int = 12,
    graph_neighbor_limit: int = 12,
    graph_min_score: float = 0.08,
    graph_vector_database_path: str = ":memory:",
    controller_snapshot_max_chars: int = 30000,
    aria_page: AriaPage | None = None,
    graph_store: RDFKnowledgeGraphStore | None = None,
    structured_data_extractor: StructuredDataExtractor | None = None,
    trace: bool = False,
    trace_path: str | Path | None = None,
    trace_console: bool = True,
    preload_models: bool = False,
    ollama_keep_alive: str | int | None = "30m",
) -> InstructorAgent:
    """Create the Instructor with its Controller and Builder collaborators."""
    tracer = ProcessTracer(trace, path=trace_path, console=trace_console)
    usage_tracker = ModelUsageTracker()
    shared = _model(model, temperature, ollama_keep_alive)
    instructor_llm = (
        _model(instructor_model, temperature, ollama_keep_alive)
        if instructor_model else shared
    )
    controller_llm = (
        _model(controller_model, temperature, ollama_keep_alive)
        if controller_model else shared
    )
    builder_llm = (
        _model(builder_model, temperature, ollama_keep_alive)
        if builder_model else shared
    )
    graph_embeddings = (
        graph_embedding_model
        if isinstance(graph_embedding_model, Embeddings)
        else init_embeddings(graph_embedding_model)
    )
    if preload_models:
        _preload([instructor_llm, controller_llm, builder_llm], tracer)
        tracer.emit("models", "graph_embedding_preload_started")
        graph_embeddings.embed_query("graph fact retrieval warmup")
        tracer.emit("models", "graph_embedding_preload_completed")

    controller = ControllerAgent(
        controller_llm,
        aria_page=aria_page,
        max_actions=max_controller_actions,
        snapshot_max_chars=controller_snapshot_max_chars,
        tracer=tracer,
        action_timeout=controller_action_timeout,
        navigation_timeout=controller_navigation_timeout,
        structured_data_extractor=structured_data_extractor,
        usage_tracker=usage_tracker,
    )
    builder = BuilderAgent(
        builder_llm,
        graph_store=graph_store,
        tracer=tracer,
        usage_tracker=usage_tracker,
    )
    return InstructorAgent(
        instructor_llm,
        controller,
        builder,
        tracer=tracer,
        usage_tracker=usage_tracker,
        max_retrieval_rounds=max_retrieval_rounds,
        max_graph_query_steps=max_graph_query_steps,
        graph_query_result_limit=graph_query_result_limit,
        fact_index=FactVectorIndex(
            graph_embeddings, database_path=graph_vector_database_path
        ),
        graph_neighbor_limit=graph_neighbor_limit,
        graph_min_score=graph_min_score,
    )
