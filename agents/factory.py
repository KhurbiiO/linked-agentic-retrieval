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
from utils.aria import AriaPage
from store import RDFKnowledgeGraphStore


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


def _embedding_model(value: EmbeddingInput) -> Embeddings:
    return value if isinstance(value, Embeddings) else init_embeddings(value)


def create_tri_agent(
    model: ModelInput = "ollama:llama3.2",
    *,
    instructor_model: ModelInput | None = None,
    controller_model: ModelInput | None = None,
    builder_model: ModelInput | None = None,
    evidence_embedding_model: EmbeddingInput = "ollama:nomic-embed-text",
    temperature: float = 0,
    max_controller_actions: int = 5,
    controller_action_timeout: float = 5,
    controller_navigation_timeout: float = 15,
    controller_max_evidence_segments: int = 12,
    controller_evidence_candidate_limit: int = 20,
    controller_evidence_min_score: float = 0.08,
    controller_evidence_fallback_blocks: int = 3,
    controller_evidence_context_depth: int = 2,
    controller_evidence_max_chars: int = 16000,
    controller_evidence_block_max_chars: int = 3000,
    controller_evidence_selection_max_chars: int = 24000,
    evidence_query_prefix: str | None = None,
    evidence_document_prefix: str | None = None,
    max_retrieval_rounds: int = 3,
    controller_snapshot_max_chars: int = 30000,
    builder_snapshot_max_chars: int = 60000,
    aria_page: AriaPage | None = None,
    graph_store: RDFKnowledgeGraphStore | None = None,
    trace: bool = False,
    trace_path: str | Path | None = None,
    trace_console: bool = True,
    preload_models: bool = False,
    ollama_keep_alive: str | int | None = "30m",
) -> InstructorAgent:
    """Create the Instructor with its Controller and Builder collaborators."""
    tracer = ProcessTracer(trace, path=trace_path, console=trace_console)
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
    evidence_embeddings = _embedding_model(evidence_embedding_model)
    if preload_models:
        _preload([instructor_llm, controller_llm, builder_llm], tracer)
        embedding_started = perf_counter()
        tracer.emit("models", "embedding_preload_started")
        evidence_embeddings.embed_query("semantic relevance warmup")
        tracer.emit(
            "models",
            "embedding_preload_completed",
            duration_ms=round((perf_counter() - embedding_started) * 1000, 3),
        )

    controller = ControllerAgent(
        controller_llm,
        aria_page=aria_page,
        max_actions=max_controller_actions,
        snapshot_max_chars=controller_snapshot_max_chars,
        tracer=tracer,
        action_timeout=controller_action_timeout,
        navigation_timeout=controller_navigation_timeout,
        max_evidence_segments=controller_max_evidence_segments,
        evidence_candidate_limit=controller_evidence_candidate_limit,
        evidence_min_score=controller_evidence_min_score,
        evidence_fallback_blocks=controller_evidence_fallback_blocks,
        evidence_context_depth=controller_evidence_context_depth,
        evidence_max_chars=min(controller_evidence_max_chars, builder_snapshot_max_chars),
        evidence_block_max_chars=controller_evidence_block_max_chars,
        evidence_selection_max_chars=controller_evidence_selection_max_chars,
        evidence_embeddings=evidence_embeddings,
        evidence_query_prefix=evidence_query_prefix,
        evidence_document_prefix=evidence_document_prefix,
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
