"""Algorithm adapters consumed by the framework-independent benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from agents import create_tri_agent


@dataclass
class BenchmarkResponse:
    """Common response returned by every benchmarked algorithm."""

    answer: str
    completed: bool | None = None
    duration_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    stage_metrics: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BenchmarkAlgorithm(Protocol):
    """Minimal interface required by the benchmark pipeline."""

    name: str

    def run(self, instruction: str, start_url: str) -> BenchmarkResponse:
        """Execute one task and return a text answer plus optional telemetry."""
        ...

    def configuration(self) -> dict[str, Any]:
        """Return serialisable configuration recorded in the summary."""
        ...


@dataclass
class FunctionAlgorithm:
    """Adapt any Python callable without coupling it to the tri-agent code.

    The callable may return either a plain answer string or BenchmarkResponse.
    """

    name: str
    function: Callable[[str, str], str | BenchmarkResponse]
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, instruction: str, start_url: str) -> BenchmarkResponse:
        result = self.function(instruction, start_url)
        return result if isinstance(result, BenchmarkResponse) else BenchmarkResponse(answer=result)

    def configuration(self) -> dict[str, Any]:
        return dict(self.config)


@dataclass
class TriAgentAlgorithm:
    """Adapter for the repository's current tri-agent implementation."""

    name: str
    instructor_model: str
    controller_model: str
    builder_model: str
    graph_embedding_model: str = "ollama:nomic-embed-text"
    options: dict[str, Any] = field(default_factory=dict)

    MODEL = "ollama:qwen3.6:27b"
    GRAPH_EMBEDDING_MODEL = "ollama:nomic-embed-text"

    MAX_CONTROLLER_ACTIONS = 3
    CONTROLLER_ACTION_TIMEOUT = 5
    CONTROLLER_NAVIGATION_TIMEOUT = 15
    CONTROLLER_SNAPSHOT_MAX_CHARS = 50_000
    MAX_RETRIEVAL_ROUNDS = 3
    MAX_GRAPH_QUERY_STEPS = 4
    GRAPH_QUERY_RESULT_LIMIT = 8
    GRAPH_NEIGHBOR_LIMIT = 5
    GRAPH_MIN_SCORE = 0.35
    GRAPH_VECTOR_DATABASE_PATH = ":memory:"  # Set a SQLite path to persist fact vectors
    OLLAMA_KEEP_ALIVE = "30m"
    PRELOAD_MODELS = False
    TRACE_PROCESS = False

    def run(self, instruction: str, start_url: str) -> BenchmarkResponse:
        prompt = f"{instruction}\nSeed URL: {start_url}"
        with create_tri_agent(
                model=self.MODEL,
                max_controller_actions=self.MAX_CONTROLLER_ACTIONS,
                controller_action_timeout=self.CONTROLLER_ACTION_TIMEOUT,
                controller_navigation_timeout=self.CONTROLLER_NAVIGATION_TIMEOUT,
                controller_snapshot_max_chars=self.CONTROLLER_SNAPSHOT_MAX_CHARS,
                max_retrieval_rounds=self.MAX_RETRIEVAL_ROUNDS,
                max_graph_query_steps=self.MAX_GRAPH_QUERY_STEPS,
                graph_query_result_limit=self.GRAPH_QUERY_RESULT_LIMIT,
                graph_neighbor_limit=self.GRAPH_NEIGHBOR_LIMIT,
                graph_min_score=self.GRAPH_MIN_SCORE,
                graph_vector_database_path=self.GRAPH_VECTOR_DATABASE_PATH,
                graph_embedding_model=self.GRAPH_EMBEDDING_MODEL,
                preload_models=self.PRELOAD_MODELS,
                ollama_keep_alive=self.OLLAMA_KEEP_ALIVE,
                trace=self.TRACE_PROCESS,
            )  as agent:
            result = agent.invoke(prompt)
        return BenchmarkResponse(
            answer=result.answer or "",
            completed=result.completed,
            duration_ms=result.total_duration_ms,
            input_tokens=sum(metric.input_tokens for metric in result.metrics),
            output_tokens=sum(metric.output_tokens for metric in result.metrics),
            total_tokens=sum(metric.total_tokens for metric in result.metrics),
            stage_metrics=[metric.model_dump() for metric in result.metrics],
        )

    def configuration(self) -> dict[str, Any]:
        return {
            "type": "tri_agent",
            "model": self.MODEL,
            "graph_embedding_model": self.graph_embedding_model,
            "options": self.options,
        }
