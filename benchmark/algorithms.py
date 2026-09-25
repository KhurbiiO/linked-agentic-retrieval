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

    def run(self, instruction: str, start_url: str) -> BenchmarkResponse:
        prompt = f"{instruction}\nSeed URL: {start_url}"
        with create_tri_agent(
            model=self.instructor_model,
            instructor_model=self.instructor_model,
            controller_model=self.controller_model,
            builder_model=self.builder_model,
            graph_embedding_model=self.graph_embedding_model,
            preload_models=False,
            **self.options,
        ) as agent:
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
            "instructor_model": self.instructor_model,
            "controller_model": self.controller_model,
            "builder_model": self.builder_model,
            "graph_embedding_model": self.graph_embedding_model,
            "options": self.options,
        }
