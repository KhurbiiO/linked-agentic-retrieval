from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal, TypeVar

from agent.models import TraceStep

T = TypeVar("T")


class WorkflowTracer:
    def __init__(
        self,
        sink: Callable[[TraceStep], None] | None = None,
        capture_details: bool = True,
    ) -> None:
        self.steps: list[TraceStep] = []
        self.sink = sink
        self.capture_details = capture_details

    def run(
        self,
        stage: str,
        actor: Literal["agent", "tool"],
        inputs: dict[str, Any],
        operation: Callable[[], T],
        summarize: Callable[[T], tuple[dict[str, Any], dict[str, Any]]],
    ) -> T:
        started_at = datetime.now(timezone.utc).isoformat()
        started = perf_counter()
        try:
            result = operation()
            output, metrics = summarize(result)
            step = TraceStep(
                sequence=len(self.steps) + 1,
                stage=stage,
                actor=actor,
                started_at=started_at,
                duration_ms=round((perf_counter() - started) * 1000, 3),
                status="ok",
                input=inputs if self.capture_details else {},
                output=output if self.capture_details else None,
                metrics=metrics,
            )
        except Exception as exc:
            step = TraceStep(
                sequence=len(self.steps) + 1,
                stage=stage,
                actor=actor,
                started_at=started_at,
                duration_ms=round((perf_counter() - started) * 1000, 3),
                status="error",
                input=inputs if self.capture_details else {},
                error=f"{type(exc).__name__}: {exc}",
            )
            self._append(step)
            raise
        self._append(step)
        return result

    def _append(self, step: TraceStep) -> None:
        self.steps.append(step)
        if self.sink:
            self.sink(step)
