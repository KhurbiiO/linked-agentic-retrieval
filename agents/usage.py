"""Provider-reported chat-model token usage, grouped by workflow stage."""

from __future__ import annotations

from threading import Lock
from time import perf_counter

from langchain_core.callbacks import BaseCallbackHandler

from agents.models import StageMetric


def _counts(data: dict | None) -> tuple[int, int, int] | None:
    if not isinstance(data, dict):
        return None
    input_tokens = data.get("input_tokens", data.get("prompt_tokens", data.get("prompt_eval_count")))
    output_tokens = data.get("output_tokens", data.get("completion_tokens", data.get("eval_count")))
    total_tokens = data.get("total_tokens")
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    incoming = int(input_tokens or 0)
    outgoing = int(output_tokens or 0)
    return incoming, outgoing, int(total_tokens or incoming + outgoing)


class ModelUsageTracker(BaseCallbackHandler):
    """Record actual usage returned by chat providers; never estimate tokens."""

    def __init__(self) -> None:
        self._events: list[tuple[float, tuple[int, int, int]]] = []
        self._lock = Lock()

    def on_llm_end(self, response, **kwargs) -> None:
        usage: list[tuple[int, int, int]] = []
        for batch in response.generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                if message is None:
                    continue
                counts = _counts(getattr(message, "usage_metadata", None))
                if counts is None:
                    metadata = getattr(message, "response_metadata", None) or {}
                    counts = _counts(metadata.get("token_usage")) or _counts(metadata)
                if counts is not None:
                    usage.append(counts)
        if not usage:
            output = response.llm_output or {}
            counts = _counts(output.get("token_usage")) or _counts(output)
            if counts is not None:
                usage.append(counts)
        if usage:
            totals = tuple(sum(item[index] for item in usage) for index in range(3))
            with self._lock:
                self._events.append((perf_counter(), totals))

    def metric(self, stage: str, started: float) -> StageMetric:
        ended = perf_counter()
        with self._lock:
            usage = [counts for timestamp, counts in self._events if started <= timestamp <= ended]
        totals = [sum(item[index] for item in usage) for index in range(3)]
        return StageMetric(
            stage=stage,
            duration_ms=round((ended - started) * 1000, 3),
            input_tokens=totals[0],
            output_tokens=totals[1],
            total_tokens=totals[2],
        )
