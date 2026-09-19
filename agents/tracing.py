"""Local, optional tracing for the tri-agent workflow."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any


class ProcessTracer:
    """Collect timestamped process events and optionally write JSON Lines."""

    def __init__(
        self,
        enabled: bool = False,
        *,
        path: str | Path | None = None,
        console: bool = True,
    ) -> None:
        self.enabled = enabled
        self.path = Path(path) if path is not None else None
        self.console = console
        self.events: list[dict[str, Any]] = []
        self._started = perf_counter()

    def emit(self, stage: str, event: str, **data: Any) -> None:
        if not self.enabled:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((perf_counter() - self._started) * 1000, 3),
            "stage": stage,
            "event": event,
            "data": data,
        }
        self.events.append(record)
        line = json.dumps(record, ensure_ascii=False, default=str)
        if self.console:
            print(f"[trace] {stage}.{event}: {json.dumps(data, default=str)}")
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as output:
                output.write(line + "\n")
