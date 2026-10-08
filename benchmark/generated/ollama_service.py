"""Health checking and bounded recovery for a local Ollama service."""

from __future__ import annotations

import subprocess
from pathlib import Path
from time import monotonic, sleep

import requests


class OllamaSupervisor:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        start_timeout_seconds: float = 30,
        log_path: str | Path = "output/benchmark/ollama.log",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.start_timeout_seconds = start_timeout_seconds
        self.log_path = Path(log_path)
        self._process: subprocess.Popen | None = None
        self._log_file = None

    def healthy(self, *, timeout_seconds: float = 3) -> bool:
        try:
            return requests.get(
                f"{self.base_url}/api/tags", timeout=max(0.05, timeout_seconds)
            ).ok
        except requests.RequestException:
            return False

    def ensure(self, *, timeout_seconds: float | None = None, restart: bool = False) -> bool:
        """Return True when this call had to start or restart Ollama."""
        allowed = self.start_timeout_seconds
        if timeout_seconds is not None:
            allowed = min(allowed, max(0, timeout_seconds))
        if allowed <= 0:
            raise TimeoutError("No time remains for the Ollama health check")
        if not restart and self.healthy(timeout_seconds=min(3, allowed)):
            return False
        self._stop_owned_process()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log_file is None or self._log_file.closed:
            self._log_file = self.log_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        deadline = monotonic() + allowed
        while monotonic() < deadline:
            remaining = deadline - monotonic()
            if self.healthy(timeout_seconds=min(1, max(0.05, remaining))):
                return True
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"Ollama exited with code {self._process.returncode}; "
                    f"inspect {self.log_path}"
                )
            sleep(0.5)
        raise TimeoutError(
            f"Ollama did not become healthy within {allowed:g} seconds"
        )

    @staticmethod
    def is_connection_refused(error: str | None) -> bool:
        text = (error or "").casefold()
        return any(marker in text for marker in (
            "errno 111",
            "connection refused",
            "connectionrefusederror",
        ))

    def _stop_owned_process(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)

    def close(self) -> None:
        self._stop_owned_process()
        if self._log_file is not None:
            self._log_file.close()

    def __enter__(self) -> OllamaSupervisor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
