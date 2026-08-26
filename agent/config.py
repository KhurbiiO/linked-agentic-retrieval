"""Validated application configuration loaded from JSON."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identifier: str = "ollama:qwen3.5:4b"
    temperature: float = Field(default=0, ge=0, le=2)


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rounds: int = Field(default=5, ge=1, le=50)
    max_candidate_urls: int = Field(default=20, ge=1, le=200)


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_results_per_page: int = Field(default=12, ge=1, le=100)
    max_links_per_page: int = Field(default=20, ge=1, le=200)


class ExtractorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=30, gt=0, le=300)
    link_context_max_fields: int = Field(default=12, ge=1, le=50)
    link_context_max_chars: int = Field(default=1000, ge=100, le=10000)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelSettings = Field(default_factory=ModelSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    extractor: ExtractorSettings = Field(default_factory=ExtractorSettings)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate config.json, falling back to typed defaults if absent."""
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        return AppConfig()
    return AppConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
