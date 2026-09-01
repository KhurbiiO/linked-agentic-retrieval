"""Validated application configuration loaded from the root config.json."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identifier: str = "ollama:llama3.2"
    temperature: float = Field(default=0, ge=0, le=2)


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rounds: int = Field(default=5, ge=1, le=50)
    max_candidate_urls: int = Field(default=20, ge=1, le=200)


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_results_per_page: int = Field(default=12, ge=1, le=100)
    max_links_per_page: int = Field(default=20, ge=1, le=200)
    scoring_method: Literal["weighted_context", "term_frequency"] = "weighted_context"
    traverse_links: bool = True
    evidence_mode: Literal["filtered", "extraction"] = "filtered"
    extraction_prompt_max_chars_per_page: int = Field(default=12000, ge=1000, le=200000)
    excluded_url_extensions: list[str] = Field(
        default_factory=lambda: [
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".avif", ".ico",
            ".mp4", ".webm", ".mov", ".avi", ".mkv", ".m3u8",
            ".mp3", ".wav", ".ogg", ".m4a", ".aac",
            ".css", ".js", ".map", ".woff", ".woff2", ".ttf", ".eot",
        ]
    )


class ExtractorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=30, gt=0, le=300)
    link_context_max_fields: int = Field(default=12, ge=1, le=50)
    link_context_max_chars: int = Field(default=1000, ge=100, le=10000)
    link_context_child_depth: int = Field(default=2, ge=0, le=6)


class TracingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelSettings = Field(default_factory=ModelSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    extractor: ExtractorSettings = Field(default_factory=ExtractorSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.json")


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate config.json, falling back to typed defaults if absent."""
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        return AppConfig()
    return AppConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
