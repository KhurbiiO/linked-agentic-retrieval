from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QuestionAnalysis(BaseModel):
    """Host interpretation of a request before retrieval starts."""

    goal: str
    context: str = Field(description="Relevant context, constraints, and assumptions")
    terms: list[str] = Field(min_length=1, description="Terms and entities to retrieve")
    seed_urls: list[str] = Field(min_length=1, description="HTTP(S) URLs from the request")
    success_criteria: list[str] = Field(min_length=1)


class RetrievalInstruction(BaseModel):
    """One bounded URL traversal action selected by the reasoning loop."""

    objective: str
    target_url: str
    search_terms: list[str] = Field(min_length=1)
    max_results: int = Field(default=12, ge=1, le=30)
    max_links: int = Field(default=20, ge=1, le=50)


class LinkMatch(BaseModel):
    source_url: str
    json_path: str
    value: str
    score: int


class CandidateLink(BaseModel):
    url: str
    json_path: str
    parent_json_path: str
    anchor_text: str | None = None
    context: dict[str, str] = Field(default_factory=dict)
    score: float = 0
    score_components: dict[str, float] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    objective: str
    requested_url: str
    final_url: str
    matches: list[LinkMatch]
    candidate_links: list[CandidateLink]


class Verification(BaseModel):
    """Reasoning-loop assessment after one retrieval turn."""

    decision: Literal["complete", "continue", "failed"]
    assessment: str
    satisfied_criteria: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class TraceStep(BaseModel):
    """One independently measurable operation in an agent run."""

    sequence: int
    stage: str
    actor: Literal["agent", "tool"]
    started_at: str
    duration_ms: float
    status: Literal["ok", "error"]
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    error: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class PerformanceMetrics(BaseModel):
    total_duration_ms: float
    model_duration_ms: float
    tool_duration_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_rounds: int
    urls_visited: int
    successful_steps: int
    failed_steps: int


class AgentAnswer(BaseModel):
    answer: str
    analysis: QuestionAnalysis
    evidence: list[RetrievalResult]
    verifications: list[Verification]
    trace: list[TraceStep]
    performance: PerformanceMetrics
