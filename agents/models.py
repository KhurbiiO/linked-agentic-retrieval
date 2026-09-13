"""Structured messages exchanged by the three-agent retrieval workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrievalPlan(BaseModel):
    goal: str
    seed_url: str
    context_terms: list[str] = Field(min_length=1)
    controller_objective: str
    success_criteria: list[str] = Field(min_length=1)


class ControllerDecision(BaseModel):
    action: Literal["click", "fill", "press", "snapshot", "back", "stop"]
    reason: str
    selector: str | None = None
    role: str | None = None
    name: str | None = None
    value: str | None = None


class ControllerObservation(BaseModel):
    sequence: int
    action: ControllerDecision
    url: str
    title: str
    aria: str
    roles: list[str] = Field(default_factory=list)
    properties: dict[str, list[str]] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float


class ControllerResult(BaseModel):
    final_url: str
    final_aria: str
    observations: list[ControllerObservation]
    stopped_reason: str


class GraphTriple(BaseModel):
    subject: str
    predicate: str
    object: str
    semantic_type: Literal["content", "layout"]
    evidence: str


class KnowledgeGraph(BaseModel):
    triples: list[GraphTriple] = Field(default_factory=list)
    summary: str
    unresolved: list[str] = Field(default_factory=list)


class StageMetric(BaseModel):
    stage: str
    duration_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class TriAgentResult(BaseModel):
    plan: RetrievalPlan
    controller: ControllerResult
    graph: KnowledgeGraph
    metrics: list[StageMetric]
    total_duration_ms: float
