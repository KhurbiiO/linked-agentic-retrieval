"""Structured messages exchanged by the three-agent retrieval workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class AriaEvidenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["selected", "no_evidence"] = Field(
        description="Use no_evidence only when none of the candidates contains useful content facts."
    )
    candidate_ids: list[str] = Field(
        max_length=20,
        description="Exact IDs from ranked_candidates; explicitly [] when status is no_evidence.",
    )
    reasoning: str

    @model_validator(mode="after")
    def check_selection(self) -> AriaEvidenceSelection:
        if (self.status == "selected") != bool(self.candidate_ids):
            raise ValueError("selected requires candidate IDs; no_evidence requires an empty list")
        return self


class ControllerResult(BaseModel):
    final_url: str
    final_aria: str
    builder_aria: str
    observations: list[ControllerObservation]
    stopped_reason: str
    filter_status: str = "unknown"


class GraphTriple(BaseModel):
    subject: str = Field(description="Stable absolute IRI for the subject")
    predicate: str = Field(
        description=(
            "Full https://schema.org property IRI, or RDF type IRI "
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        )
    )
    object: str
    object_kind: Literal["literal", "iri"] = "literal"
    semantic_type: Literal["content"] = "content"
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


class GoalVerification(BaseModel):
    sufficient: bool
    answer: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    controller_instruction: str | None = None
    reasoning: str


class TriAgentResult(BaseModel):
    plan: RetrievalPlan
    controller: ControllerResult
    graph: KnowledgeGraph
    verification: GoalVerification
    verification_history: list[GoalVerification]
    answer: str | None = None
    completed: bool
    metrics: list[StageMetric]
    total_duration_ms: float
