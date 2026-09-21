"""Structured messages exchanged by the three-agent retrieval workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class NavigationGoal(BaseModel):
    goal: str = Field(description="Concrete page or route to find, not a content fact")
    priority: int = Field(default=3, ge=1, le=5, description="1 is highest priority")
    source: Literal["instructor", "controller"] = "instructor"


class RetrievalPlan(BaseModel):
    goal: str
    seed_url: str
    context_terms: list[str] = Field(min_length=1)
    controller_objective: str
    navigation_goals: list[NavigationGoal] = Field(default_factory=list)
    success_criteria: list[str] = Field(min_length=1)


class ControllerDecision(BaseModel):
    action: Literal["goto", "back", "stop"]
    reason: str
    url: str | None = None
    new_navigation_goal: NavigationGoal | None = None
    completed_navigation_goal_indices: list[int] = Field(default_factory=list)


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
    builder_aria: str
    observations: list[ControllerObservation]
    stopped_reason: str
    navigation_stopped: bool = False
    filter_status: str = "unknown"
    navigation_goals: list[NavigationGoal] = Field(default_factory=list)
    completed_navigation_goal_indices: list[int] = Field(default_factory=list)


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


class GroundedAnswer(BaseModel):
    answer: str = Field(
        description="Direct answer grounded only in the accumulated discovered facts"
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Requested information that could not be supported by discovered evidence",
    )


class GraphQueryStep(BaseModel):
    action: Literal["query", "answer"]
    query: str | None = None
    answer: str | None = None
    limitations: list[str] = Field(default_factory=list)
    reasoning: str
class TriAgentResult(BaseModel):
    plan: RetrievalPlan
    controller: ControllerResult | None = None
    graph: KnowledgeGraph
    verification: GoalVerification
    verification_history: list[GoalVerification]
    answer: str | None = None
    completed: bool
    metrics: list[StageMetric]
    graph_queries: list[dict] = Field(default_factory=list)
    total_duration_ms: float
