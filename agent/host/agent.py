"""Single reasoning agent with a traced web-retrieval loop."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from agent.models import (
    AgentAnswer,
    CandidateLink,
    LinkMatch,
    PerformanceMetrics,
    QuestionAnalysis,
    RetrievalInstruction,
    RetrievalResult,
    TraceStep,
    Verification,
)
from agent.models_factory import ModelInput, create_chat_model
from agent.tracing import WorkflowTracer
from tools.extract.webex import StructuredDataExtractor


ANALYSIS_PROMPT = """You are the analysis stage of a web retrieval host.
Separate the request into its goal, conversational context, retrieval terms,
starting HTTP(S) URLs, and observable success criteria. Preserve important terms
verbatim and add useful synonyms. Never invent a URL; use only URLs present in
the request or conversation context.
"""

INSTRUCTION_PROMPT = """You are choosing the next action in your retrieval loop.
Your extraction tool can retrieve exactly one web page
per turn. Produce the next bounded instruction. Select target_url only from the
provided candidate URLs, never from memory. Use prior evidence and verification
to decide which link best advances the goal. Do not revisit a visited URL.
"""

VERIFY_PROMPT = """Verify the retrieved evidence against the original goal and
success criteria. Return complete only when the evidence directly supports an
answer, continue when another candidate link can resolve missing information,
and failed when there is no useful evidence or candidate path. Do not infer facts
that are absent from evidence.
"""

ANSWER_PROMPT = """Answer strictly from verified retrieval evidence. Cite claims as
[URL:json_path]. State missing or inconclusive information plainly. Do not invent
facts, citations, or extraction results.
"""


class RetrievalAgent:
    """One reasoning loop that analyzes, retrieves, verifies, and answers."""

    def __init__(
        self,
        analysis_model: BaseChatModel,
        navigation_model: BaseChatModel,
        verification_model: BaseChatModel,
        answer_model: BaseChatModel,
        extractor: StructuredDataExtractor,
        max_rounds: int = 5,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self.answer_model = answer_model
        self.extractor = extractor
        self.max_rounds = max_rounds
        self.analyzer = analysis_model.with_structured_output(QuestionAnalysis, include_raw=True)
        self.navigator = navigation_model.with_structured_output(RetrievalInstruction, include_raw=True)
        self.verifier = verification_model.with_structured_output(Verification, include_raw=True)

    def invoke(
        self,
        question: str,
        context: Sequence[dict[str, str]] = (),
        trace_sink: Callable[[TraceStep], None] | None = None,
    ) -> AgentAnswer:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        run_started = perf_counter()
        tracer = WorkflowTracer(trace_sink)

        analysis_packet = tracer.run(
            "agent.question_analysis",
            "agent",
            {"question": question, "context": list(context)},
            lambda: self.analyzer.invoke(
                [
                    ("system", ANALYSIS_PROMPT),
                    ("user", json.dumps({"question": question, "context": list(context)})),
                ]
            ),
            self._summarize_structured,
        )
        analysis = self._parsed(analysis_packet, QuestionAnalysis)
        if not analysis.seed_urls:
            raise ValueError("The request or its context must include a starting HTTP(S) URL")

        evidence: list[RetrievalResult] = []
        verifications: list[Verification] = []
        visited: set[str] = set()
        candidate_urls = list(dict.fromkeys(analysis.seed_urls))

        for round_number in range(1, self.max_rounds + 1):
            instruction_packet = tracer.run(
                "agent.select_action",
                "agent",
                {
                    "round": round_number,
                    "analysis": analysis.model_dump(),
                    "candidate_urls": candidate_urls,
                    "visited_urls": sorted(visited),
                    "previous_verification": (
                        verifications[-1].model_dump() if verifications else None
                    ),
                },
                lambda: self.navigator.invoke(
                    [
                        ("system", INSTRUCTION_PROMPT),
                        (
                            "user",
                            json.dumps(
                                {
                                    "analysis": analysis.model_dump(),
                                    "candidate_urls": candidate_urls,
                                    "visited_urls": sorted(visited),
                                    "evidence": [item.model_dump() for item in evidence],
                                    "previous_verification": (
                                        verifications[-1].model_dump() if verifications else None
                                    ),
                                }
                            ),
                        ),
                    ]
                ),
                self._summarize_structured,
            )
            instruction = self._parsed(instruction_packet, RetrievalInstruction)
            if instruction.target_url not in candidate_urls or instruction.target_url in visited:
                raise ValueError("Host selected a URL outside the allowed unvisited candidates")

            extracted = tracer.run(
                "tool.extract_url",
                "tool",
                {"round": round_number, "instruction": instruction.model_dump()},
                lambda: self.extractor.extract(instruction.target_url),
                self._summarize_extraction,
            )
            visited.add(instruction.target_url)
            visited.add(str(extracted.get("url", instruction.target_url)))

            matches = tracer.run(
                "tool.traverse_data",
                "tool",
                {"round": round_number, "url": instruction.target_url, "terms": instruction.search_terms},
                lambda: [
                    LinkMatch.model_validate(item)
                    for item in self.extractor.traverse(
                        extracted, instruction.search_terms, instruction.max_results
                    )
                ],
                lambda result: ({"matches": [item.model_dump() for item in result]}, {"match_count": len(result)}),
            )
            links = tracer.run(
                "tool.discover_links",
                "tool",
                {"round": round_number, "url": instruction.target_url},
                lambda: [
                    CandidateLink.model_validate(item)
                    for item in self.extractor.discover_links(
                        extracted, instruction.search_terms, instruction.max_links
                    )
                ],
                lambda result: ({"links": [item.model_dump() for item in result]}, {"link_count": len(result)}),
            )
            result = RetrievalResult(
                objective=instruction.objective,
                requested_url=instruction.target_url,
                final_url=str(extracted.get("url", instruction.target_url)),
                matches=matches,
                candidate_links=links,
            )
            evidence.append(result)
            candidate_urls = list(
                dict.fromkeys(
                    [url for url in candidate_urls if url not in visited]
                    + [item.url for item in links if item.url not in visited]
                )
            )

            verification_packet = tracer.run(
                "agent.verify_evidence",
                "agent",
                {"round": round_number, "analysis": analysis.model_dump(), "result": result.model_dump()},
                lambda: self.verifier.invoke(
                    [
                        ("system", VERIFY_PROMPT),
                        (
                            "user",
                            json.dumps(
                                {
                                    "analysis": analysis.model_dump(),
                                    "all_evidence": [item.model_dump() for item in evidence],
                                    "remaining_candidate_urls": candidate_urls,
                                }
                            ),
                        ),
                    ]
                ),
                self._summarize_structured,
            )
            verification = self._parsed(verification_packet, Verification)
            verifications.append(verification)
            if verification.decision != "continue" or not candidate_urls:
                break

        answer_message = tracer.run(
            "agent.synthesize_answer",
            "agent",
            {"question": question, "verification": verifications[-1].model_dump(), "rounds": len(evidence)},
            lambda: self.answer_model.invoke(
                [
                    ("system", ANSWER_PROMPT),
                    (
                        "user",
                        json.dumps(
                            {
                                "question": question,
                                "analysis": analysis.model_dump(),
                                "evidence": [item.model_dump() for item in evidence],
                                "verifications": [item.model_dump() for item in verifications],
                            }
                        ),
                    ),
                ]
            ),
            self._summarize_message,
        )
        performance = self._performance_metrics(
            tracer.steps,
            total_duration_ms=round((perf_counter() - run_started) * 1000, 3),
            reasoning_rounds=len(evidence),
            urls_visited=len(visited),
        )
        return AgentAnswer(
            answer=answer_message.text,
            analysis=analysis,
            evidence=evidence,
            verifications=verifications,
            trace=tracer.steps,
            performance=performance,
        )

    @staticmethod
    def _performance_metrics(
        steps: list[TraceStep],
        total_duration_ms: float,
        reasoning_rounds: int,
        urls_visited: int,
    ) -> PerformanceMetrics:
        input_tokens = sum(int(step.metrics.get("input_tokens", 0)) for step in steps)
        output_tokens = sum(int(step.metrics.get("output_tokens", 0)) for step in steps)
        reported_total = sum(int(step.metrics.get("total_tokens", 0)) for step in steps)
        return PerformanceMetrics(
            total_duration_ms=total_duration_ms,
            model_duration_ms=round(sum(step.duration_ms for step in steps if step.actor == "agent"), 3),
            tool_duration_ms=round(sum(step.duration_ms for step in steps if step.actor == "tool"), 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=reported_total or input_tokens + output_tokens,
            reasoning_rounds=reasoning_rounds,
            urls_visited=urls_visited,
            successful_steps=sum(step.status == "ok" for step in steps),
            failed_steps=sum(step.status == "error" for step in steps),
        )

    @staticmethod
    def _parsed(packet: dict[str, Any], expected_type: type[Any]) -> Any:
        if packet.get("parsing_error") is not None:
            raise ValueError(f"Structured output parsing failed: {packet['parsing_error']}")
        parsed = packet.get("parsed")
        if not isinstance(parsed, expected_type):
            raise TypeError(f"Expected {expected_type.__name__} structured output")
        return parsed

    @staticmethod
    def _summarize_structured(packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        parsed = packet.get("parsed")
        raw = packet.get("raw")
        output = parsed.model_dump() if parsed is not None else {"parsed": None}
        if packet.get("parsing_error") is not None:
            output["parsing_error"] = str(packet["parsing_error"])
        return output, dict(getattr(raw, "usage_metadata", None) or {})

    @staticmethod
    def _summarize_message(message: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"text": message.text}, dict(getattr(message, "usage_metadata", None) or {})

    @staticmethod
    def _summarize_extraction(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        output = {
            "final_url": result.get("url"),
            "standard_types": sorted(result.get("standard", {}).keys()),
            "embedded_json_count": len(result.get("embedded_json", [])),
            "schema_object_count": len(result.get("schema_objects", [])),
            "typed_object_count": len(result.get("all_typed_objects", [])),
        }
        return output, {key: value for key, value in output.items() if key.endswith("_count")}


def create_retrieval_agent(
    model: ModelInput | None = None,
    *,
    analysis_model: ModelInput | None = None,
    navigation_model: ModelInput | None = None,
    verification_model: ModelInput | None = None,
    answer_model: ModelInput | None = None,
    extractor: StructuredDataExtractor | None = None,
    max_rounds: int = 5,
) -> RetrievalAgent:
    """Build one reasoning agent; each reasoning stage may use a different model."""
    load_dotenv()
    overrides = [analysis_model, navigation_model, verification_model, answer_model]
    shared = create_chat_model(model) if any(item is None for item in overrides) else None

    def resolve(value: ModelInput | None) -> BaseChatModel:
        resolved = create_chat_model(value) if value is not None else shared
        if resolved is None:
            raise ValueError("Every host model stage must be configured")
        return resolved

    return RetrievalAgent(
        analysis_model=resolve(analysis_model),
        navigation_model=resolve(navigation_model),
        verification_model=resolve(verification_model),
        answer_model=resolve(answer_model),
        extractor=extractor or StructuredDataExtractor(),
        max_rounds=max_rounds,
    )
