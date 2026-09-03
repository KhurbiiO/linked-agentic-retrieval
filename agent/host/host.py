"""Single reasoning agent with a traced web-retrieval loop."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from requests import RequestException

from config import AppConfig, load_config
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
from tools import StructuredDataExtractor, create_candidate_scorer


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
to decide which link best advances the goal. Pay particular attention to each
candidate link's anchor_text, context, and parent_json_path. Do not revisit a
visited URL.
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
        max_candidate_urls: int = 20,
        max_results_per_page: int = 12,
        max_links_per_page: int = 20,
        traverse_links: bool = True,
        evidence_mode: str = "filtered",
        extraction_prompt_max_chars_per_page: int = 12000,
        trace_enabled: bool = False,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if min(max_candidate_urls, max_results_per_page, max_links_per_page) < 1:
            raise ValueError("Retrieval limits must be at least 1")
        if evidence_mode not in {"filtered", "extraction"}:
            raise ValueError("evidence_mode must be 'filtered' or 'extraction'")
        if extraction_prompt_max_chars_per_page < 1000:
            raise ValueError("extraction_prompt_max_chars_per_page must be at least 1000")
        self.answer_model = answer_model
        self.extractor = extractor
        self.max_rounds = max_rounds
        self.max_candidate_urls = max_candidate_urls
        self.max_results_per_page = max_results_per_page
        self.max_links_per_page = max_links_per_page
        self.traverse_links = traverse_links
        self.evidence_mode = evidence_mode
        self.extraction_prompt_max_chars_per_page = extraction_prompt_max_chars_per_page
        self.trace_enabled = trace_enabled
        self.analyzer = analysis_model.with_structured_output(QuestionAnalysis, include_raw=True)
        self.navigator = navigation_model.with_structured_output(RetrievalInstruction, include_raw=True)
        self.verifier = verification_model.with_structured_output(Verification, include_raw=True)

    def invoke(
        self,
        question: str,
        context: Sequence[dict[str, str]] = (),
        trace_sink: Callable[[TraceStep], None] | None = None,
        trace_enabled: bool | None = None,
    ) -> AgentAnswer:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        run_started = perf_counter()
        debug_trace = self.trace_enabled if trace_enabled is None else trace_enabled
        if trace_sink is not None:
            debug_trace = True
        tracer = WorkflowTracer(
            trace_sink if debug_trace else None,
            capture_details=debug_trace,
        )

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
        raw_extractions: list[dict[str, Any]] = []
        verifications: list[Verification] = []
        visited: set[str] = set()
        failed_urls: dict[str, str] = {}
        candidate_urls = list(dict.fromkeys(analysis.seed_urls))[: self.max_candidate_urls]

        round_number = 0
        while round_number < self.max_rounds and candidate_urls:
            round_number += 1
            instruction_packet = tracer.run(
                "agent.select_action",
                "agent",
                {
                    "round": round_number,
                    "analysis": analysis.model_dump(),
                    "candidate_urls": candidate_urls,
                    "visited_urls": sorted(visited),
                    "failed_urls": failed_urls,
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
                                    "failed_urls": failed_urls,
                                    "evidence": self._model_evidence(evidence, raw_extractions),
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
            allowed_urls = [url for url in candidate_urls if url not in visited]
            if instruction.target_url not in allowed_urls:
                if not allowed_urls:
                    break
                fallback_url = tracer.run(
                    "agent.selection_fallback",
                    "agent",
                    {
                        "round": round_number,
                        "rejected_url": instruction.target_url,
                        "allowed_urls": allowed_urls,
                    },
                    lambda: allowed_urls[0],
                    lambda selected: (
                        {"selected_url": selected},
                        {"fallback_count": 1},
                    ),
                )
                instruction = instruction.model_copy(
                    update={"target_url": fallback_url}
                )
            instruction = instruction.model_copy(
                update={
                    "max_results": self.max_results_per_page,
                    "max_links": self.max_links_per_page,
                }
            )

            try:
                extracted = tracer.run(
                    "tool.extract_url",
                    "tool",
                    {"round": round_number, "instruction": instruction.model_dump()},
                    lambda: self.extractor.extract(instruction.target_url),
                    self._summarize_extraction,
                )
            except RequestException as error:
                failed_urls[instruction.target_url] = f"{type(error).__name__}: {error}"
                visited.add(instruction.target_url)
                candidate_urls = [
                    url for url in candidate_urls if url != instruction.target_url
                ]
                if candidate_urls:
                    # Failed downloads do not consume a successful retrieval round.
                    round_number -= 1
                    continue
                break
            visited.add(instruction.target_url)
            visited.add(str(extracted.get("url", instruction.target_url)))
            raw_extractions.append(extracted)

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
            links: list[CandidateLink] = []
            if self.traverse_links:
                links = tracer.run(
                    "tool.discover_links",
                    "tool",
                    {"round": round_number, "url": instruction.target_url},
                    lambda: [
                        CandidateLink.model_validate(item)
                        for item in self.extractor.discover_links(
                            extracted,
                            instruction.search_terms,
                            instruction.max_links,
                            goal=analysis.goal,
                        )
                    ],
                    lambda result: (
                        {"links": [item.model_dump() for item in result]},
                        {"link_count": len(result)},
                    ),
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
            )[: self.max_candidate_urls]

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
                                    "all_evidence": self._model_evidence(evidence, raw_extractions),
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

        final_verification = (
            verifications[-1].model_dump()
            if verifications
            else {
                "decision": "failed",
                "assessment": "No unvisited candidate URL remained.",
                "satisfied_criteria": [],
                "missing_information": analysis.success_criteria,
            }
        )
        answer_message = tracer.run(
            "agent.synthesize_answer",
            "agent",
            {
                "question": question,
                "verification": final_verification,
                "rounds": len(evidence),
                "failed_urls": failed_urls,
            },
            lambda: self.answer_model.invoke(
                [
                    ("system", ANSWER_PROMPT),
                    (
                        "user",
                        json.dumps(
                            {
                                "question": question,
                                "analysis": analysis.model_dump(),
                                "evidence": self._model_evidence(evidence, raw_extractions),
                                "verifications": [item.model_dump() for item in verifications],
                                "failed_urls": failed_urls,
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
            trace=tracer.steps if debug_trace else [],
            performance=performance,
        )

    def _model_evidence(
        self,
        evidence: list[RetrievalResult],
        raw_extractions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.evidence_mode == "filtered":
            return [item.model_dump() for item in evidence]

        payload = []
        for index, extraction in enumerate(raw_extractions):
            serialized = json.dumps(extraction, ensure_ascii=False, default=str)
            limit = self.extraction_prompt_max_chars_per_page
            payload.append(
                {
                    "url": extraction.get("url"),
                    "extraction": serialized[:limit],
                    "truncated": len(serialized) > limit,
                    "original_char_count": len(serialized),
                    "included_char_count": min(len(serialized), limit),
                    "filtered_evidence": (
                        evidence[index].model_dump() if index < len(evidence) else None
                    ),
                }
            )
        return payload

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
    config: AppConfig | str | Path | None = None,
    analysis_model: ModelInput | None = None,
    navigation_model: ModelInput | None = None,
    verification_model: ModelInput | None = None,
    answer_model: ModelInput | None = None,
    extractor: StructuredDataExtractor | None = None,
    max_rounds: int | None = None,
    max_candidate_urls: int | None = None,
    max_results_per_page: int | None = None,
    max_links_per_page: int | None = None,
    traverse_links: bool | None = None,
    evidence_mode: str | None = None,
    extraction_prompt_max_chars_per_page: int | None = None,
    trace_enabled: bool | None = None,
) -> RetrievalAgent:
    """Build one reasoning agent from config, with optional explicit overrides."""
    load_dotenv()
    settings = config if isinstance(config, AppConfig) else load_config(config)
    overrides = [analysis_model, navigation_model, verification_model, answer_model]
    configured_model = model or os.getenv("AGENT_MODEL") or settings.model.identifier
    shared = (
        create_chat_model(configured_model, temperature=settings.model.temperature)
        if any(item is None for item in overrides)
        else None
    )

    def resolve(value: ModelInput | None) -> BaseChatModel:
        resolved = (
            create_chat_model(value, temperature=settings.model.temperature)
            if value is not None
            else shared
        )
        if resolved is None:
            raise ValueError("Every host model stage must be configured")
        return resolved

    return RetrievalAgent(
        analysis_model=resolve(analysis_model),
        navigation_model=resolve(navigation_model),
        verification_model=resolve(verification_model),
        answer_model=resolve(answer_model),
        extractor=extractor or StructuredDataExtractor(
            timeout=settings.extractor.timeout_seconds,
            link_context_max_fields=settings.extractor.link_context_max_fields,
            link_context_max_chars=settings.extractor.link_context_max_chars,
            link_context_child_depth=settings.extractor.link_context_child_depth,
            candidate_scorer=create_candidate_scorer(
                settings.retrieval.scoring_method,
                semantic_model_name=settings.retrieval.semantic_model_name,
            ),
            excluded_url_extensions=settings.retrieval.excluded_url_extensions,
        ),
        max_rounds=max_rounds if max_rounds is not None else settings.agent.max_rounds,
        max_candidate_urls=(
            max_candidate_urls
            if max_candidate_urls is not None
            else settings.agent.max_candidate_urls
        ),
        max_results_per_page=(
            max_results_per_page
            if max_results_per_page is not None
            else settings.retrieval.max_results_per_page
        ),
        max_links_per_page=(
            max_links_per_page
            if max_links_per_page is not None
            else settings.retrieval.max_links_per_page
        ),
        traverse_links=(
            traverse_links
            if traverse_links is not None
            else settings.retrieval.traverse_links
        ),
        evidence_mode=(
            evidence_mode if evidence_mode is not None else settings.retrieval.evidence_mode
        ),
        extraction_prompt_max_chars_per_page=(
            extraction_prompt_max_chars_per_page
            if extraction_prompt_max_chars_per_page is not None
            else settings.retrieval.extraction_prompt_max_chars_per_page
        ),
        trace_enabled=(
            trace_enabled if trace_enabled is not None else settings.tracing.enabled
        ),
    )
