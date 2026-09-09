"""Retrieval agent variant that exposes complete extractions to its model stages."""

from __future__ import annotations

from typing import Any

from agent.host.host import RetrievalAgent, create_retrieval_agent
from agent.models import RetrievalResult


class FullExtractionAgent(RetrievalAgent):
    """Run the standard loop with complete per-page structured extractions."""

    def _model_evidence(
        self,
        evidence: list[RetrievalResult],
        raw_extractions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "url": extraction.get("url"),
                "extraction": dict(extraction),
                "relevant_evidence": (
                    evidence[index].model_dump() if index < len(evidence) else None
                ),
            }
            for index, extraction in enumerate(raw_extractions)
        ]


def create_full_extraction_agent(*args, **kwargs) -> FullExtractionAgent:
    """Build the full-extraction variant using the standard agent configuration."""
    return create_retrieval_agent(
        *args,
        _agent_class=FullExtractionAgent,
        **kwargs,
    )
