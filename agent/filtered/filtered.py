"""Filtered-evidence implementation of the retrieval-agent loop."""

from __future__ import annotations

from typing import Any

from agent.host.host import RetrievalAgent, _create_retrieval_agent
from agent.models import RetrievalResult


class FilteredRetrievalAgent(RetrievalAgent):
    """Supply only score-filtered evidence to the model-driven stages."""

    def _model_evidence(
        self,
        evidence: list[RetrievalResult],
        raw_extractions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [item.model_dump() for item in evidence]


def create_retrieval_agent(*args, **kwargs) -> FilteredRetrievalAgent:
    """Build the default filtered-evidence retrieval agent."""
    return _create_retrieval_agent(
        *args,
        _agent_class=FilteredRetrievalAgent,
        **kwargs,
    )


create_filtered_retrieval_agent = create_retrieval_agent
