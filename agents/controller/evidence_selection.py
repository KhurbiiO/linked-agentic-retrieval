"""Validate structured model selections without confusing empty evidence with errors."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from pydantic import ValidationError

from agents.models import AriaEvidenceSelection


@dataclass
class SelectionResult:
    status: str
    candidate_ids: list[str] = field(default_factory=list)
    requested_ids: Any = None
    invalid_ids: list[str] = field(default_factory=list)
    raw_selection: Any = None
    error: str | None = None
    reasoning: str = ""


def validate_selection(response: Any, allowed_ids: set[str], limit: int) -> SelectionResult:
    """Validate the include_raw response from LangChain's structured wrapper."""
    raw = response.get("raw") if isinstance(response, dict) else None
    payload = _payload(raw)
    parsed = response.get("parsed") if isinstance(response, dict) else response
    if payload is None and isinstance(parsed, AriaEvidenceSelection):
        payload = parsed.model_dump()
    elif payload is None and isinstance(parsed, dict):
        payload = parsed
    result = SelectionResult(
        status="invalid_selection",
        requested_ids=payload.get("candidate_ids") if isinstance(payload, dict) else None,
        raw_selection=payload,
    )
    parsing_error = response.get("parsing_error") if isinstance(response, dict) else None
    if parsing_error is not None:
        result.error = str(parsing_error)
        return result
    try:
        selection = AriaEvidenceSelection.model_validate(parsed)
    except (ValidationError, TypeError, ValueError) as exc:
        result.error = str(exc)
        return result
    result.requested_ids = selection.candidate_ids
    result.reasoning = selection.reasoning
    result.invalid_ids = [item for item in selection.candidate_ids if item not in allowed_ids]
    if result.invalid_ids:
        result.error = "Selection includes IDs that were not supplied as candidates"
        return result
    # Repeated IDs do not consume the configured selection budget.
    ids = list(dict.fromkeys(selection.candidate_ids))
    if len(ids) > limit:
        result.error = f"Selection exceeds maximum_segments={limit}"
        return result
    result.status = selection.status
    result.candidate_ids = ids
    return result


def _payload(raw: Any) -> Any:
    if raw is None:
        return None
    calls = getattr(raw, "tool_calls", None)
    if calls:
        return calls[0].get("args")
    content = getattr(raw, "content", raw)
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    return content
