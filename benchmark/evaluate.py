"""Deterministic evaluation of free-text answers against structured gold data."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any


_FRACTIONS = {
    "½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4",
    "¾": "3/4", "⅕": "1/5", "⅖": "2/5", "⅗": "3/5",
    "⅘": "4/5", "⅙": "1/6", "⅚": "5/6", "⅛": "1/8",
    "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
}


def _normalise(value: Any) -> str:
    text = str(value).casefold()
    for symbol, replacement in _FRACTIONS.items():
        text = text.replace(symbol, f" {replacement} ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("×", " x ").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9/]+", " ", text)
    return " ".join(text.split())


def _leaves(value: Any, path: str = "answer") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [
            leaf
            for key, child in value.items()
            for leaf in _leaves(child, f"{path}.{key}")
        ]
    if isinstance(value, list):
        if not value:
            return [(path, "none")]
        return [
            leaf
            for index, child in enumerate(value)
            for leaf in _leaves(child, f"{path}[{index}]")
        ]
    return [(path, value)]


def _token_f1(expected: str, answer: str) -> float:
    expected_tokens = Counter(expected.split())
    answer_tokens = Counter(answer.split())
    overlap = sum((expected_tokens & answer_tokens).values())
    if not expected_tokens:
        return 1.0
    if not overlap:
        return 0.0
    precision = overlap / sum(answer_tokens.values()) if answer_tokens else 0.0
    recall = overlap / sum(expected_tokens.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _leaf_score(expected: Any, answer: str) -> tuple[bool, float]:
    if isinstance(expected, bool):
        expected_text = "true" if expected else "false"
        aliases = {"true": ("true", "yes", "present"),
                   "false": ("false", "no", "absent", "not")}[expected_text]
        matched = any(re.search(rf"\b{term}\b", answer) for term in aliases)
        return matched, 1.0 if matched else 0.0
    expected_text = _normalise(expected)
    if expected_text and expected_text in answer:
        return True, 1.0
    expected_tokens = expected_text.split()
    if not expected_tokens:
        return True, 1.0
    recall = sum(1 for token in expected_tokens if token in answer.split()) / len(expected_tokens)
    return recall >= 0.8, recall


def evaluate_answer(answer: str | None, gold_answer: Any) -> dict[str, Any]:
    """Return transparent leaf-level coverage metrics for a model answer.

    A leaf passes when its normalised reference text occurs verbatim or at
    least 80% of its reference tokens occur in the answer. The result retains
    every leaf assessment so benchmark failures can be inspected rather than
    hidden behind one aggregate score.
    """
    normalised_answer = _normalise(answer or "")
    assessments = []
    for path, expected in _leaves(gold_answer):
        matched, recall = _leaf_score(expected, normalised_answer)
        assessments.append({
            "path": path,
            "expected": expected,
            "matched": matched,
            "token_recall": round(recall, 4),
        })
    matched_count = sum(item["matched"] for item in assessments)
    leaf_count = len(assessments)
    gold_text = _normalise(" ".join(str(value) for _, value in _leaves(gold_answer)))
    return {
        "passed": bool(assessments) and matched_count == leaf_count,
        "matched_leaves": matched_count,
        "total_leaves": leaf_count,
        "leaf_coverage": round(matched_count / leaf_count, 4) if leaf_count else 0.0,
        "token_f1": round(_token_f1(gold_text, normalised_answer), 4),
        "leaves": assessments,
    }
