"""Deterministic ARIA block construction and relevance ranking."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
import re

from langchain_core.embeddings import Embeddings

ROLE_RE = re.compile(r"^(\s*)-\s+([\w-]+)")
CANDIDATE_ROLES = {
    "article", "cell", "definition", "heading", "img", "link", "listitem",
    "paragraph", "region", "row", "term", "text",
}


@dataclass(frozen=True, slots=True)
class AriaBlock:
    id: str
    start_line: int
    end_line: int
    role: str
    text: str
    score: float
    ancestor_lines: tuple[int, ...] = ()


class EmbeddingCosineScorer:
    """Rank blocks using embedding cosine similarity to the retrieval query."""

    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        query_vector = self.embeddings.embed_query(query)
        vectors = self.embeddings.embed_documents(documents)
        return [self._cosine(query_vector, vector) for vector in vectors]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding vectors must have equal dimensions")
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def rank_aria_blocks(
    aria: str,
    query: str,
    scorer: EmbeddingCosineScorer,
    *,
    max_candidates: int = 20,
    min_score: float = 0.08,
    context_depth: int = 2,
) -> tuple[list[str], list[AriaBlock]]:
    """Parse indentation-based ARIA subtrees and rank them against a query."""
    lines = aria.splitlines()
    nodes: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        match = ROLE_RE.match(line)
        if match:
            nodes.append((index, len(match.group(1)), match.group(2).casefold()))

    blocks: list[AriaBlock] = []
    seen_ranges: set[tuple[int, int]] = set()
    for node_index, (start, indent, role) in enumerate(nodes):
        end = len(lines) - 1
        for later_start, later_indent, _ in nodes[node_index + 1:]:
            if later_indent <= indent:
                end = later_start - 1
                break
        if role not in CANDIDATE_ROLES or (start, end) in seen_ranges:
            continue
        if end - start + 1 > 120:
            continue
        seen_ranges.add((start, end))
        text = "\n".join(lines[start:end + 1]).strip()
        if not text:
            continue
        ancestors = _ancestor_lines(nodes[:node_index], indent, context_depth)
        blocks.append(AriaBlock(
            id=f"b{len(blocks) + 1}",
            start_line=start + 1,
            end_line=end + 1,
            role=role,
            text=text,
            score=0.0,
            ancestor_lines=ancestors,
        ))

    scores = scorer.score(query, [block.text for block in blocks])
    blocks = [
        replace(block, score=round(score, 6))
        for block, score in zip(blocks, scores)
    ]
    eligible = [block for block in blocks if block.score >= min_score]
    if not eligible:
        eligible = blocks
    eligible.sort(key=lambda block: (-block.score, block.start_line))
    return lines, eligible[:max_candidates]


def render_blocks(
    lines: list[str],
    blocks: list[AriaBlock],
    *,
    max_chars: int,
) -> str:
    """Render selected raw blocks with ancestor lines, deduplicated verbatim."""
    indexes: set[int] = set()
    for block in blocks:
        indexes.update(line - 1 for line in block.ancestor_lines)
        indexes.update(range(block.start_line - 1, block.end_line))
    output: list[str] = []
    size = 0
    for index in sorted(indexes):
        line = lines[index]
        added = len(line) + (1 if output else 0)
        if size + added > max_chars:
            break
        output.append(line)
        size += added
    return "\n".join(output)


def _ancestor_lines(
    earlier_nodes: list[tuple[int, int, str]],
    indent: int,
    depth: int,
) -> tuple[int, ...]:
    ancestors: list[int] = []
    ceiling = indent
    for line_index, candidate_indent, _ in reversed(earlier_nodes):
        if candidate_indent < ceiling:
            ancestors.append(line_index + 1)
            ceiling = candidate_indent
            if len(ancestors) == depth:
                break
    return tuple(reversed(ancestors))
