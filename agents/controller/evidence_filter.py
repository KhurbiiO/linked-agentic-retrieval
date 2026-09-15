"""Deterministic ARIA block construction and relevance ranking.

The controller deliberately keeps the final evidence selection deterministic.
This module therefore does three small jobs:

* identify indentation based ARIA subtrees;
* rank those subtrees against one or more retrieval requirements; and
* fit the selected, complete subtrees into the builder's character budget.

The fitting step is intentionally line based.  A selected subtree is either
included in full (together with its requested ancestor context) or omitted;
it is never cut in the middle of an ARIA block.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, replace
from math import sqrt
import re
from typing import Any

from langchain_core.embeddings import Embeddings

ROLE_RE = re.compile(r"^(\s*)-\s+([\w-]+)")
CANDIDATE_ROLES = {
    # Content-bearing roles commonly emitted by browser accessibility trees.
    # Interactive roles are included because their names, values, and states
    # can be evidence (for example, a checked filter or a search textbox).
    "article", "button", "cell", "checkbox", "combobox", "definition",
    "form", "heading", "img", "link", "listitem", "menuitem", "option",
    "paragraph", "radio", "region", "row", "searchbox", "slider", "spinbutton",
    "status", "tab", "tabpanel", "term", "text", "textbox", "treeitem",
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
    # Scores are kept separately so the controller can show the model which
    # requirement(s) a candidate supports.  The default keeps construction
    # backwards compatible with callers that only use the overall score.
    requirement_scores: tuple[float, ...] = ()


class EmbeddingCosineScorer:
    """Rank blocks using embedding cosine similarity to the retrieval query."""

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.query_prefix = query_prefix or ""
        self.document_prefix = document_prefix or ""

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        return self.score_many([query], documents)[0]

    def score_many(
        self,
        queries: Sequence[str],
        documents: list[str],
    ) -> list[list[float]]:
        """Score several queries while embedding the documents only once."""
        if not queries:
            return []
        if not documents:
            return [[] for _ in queries]
        query_vectors = [
            self.embeddings.embed_query(f"{self.query_prefix}{query}")
            for query in queries
        ]
        document_vectors = self.embeddings.embed_documents(
            [f"{self.document_prefix}{document}" for document in documents]
        )
        return [
            [self._cosine(query_vector, document_vector)
             for document_vector in document_vectors]
            for query_vector in query_vectors
        ]

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
    requirements: Sequence[str] | None = None,
    max_candidates: int = 20,
    min_score: float = 0.08,
    context_depth: int = 2,
    block_max_chars: int | None = None,
    page_title: str = "",
    stats: MutableMapping[str, Any] | None = None,
) -> tuple[list[str], list[AriaBlock]]:
    """Parse indentation-based ARIA subtrees and rank them against a query.

    ``requirements`` lets the caller score the same candidate against every
    missing piece of evidence.  The overall candidate score is the largest of
    the goal score and requirement scores, which preserves candidates that are
    highly relevant to at least one requirement.  The individual scores are
    retained on :class:`AriaBlock` for downstream coverage-aware selection.

    ``block_max_chars`` is a hard limit on a candidate's raw subtree.  Large
    parent subtrees are skipped while their smaller candidate descendants can
    still be considered.  This avoids presenting the selector with a block it
    can never fit into the final evidence budget.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    if not -1 <= min_score <= 1:
        raise ValueError("min_score must be between -1 and 1")
    if context_depth < 0:
        raise ValueError("context_depth must be nonnegative")
    if block_max_chars is not None and block_max_chars < 1:
        raise ValueError("block_max_chars must be positive")

    lines = aria.splitlines()
    nodes: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        match = ROLE_RE.match(line)
        if match:
            nodes.append((index, len(match.group(1)), match.group(2).casefold()))

    blocks: list[AriaBlock] = []
    seen_ranges: set[tuple[int, int]] = set()
    skipped_long = 0
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
        if block_max_chars is not None and len(text) > block_max_chars:
            skipped_long += 1
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

    documents = [block.text for block in blocks]
    retrieval_query = query
    if page_title.strip():
        retrieval_query = f"{query}\nPage title: {page_title.strip()}"

    requirement_list = [
        item.strip() for item in (requirements or ()) if item and item.strip()
    ]
    queries = [retrieval_query, *requirement_list]
    if hasattr(scorer, "score_many"):
        score_rows = scorer.score_many(queries, documents)
    else:  # pragma: no cover - compatibility with small custom test scorers
        score_rows = [scorer.score(item, documents) for item in queries]
    overall_scores = score_rows[0] if score_rows else []
    requirement_scores = score_rows[1:]

    # A scorer should return one score per document.  Failing early here makes
    # a misconfigured embedding adapter much easier to diagnose than a silent
    # truncation caused by ``zip``.
    if len(overall_scores) != len(blocks):
        raise ValueError("Embedding scorer returned the wrong number of scores")
    if any(len(scores) != len(blocks) for scores in requirement_scores):
        raise ValueError("Requirement scorer returned the wrong number of scores")

    scored_blocks: list[AriaBlock] = []
    for index, block in enumerate(blocks):
        per_requirement = tuple(
            round(scores[index], 6) for scores in requirement_scores
        )
        score = max((overall_scores[index], *per_requirement), default=0.0)
        scored_blocks.append(replace(
            block,
            score=round(score, 6),
            requirement_scores=per_requirement,
        ))

    blocks = scored_blocks
    eligible = [block for block in blocks if block.score >= min_score]
    if not eligible:
        eligible = blocks
    eligible.sort(key=lambda block: (-block.score, block.start_line))

    if stats is not None:
        stats.update({
            "input_lines": len(lines),
            "parsed_nodes": len(nodes),
            "candidate_blocks": len(blocks),
            "eligible_blocks": len(eligible),
            "skipped_long_blocks": skipped_long,
            "requirements": len(requirement_list),
        })
    return lines, eligible[:max_candidates]


def fit_block(
    lines: list[str],
    block: AriaBlock,
    *,
    max_chars: int,
) -> AriaBlock | None:
    """Return ``block`` when its complete rendering fits ``max_chars``.

    The function is useful for callers that need to check a single candidate.
    It returns ``None`` rather than a clipped block, because partial ARIA
    subtrees can change the meaning of roles, names, and relationships.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if _block_line_indexes(lines, block) and _rendered_char_count(lines, [block]) <= max_chars:
        return block
    return None


def fit_blocks(
    lines: list[str],
    blocks: Sequence[AriaBlock],
    *,
    max_chars: int,
) -> list[AriaBlock]:
    """Keep complete selected blocks within a shared character budget.

    Blocks are considered in the supplied order, which is the ranking/coverage
    order produced by :func:`rank_aria_blocks`.  Ancestor and overlapping lines
    are counted once, matching :func:`render_blocks`.  If adding a block would
    exceed the budget, that block is skipped and later blocks are still tried.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    fitted: list[AriaBlock] = []
    indexes: set[int] = set()
    seen_ids: set[str] = set()
    for block in blocks:
        if block.id in seen_ids:
            continue
        seen_ids.add(block.id)
        block_indexes = _block_line_indexes(lines, block)
        if not block_indexes:
            continue
        candidate_indexes = indexes | block_indexes
        if _char_count_for_indexes(lines, candidate_indexes) <= max_chars:
            fitted.append(block)
            indexes = candidate_indexes
    return fitted


def render_blocks(
    lines: list[str],
    blocks: list[AriaBlock],
    *,
    max_chars: int,
) -> str:
    """Render selected raw blocks with ancestor lines, deduplicated verbatim."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    indexes: set[int] = set()
    for block in blocks:
        indexes.update(_block_line_indexes(lines, block))
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


def _block_line_indexes(lines: list[str], block: AriaBlock) -> set[int]:
    """Return valid zero-based lines for a block and its ancestor context."""
    if not lines:
        return set()
    indexes = {
        line - 1 for line in block.ancestor_lines
        if 1 <= line <= len(lines)
    }
    start = max(1, block.start_line)
    end = min(len(lines), block.end_line)
    if start <= end:
        indexes.update(range(start - 1, end))
    return indexes


def _char_count_for_indexes(lines: list[str], indexes: set[int]) -> int:
    """Count a rendered line set exactly as :func:`render_blocks` does."""
    size = 0
    for index in sorted(indexes):
        if not 0 <= index < len(lines):
            continue
        size += len(lines[index]) + (1 if size else 0)
    return size


def _rendered_char_count(lines: list[str], blocks: Sequence[AriaBlock]) -> int:
    return _char_count_for_indexes(
        lines,
        {
            index
            for block in blocks
            for index in _block_line_indexes(lines, block)
        },
    )


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
