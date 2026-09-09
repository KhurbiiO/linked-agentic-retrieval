"""Pluggable scoring strategies for discovered link candidates."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class CandidateScore:
    total: float
    components: dict[str, float]


class CandidateScorer(ABC):
    """Strategy interface for ranking links and extracted evidence."""

    @abstractmethod
    def score(
        self,
        *,
        url: str,
        json_path: str,
        context: dict[str, str],
        goal: str,
        search_terms: list[str],
    ) -> CandidateScore:
        raise NotImplementedError

    def score_batch(self, items: list[dict]) -> list[CandidateScore]:
        return [self.score(**item) for item in items]

    def score_evidence(
        self,
        *,
        source_url: str,
        json_path: str,
        value: str,
        goal: str,
        search_terms: list[str],
    ) -> CandidateScore:
        """Rank a scalar extraction while preserving lexical-mode behavior."""
        haystack = f"{json_path} {value}".casefold()
        total = float(
            sum(
                haystack.count(term.casefold())
                for term in search_terms
                if term.strip()
            )
        )
        return CandidateScore(total=total, components={"term_frequency": total})

    def score_evidence_batch(
        self,
        *,
        items: list[tuple[str, str]],
        source_url: str,
        goal: str,
        search_terms: list[str],
    ) -> list[CandidateScore]:
        return [
            self.score_evidence(
                source_url=source_url,
                json_path=json_path,
                value=value,
                goal=goal,
                search_terms=search_terms,
            )
            for json_path, value in items
        ]


class TermFrequencyScorer(CandidateScorer):
    """Compatibility scorer using unweighted exact substring counts."""

    def score(self, *, url, json_path, context, goal, search_terms):
        text = f"{json_path} {url} " + " ".join(
            f"{key} {value}" for key, value in context.items()
        )
        total = float(sum(text.casefold().count(term.casefold()) for term in search_terms if term))
        return CandidateScore(total=total, components={"term_frequency": total})


class WeightedContextScorer(CandidateScorer):
    """Goal-aware lexical scorer that favors descriptive nested context."""

    FIELD_WEIGHTS = {
        "name": 4.0,
        "title": 4.0,
        "label": 3.0,
        "heading": 3.0,
        "description": 2.0,
        "text": 1.5,
        "type": 1.0,
    }

    def score(self, *, url, json_path, context, goal, search_terms):
        terms = [term.casefold().strip() for term in search_terms if term.strip()]
        url_text = url.casefold()
        path_text = json_path.casefold()
        context_text = " ".join(f"{key} {value}" for key, value in context.items()).casefold()

        exact_url = float(sum(url_text.count(term) for term in terms))
        exact_path = float(sum(path_text.count(term) for term in terms))
        exact_context = float(sum(context_text.count(term) for term in terms))
        field_match = 0.0
        for key, value in context.items():
            leaf_key = re.split(r"[.\[\]]+", key.casefold())[-1]
            weight = self.FIELD_WEIGHTS.get(leaf_key, 1.0)
            field_match += weight * sum(value.casefold().count(term) for term in terms)

        goal_tokens = self._tokens(goal)
        candidate_tokens = self._tokens(f"{url} {json_path} {context_text}")
        goal_overlap = float(len(goal_tokens & candidate_tokens))
        components = {
            "url_match": exact_url,
            "path_match": exact_path * 0.5,
            "context_match": exact_context * 2.0,
            "weighted_field_match": field_match,
            "goal_token_overlap": goal_overlap * 0.5,
        }
        return CandidateScore(total=round(sum(components.values()), 3), components=components)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[\w-]+", text.casefold()) if len(token) >= 3}


class SemanticScorer(CandidateScorer):
    """Score candidates only by embedding cosine similarity."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self._model = None

    def score(self, *, url, json_path, context, goal, search_terms):
        query = goal.strip() or " ".join(search_terms)
        candidate = f"{url} {json_path} " + " ".join(
            f"{key}: {value}" for key, value in context.items()
        )
        similarity = float(self._encode(query) @ self._encode(candidate))
        similarity = round(similarity, 6)
        return CandidateScore(
            total=similarity,
            components={"semantic_similarity": similarity},
        )

    def score_batch(self, items):
        if not items:
            return []
        queries = [item["goal"].strip() or " ".join(item["search_terms"]) for item in items]
        candidates = [
            f'{item["url"]} {item["json_path"]} '
            + " ".join(f"{key}: {value}" for key, value in item["context"].items())
            for item in items
        ]
        query_embeddings = self._get_model().encode(queries, normalize_embeddings=True)
        candidate_embeddings = self._get_model().encode(candidates, normalize_embeddings=True)
        return [
            CandidateScore(
                total=(similarity := round(float(query @ candidate), 6)),
                components={"semantic_similarity": similarity},
            )
            for query, candidate in zip(query_embeddings, candidate_embeddings)
        ]

    def score_evidence(
        self, *, source_url, json_path, value, goal, search_terms
    ):
        query = goal.strip() or " ".join(search_terms)
        candidate = f"{json_path}: {value}"
        similarity = round(float(self._encode(query) @ self._encode(candidate)), 6)
        return CandidateScore(
            total=similarity,
            components={"semantic_similarity": similarity},
        )

    def score_evidence_batch(
        self, *, items, source_url, goal, search_terms
    ):
        if not items:
            return []
        query = goal.strip() or " ".join(search_terms)
        candidates = [f"{json_path}: {value}" for json_path, value in items]
        query_embedding = self._encode(query)
        candidate_embeddings = self._get_model().encode(
            candidates, normalize_embeddings=True
        )
        return [
            CandidateScore(
                total=(similarity := round(float(query_embedding @ embedding), 6)),
                components={"semantic_similarity": similarity},
            )
            for embedding in candidate_embeddings
        ]

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @lru_cache(maxsize=512)
    def _encode(self, text: str):
        return self._get_model().encode(text, normalize_embeddings=True)


def create_candidate_scorer(
    method: str,
    *,
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> CandidateScorer:
    scorers = {
        "term_frequency": TermFrequencyScorer,
        "weighted_context": WeightedContextScorer,
    }
    if method == "semantic":
        return SemanticScorer(
            model_name=semantic_model_name,
        )
    try:
        return scorers[method]()
    except KeyError as exc:
        raise ValueError(f"Unknown candidate scoring method: {method}") from exc
