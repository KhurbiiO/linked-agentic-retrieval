"""SQLite-backed vector store for RDF facts with bounded graph expansion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import sqrt
import sqlite3
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings
from rdflib import URIRef

if TYPE_CHECKING:
    from .graph_store import RDFKnowledgeGraphStore, StoredEvidence


@dataclass(slots=True)
class FactVector:
    fact: StoredEvidence
    text: str
    vector: list[float]


class FactVectorIndex:
    """Index each distinct statement and retain its RDF subject as graph anchor."""

    def __init__(self, embeddings: Embeddings, *, query_prefix: str = "", document_prefix: str = "", database_path: str = ":memory:") -> None:
        self.embeddings = embeddings
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.entries: dict[tuple[str, str, str], FactVector] = {}
        self.database = sqlite3.connect(database_path)
        self.database.execute(
            "CREATE TABLE IF NOT EXISTS fact_vectors ("
            "subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL, "
            "text TEXT NOT NULL, vector TEXT NOT NULL, "
            "PRIMARY KEY(subject, predicate, object))"
        )

    @staticmethod
    def _key(fact: StoredEvidence) -> tuple[str, str, str]:
        return str(fact.subject), str(fact.predicate), str(fact.object)

    @staticmethod
    def _text(fact: StoredEvidence) -> str:
        predicate = str(fact.predicate).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        return f"Subject: {fact.subject}; {predicate}: {fact.object}"

    def sync(self, facts: tuple[StoredEvidence, ...]) -> int:
        for fact in facts:
            key = self._key(fact)
            if key in self.entries:
                continue
            row = self.database.execute(
                "SELECT text, vector FROM fact_vectors WHERE subject=? AND predicate=? AND object=?",
                key,
            ).fetchone()
            if row is not None:
                self.entries[key] = FactVector(fact, row[0], json.loads(row[1]))
        pending_by_key = {
            self._key(fact): fact for fact in facts
            if self._key(fact) not in self.entries
        }
        pending = list(pending_by_key.values())
        if not pending:
            return 0
        texts = [self._text(fact) for fact in pending]
        vectors = self.embeddings.embed_documents([self.document_prefix + text for text in texts])
        if len(vectors) != len(pending):
            raise ValueError("Embedding model returned the wrong number of fact vectors")
        for fact, text, vector in zip(pending, texts, vectors):
            key = self._key(fact)
            self.entries[key] = FactVector(fact, text, vector)
            self.database.execute(
                "INSERT OR IGNORE INTO fact_vectors VALUES (?, ?, ?, ?, ?)",
                (*key, text, json.dumps(vector)),
            )
        self.database.commit()
        return len(pending)

    def close(self) -> None:
        self.database.close()

    def search(
        self,
        goals: list[str],
        store: RDFKnowledgeGraphStore,
        *,
        entry_limit: int,
        neighbor_limit: int,
        min_score: float,
    ) -> dict:
        """Rank entry facts, then collect bounded same-subject and linked facts."""
        if entry_limit < 1 or neighbor_limit < 0:
            raise ValueError("entry_limit must be positive and neighbor_limit nonnegative")
        indexed = self.sync(store.evidence)
        queries = [goal.strip() for goal in goals if goal.strip()]
        if not queries or not self.entries:
            return {"entries": [], "facts": [], "indexed_count": indexed}
        query_vectors = [self.embeddings.embed_query(self.query_prefix + query) for query in queries]
        ranked = sorted(
            ((max(self._cosine(query, entry.vector) for query in query_vectors), entry)
             for entry in self.entries.values()),
            key=lambda item: item[0], reverse=True,
        )
        seeds = [(score, entry) for score, entry in ranked if score >= min_score][:entry_limit]
        selected: dict[tuple[str, str, str], dict] = {}
        for score, entry in seeds:
            selected[self._key(entry.fact)] = self._record(entry.fact, entry.text, score, "entry")
        for _, entry in seeds:
            if neighbor_limit == 0:
                break
            anchors = [entry.fact.subject]
            if isinstance(entry.fact.object, URIRef):
                anchors.append(entry.fact.object)
            added = 0
            for subject in anchors:
                for _, predicate, object_ in store.content_graph.triples((subject, None, None)):
                    key = str(subject), str(predicate), str(object_)
                    if key in selected:
                        continue
                    neighbor = self.entries.get(key)
                    if neighbor is None:
                        continue
                    selected[key] = self._record(neighbor.fact, neighbor.text, None, "neighbor")
                    added += 1
                    if added >= neighbor_limit:
                        break
                if added >= neighbor_limit:
                    break
        return {
            "entries": [{"subject": str(entry.fact.subject), "score": round(score, 6),
                         "text": entry.text} for score, entry in seeds],
            "facts": list(selected.values()),
            "indexed_count": indexed,
        }

    @staticmethod
    def _record(fact: StoredEvidence, text: str, score: float | None, role: str) -> dict:
        return {"subject": str(fact.subject), "predicate": str(fact.predicate),
                "object": str(fact.object), "text": text, "evidence": fact.evidence,
                "source_url": fact.source_url, "role": role,
                "score": round(score, 6) if score is not None else None}

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding vectors must have equal dimensions")
        norms = sqrt(sum(x * x for x in left)) * sqrt(sum(x * x for x in right))
        return sum(a * b for a, b in zip(left, right)) / norms if norms else 0.0
