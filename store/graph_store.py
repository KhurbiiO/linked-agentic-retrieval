"""RDFLib storage for Builder-generated content and layout graphs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Iterable, Literal as TypingLiteral
from urllib.parse import urlparse

from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.term import Node


GraphKind = TypingLiteral["content", "layout"]


@dataclass(frozen=True, slots=True)
class StoredEvidence:
    """Provenance retained for one generated RDF statement."""

    graph: GraphKind
    subject: Node
    predicate: URIRef
    object: Node
    evidence: str
    source_url: str | None = None


class RDFKnowledgeGraphStore:
    """Own separate RDF named graphs for content and page-layout semantics.

    Builder labels are deterministically converted to URIs. Plain object values
    become RDF literals, while absolute HTTP(S) object values remain URIRefs.
    """

    BASE = Namespace("urn:linked-agentic-retrieval:")
    CONTENT_GRAPH = URIRef(BASE["graph/content"])
    LAYOUT_GRAPH = URIRef(BASE["graph/layout"])

    def __init__(self, dataset: Dataset | None = None) -> None:
        self.dataset = dataset if dataset is not None else Dataset()
        self.dataset.bind("lar", self.BASE)
        self.content_graph = self.dataset.graph(self.CONTENT_GRAPH)
        self.layout_graph = self.dataset.graph(self.LAYOUT_GRAPH)
        self._evidence: list[StoredEvidence] = []

    @property
    def evidence(self) -> tuple[StoredEvidence, ...]:
        return tuple(self._evidence)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "content": len(self.content_graph),
            "layout": len(self.layout_graph),
            "total": len(self.content_graph) + len(self.layout_graph),
        }

    def graph(self, kind: GraphKind) -> Graph:
        if kind == "content":
            return self.content_graph
        if kind == "layout":
            return self.layout_graph
        raise ValueError(f"Unknown graph kind: {kind!r}")

    def add_knowledge_graph(
        self,
        knowledge_graph: Any,
        *,
        source_url: str | None = None,
    ) -> None:
        """Store every GraphTriple-like item from a KnowledgeGraph-like object."""
        triples = self._field(knowledge_graph, "triples", [])
        for triple in triples:
            self.add_triple(triple, source_url=source_url)

    def add_triple(self, triple: Any, *, source_url: str | None = None) -> None:
        kind = self._field(triple, "semantic_type")
        target = self.graph(kind)
        subject = self._resource(self._field(triple, "subject"), "entity")
        predicate = self._resource(self._field(triple, "predicate"), "property")
        object_ = self._object(self._field(triple, "object"))
        target.add((subject, predicate, object_))
        self._evidence.append(StoredEvidence(
            graph=kind,
            subject=subject,
            predicate=predicate,
            object=object_,
            evidence=str(self._field(triple, "evidence", "")),
            source_url=source_url,
        ))

    def serialize(
        self,
        kind: GraphKind | None = None,
        *,
        format: str = "trig",
    ) -> str:
        """Serialize one named graph, or the complete Dataset when kind is None."""
        value = (self.graph(kind) if kind else self.dataset).serialize(format=format)
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def save(
        self,
        path: str | Path,
        kind: GraphKind | None = None,
        *,
        format: str = "trig",
    ) -> None:
        Path(path).write_text(self.serialize(kind, format=format), encoding="utf-8")

    def clear(self, kind: GraphKind | None = None) -> None:
        kinds: Iterable[GraphKind] = (kind,) if kind else ("content", "layout")
        selected = set(kinds)
        for selected_kind in selected:
            self.graph(selected_kind).remove((None, None, None))
        self._evidence = [item for item in self._evidence if item.graph not in selected]

    @classmethod
    def _resource(cls, value: Any, category: str) -> URIRef:
        text = str(value).strip()
        if cls._is_web_url(text):
            return URIRef(text)
        slug = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")[:60]
        slug = slug or "unnamed"
        digest = sha256(text.encode("utf-8")).hexdigest()[:10]
        return URIRef(cls.BASE[f"{category}/{slug}-{digest}"])

    @classmethod
    def _object(cls, value: Any) -> Node:
        text = str(value).strip()
        return URIRef(text) if cls._is_web_url(text) else Literal(text)

    @staticmethod
    def _is_web_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)
