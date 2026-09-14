"""RDFLib storage for Builder-generated content facts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from rdflib import Dataset, Graph, Literal, Namespace, RDF, URIRef
from rdflib.term import Node


GraphKind = str


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
    """Own the RDF named graph containing Builder-supported content facts.

    Builder labels are deterministically converted to URIs. Plain object values
    become RDF literals, while absolute HTTP(S) object values remain URIRefs.
    """

    BASE = Namespace("urn:linked-agentic-retrieval:")
    SCHEMA = Namespace("https://schema.org/")
    CONTENT_GRAPH = URIRef(BASE["graph/content"])

    def __init__(self, dataset: Dataset | None = None) -> None:
        self.dataset = dataset if dataset is not None else Dataset()
        self.dataset.bind("lar", self.BASE)
        self.dataset.bind("schema", self.SCHEMA)
        self.dataset.bind("rdf", RDF)
        self.content_graph = self.dataset.graph(self.CONTENT_GRAPH)
        self._evidence: list[StoredEvidence] = []

    @property
    def evidence(self) -> tuple[StoredEvidence, ...]:
        return tuple(self._evidence)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "content": len(self.content_graph),
            "total": len(self.content_graph),
        }

    def graph(self, kind: GraphKind) -> Graph:
        if kind == "content":
            return self.content_graph
        raise ValueError("Only the content graph is available")

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
        kind = self._field(triple, "semantic_type", "content")
        if kind != "content":
            raise ValueError("Builder triples must be content facts")
        target = self.content_graph
        subject = self._resource(self._field(triple, "subject"), "entity")
        predicate = self._predicate(self._field(triple, "predicate"))
        object_ = self._object(
            self._field(triple, "object"),
            self._field(triple, "object_kind", "literal"),
        )
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
        if kind not in {None, "content"}:
            raise ValueError("Only the content graph is available")
        self.content_graph.remove((None, None, None))
        self._evidence.clear()

    @classmethod
    def _resource(cls, value: Any, category: str) -> URIRef:
        text = str(value).strip()
        if cls._is_absolute_iri(text):
            return URIRef(text)
        slug = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")[:60]
        slug = slug or "unnamed"
        digest = sha256(text.encode("utf-8")).hexdigest()[:10]
        return URIRef(cls.BASE[f"{category}/{slug}-{digest}"])

    @classmethod
    def _predicate(cls, value: Any) -> URIRef:
        text = str(value).strip()
        if text == "rdf:type":
            return RDF.type
        if text == str(RDF.type) or text.startswith(str(cls.SCHEMA)):
            return URIRef(text)
        if text.startswith("http://schema.org/"):
            return URIRef("https://schema.org/" + text.removeprefix("http://schema.org/"))
        raise ValueError(
            "Predicate must be rdf:type or a full https://schema.org/ IRI: "
            f"{text!r}"
        )

    @classmethod
    def _object(cls, value: Any, kind: str) -> Node:
        text = str(value).strip()
        if kind == "iri":
            if not cls._is_absolute_iri(text):
                raise ValueError(f"IRI object is not an absolute IRI: {text!r}")
            return URIRef(text)
        if kind != "literal":
            raise ValueError(f"Unknown object kind: {kind!r}")
        return Literal(text)

    @staticmethod
    def _is_absolute_iri(value: str) -> bool:
        parsed = urlparse(value)
        return bool(parsed.scheme) and (
            parsed.scheme == "urn" or bool(parsed.netloc)
        )

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)
