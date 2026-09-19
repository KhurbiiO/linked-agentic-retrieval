"""RDFLib storage for Builder-generated content facts."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from rdflib import BNode, Dataset, Graph, Literal, Namespace, RDF, URIRef
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

    def add_rdflib_graph(
        self,
        graph: Graph,
        *,
        source_url: str,
    ) -> list[StoredEvidence]:
        """Import schema.org facts from an extracted RDF graph."""
        imported: list[StoredEvidence] = []
        for subject, predicate, object_ in graph:
            normalized_predicate = self._structured_predicate(predicate)
            if normalized_predicate is None:
                continue
            normalized_subject = self._structured_node(subject, source_url, subject=True)
            normalized_object = self._structured_node(object_, source_url, subject=False)
            triple = (normalized_subject, normalized_predicate, normalized_object)
            self.content_graph.add(triple)
            evidence = StoredEvidence(
                graph="content",
                subject=normalized_subject,
                predicate=normalized_predicate,
                object=normalized_object,
                evidence=f"Embedded structured data: {normalized_subject} {normalized_predicate} {normalized_object}",
                source_url=source_url,
            )
            self._evidence.append(evidence)
            imported.append(evidence)
        return imported

    def query_sparql(self, query: str, *, limit: int = 12) -> dict[str, Any]:
        """Execute one bounded, read-only SPARQL SELECT or ASK query."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if len(query) > 12000:
            raise ValueError("SPARQL query is too long")
        # Keep '#' characters inside namespace IRIs (notably rdf:). RDFLib
        # handles SPARQL comments during parsing; validation remains stricter
        # if forbidden words in comments are rejected too.
        cleaned = query.strip()
        if re.search(
            r"\b(?:INSERT|DELETE|LOAD|CLEAR|CREATE|DROP|COPY|MOVE|ADD|WITH|SERVICE|FROM)\b",
            cleaned,
            re.IGNORECASE,
        ):
            raise ValueError("SPARQL updates and external graph access are not allowed")
        query_match = re.search(r"\b(SELECT|ASK)\b", cleaned, re.IGNORECASE)
        query_type = query_match.group(1).upper() if query_match else ""
        if not query_type:
            raise ValueError("Only SPARQL SELECT and ASK queries are allowed")
        bounded = cleaned
        if query_type == "SELECT":
            limit_match = re.search(r"\bLIMIT\s+(\d+)\b", bounded, re.IGNORECASE)
            if limit_match:
                bounded = (
                    bounded[:limit_match.start(1)]
                    + str(min(int(limit_match.group(1)), limit))
                    + bounded[limit_match.end(1):]
                )
            else:
                bounded = f"{bounded}\nLIMIT {limit}"
        result = self.content_graph.query(bounded)
        if query_type == "ASK":
            return {"type": "ASK", "boolean": bool(result.askAnswer), "rows": []}
        rows = []
        variables = [str(variable) for variable in result.vars]
        for row in result:
            binding = {
                variable: str(value)
                for variable, value in zip(variables, row)
                if value is not None
            }
            evidence, sources = self._provenance_for_values(set(binding.values()))
            rows.append({
                "bindings": binding,
                "evidence": evidence,
                "source_urls": sources,
            })
        return {"type": "SELECT", "variables": variables, "rows": rows}

    def ontology_grounding(self, *, max_terms: int = 40) -> dict[str, Any]:
        """Describe only classes and predicates currently present in the graph."""
        if max_terms < 1:
            raise ValueError("max_terms must be at least 1")
        class_counts: Counter[str] = Counter()
        predicate_counts: Counter[str] = Counter()
        object_kinds: dict[str, set[str]] = defaultdict(set)
        for subject, predicate, object_ in self.content_graph:
            predicate_iri = str(predicate)
            predicate_counts[predicate_iri] += 1
            object_kinds[predicate_iri].add(
                "iri" if isinstance(object_, URIRef) else "literal"
            )
            if predicate == RDF.type and isinstance(object_, URIRef):
                class_counts[str(object_)] += 1

        def term(iri: str) -> str:
            if iri.startswith(str(self.SCHEMA)):
                return "schema:" + iri.removeprefix(str(self.SCHEMA))
            if iri == str(RDF.type):
                return "rdf:type"
            return iri

        classes = [
            {"term": term(iri), "iri": iri, "usage_count": count}
            for iri, count in class_counts.most_common(max_terms)
        ]
        properties = [
            {
                "term": term(iri),
                "iri": iri,
                "usage_count": count,
                "object_kinds": sorted(object_kinds[iri]),
            }
            for iri, count in predicate_counts.most_common(max_terms)
        ]
        return {
            "namespaces": {
                "schema": str(self.SCHEMA),
                "rdf": str(RDF),
            },
            "available_classes": classes,
            "available_properties": properties,
            "total_graph_triples": len(self.content_graph),
            "truncated": (
                len(class_counts) > max_terms or len(predicate_counts) > max_terms
            ),
        }

    def _provenance_for_values(self, values: set[str]) -> tuple[list[str], list[str]]:
        evidence: list[str] = []
        sources: list[str] = []
        for item in self._evidence:
            if values.intersection({str(item.subject), str(item.predicate), str(item.object)}):
                if item.evidence and item.evidence not in evidence:
                    evidence.append(item.evidence)
                if item.source_url and item.source_url not in sources:
                    sources.append(item.source_url)
        return evidence[:5], sources[:5]

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

    @classmethod
    def _structured_predicate(cls, value: Node) -> URIRef | None:
        text = str(value)
        if text == str(RDF.type):
            return RDF.type
        if text.startswith("http://schema.org/"):
            return URIRef(str(cls.SCHEMA) + text.removeprefix("http://schema.org/"))
        if text.startswith(str(cls.SCHEMA)):
            return URIRef(text)
        return None

    @classmethod
    def _structured_node(cls, value: Node, source_url: str, *, subject: bool) -> Node:
        if isinstance(value, BNode):
            digest = sha256(f"{source_url}:{value}".encode("utf-8")).hexdigest()
            return URIRef(cls.BASE[f"structured/{digest}"])
        if isinstance(value, URIRef):
            text = str(value)
            if text.startswith("http://schema.org/"):
                return URIRef(str(cls.SCHEMA) + text.removeprefix("http://schema.org/"))
            return value
        if subject:
            return cls._resource(value, "structured")
        return value if isinstance(value, Literal) else Literal(str(value))

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
