"""Ontology-grounded hyperedge indexing over the RDF content graph."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import sqrt
import re
import sqlite3
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

from langchain_core.embeddings import Embeddings
from rdflib import RDF, URIRef

if TYPE_CHECKING:
    from .graph_store import RDFKnowledgeGraphStore, StoredEvidence


SCHEMA = "https://schema.org/"

# Schema.org grounding used to turn related predicates into retrieval units.
ONTOLOGY_FACETS: dict[str, frozenset[str]] = {
    "identity": frozenset({
        "type", "name", "headline", "alternateName", "description", "url",
        "mainEntity", "about",
    }),
    "content": frozenset({
        "text", "articleBody", "abstract", "encodingFormat", "contentSize",
        "inLanguage", "genre",
    }),
    "ingredients": frozenset({"recipeIngredient"}),
    "instructions": frozenset({"recipeInstructions", "step", "itemListElement", "position"}),
    "timing": frozenset({"prepTime", "cookTime", "totalTime", "duration"}),
    "yield": frozenset({"recipeYield"}),
    "dimensions": frozenset({"width", "height", "depth", "weight", "size"}),
    "quantities": frozenset({
        "value", "minValue", "maxValue", "unitCode", "unitText",
        "amountOfThisGood", "numberOfItems",
    }),
    "physical_characteristics": frozenset({
        "color", "material", "pattern", "model", "brand", "manufacturer",
    }),
    "nutrition": frozenset({
        "nutrition", "calories", "carbohydrateContent", "cholesterolContent",
        "fatContent", "fiberContent", "proteinContent", "saturatedFatContent",
        "servingSize", "sodiumContent", "sugarContent", "transFatContent",
        "unsaturatedFatContent",
    }),
    "classification": frozenset({
        "recipeCategory", "recipeCuisine", "suitableForDiet", "keywords",
    }),
    "attribution": frozenset({
        "author", "creator", "contributor", "publisher", "copyrightHolder",
    }),
    "publication": frozenset({"datePublished", "dateModified", "copyrightYear"}),
    "media": frozenset({"image", "video", "thumbnailUrl", "contentUrl"}),
    "list": frozenset({"item"}),
    "commerce": frozenset({
        "offers", "price", "priceCurrency", "priceSpecification",
        "availability", "itemCondition", "seller", "validFrom",
        "priceValidUntil", "eligibleQuantity",
    }),
    "ratings_reviews": frozenset({
        "aggregateRating", "review", "ratingValue", "bestRating", "worstRating",
        "ratingCount", "reviewCount", "reviewRating", "reviewBody",
    }),
    "location": frozenset({
        "location", "address", "streetAddress", "addressLocality",
        "addressRegion", "addressCountry", "postalCode", "geo", "latitude",
        "longitude", "areaServed", "amenityFeature",
    }),
    "hours": frozenset({
        "openingHours", "openingHoursSpecification", "opens", "closes",
        "dayOfWeek",
    }),
    "contact": frozenset({"telephone", "email", "contactPoint", "faxNumber"}),
    "identifiers": frozenset({
        "identifier", "sameAs", "sku", "mpn", "productID", "gtin", "gtin8",
        "gtin12", "gtin13", "gtin14", "isbn",
    }),
    "people_organization": frozenset({
        "givenName", "familyName", "jobTitle", "affiliation", "member",
        "employee", "worksFor", "department",
    }),
    "events": frozenset({
        "startDate", "endDate", "doorTime", "eventStatus",
        "eventAttendanceMode", "organizer", "performer", "attendee",
    }),
    "audience": frozenset({"audience", "typicalAgeRange"}),
    "relationships": frozenset({
        "isPartOf", "hasPart", "mentions", "subjectOf", "relatedLink",
        "significantLink",
    }),
    "technical": frozenset({
        "softwareVersion", "operatingSystem", "applicationCategory",
        "applicationSubCategory", "featureList", "memoryRequirements",
        "processorRequirements", "storageRequirements",
    }),
    "accessibility": frozenset({
        "accessibilityAPI", "accessibilityControl", "accessibilityFeature",
        "accessibilityHazard", "accessMode", "accessModeSufficient",
    }),
    "rights": frozenset({"license", "copyrightNotice", "usageInfo"}),
}


@dataclass(slots=True)
class Hyperedge:
    id: str
    anchor_subject: str
    facet: str
    predicates: tuple[str, ...]
    members: tuple[StoredEvidence, ...]
    description: str
    fingerprint: str
    vector: list[float]

    @property
    def vertices(self) -> frozenset[str]:
        values = {self.anchor_subject}
        for fact in self.members:
            values.add(str(fact.subject))
            if isinstance(fact.object, URIRef):
                values.add(str(fact.object))
        return frozenset(values)


class HyperedgeVectorIndex:
    """Embed semantic groups of RDF facts and expand selected hyperedges."""

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        query_prefix: str = "",
        document_prefix: str = "",
        database_path: str = ":memory:",
        max_description_chars: int = 12000,
    ) -> None:
        if max_description_chars < 500:
            raise ValueError("max_description_chars must be at least 500")
        self.embeddings = embeddings
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.max_description_chars = max_description_chars
        self.entries: dict[str, Hyperedge] = {}
        self.database = sqlite3.connect(database_path)
        self.database.execute(
            "CREATE TABLE IF NOT EXISTS hyperedge_vectors ("
            "hyperedge_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, "
            "description TEXT NOT NULL, vector TEXT NOT NULL)"
        )

    @staticmethod
    def _words(value: str) -> str:
        value = unquote(value)
        value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
        return re.sub(r"[_\-/]+", " ", value).strip()

    @classmethod
    def _local_name(cls, value: object) -> str:
        text = str(value)
        if text == str(RDF.type):
            return "type"
        return text.rsplit("/", 1)[-1].rsplit("#", 1)[-1]

    @classmethod
    def _facet(cls, predicate: object) -> str:
        local = cls._local_name(predicate)
        for facet, predicates in ONTOLOGY_FACETS.items():
            if local in predicates:
                return facet
        # Unknown ontology terms remain coherent, property-specific groups.
        return f"property:{local}"

    @classmethod
    def _entity_name(cls, node: object, store: RDFKnowledgeGraphStore) -> str:
        if isinstance(node, URIRef):
            for predicate in (
                URIRef(f"{SCHEMA}name"), URIRef(f"{SCHEMA}headline"),
                URIRef(f"{SCHEMA}alternateName"),
            ):
                for value in store.content_graph.objects(node, predicate):
                    if str(value).strip():
                        return str(value).strip()
            parsed = urlparse(str(node))
            if parsed.scheme in {"http", "https"}:
                path = cls._words(parsed.path).strip()
                if path:
                    return path
        return cls._words(cls._local_name(node)) or "entity"

    @classmethod
    def _fact_text(cls, fact: StoredEvidence, store: RDFKnowledgeGraphStore) -> str:
        subject = cls._entity_name(fact.subject, store)
        predicate = cls._words(cls._local_name(fact.predicate)).casefold()
        if fact.predicate == RDF.type:
            return f"{subject} is a {cls._words(cls._local_name(fact.object))}."
        object_text = (
            cls._entity_name(fact.object, store)
            if isinstance(fact.object, URIRef)
            else str(fact.object)
        )
        return f"{subject} has {predicate}: {object_text}."

    @staticmethod
    def _fact_key(fact: StoredEvidence) -> tuple[str, str, str]:
        return str(fact.subject), str(fact.predicate), str(fact.object)

    def _groups(
        self, facts: tuple[StoredEvidence, ...], store: RDFKnowledgeGraphStore,
    ) -> dict[tuple[str, str], list[StoredEvidence]]:
        # URI-valued relations allow facts about a nested node (for example a
        # NutritionInformation object) to join the parent's ontology facet.
        nested_facets = {
            "nutrition", "instructions", "attribution", "list", "media",
            "dimensions", "quantities", "commerce", "ratings_reviews",
            "location", "hours", "contact", "events", "audience",
        }
        child_owners: dict[str, set[tuple[str, str]]] = {}
        for fact in facts:
            facet = self._facet(fact.predicate)
            if isinstance(fact.object, URIRef) and facet in nested_facets:
                child_owners.setdefault(str(fact.object), set()).add(
                    (str(fact.subject), facet)
                )
        groups: dict[tuple[str, str], list[StoredEvidence]] = {}
        seen: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
        for fact in facts:
            fact_key = self._fact_key(fact)
            owners = child_owners.get(str(fact.subject))
            keys = owners or {(str(fact.subject), self._facet(fact.predicate))}
            for key in keys:
                group_seen = seen.setdefault(key, set())
                if fact_key not in group_seen:
                    group_seen.add(fact_key)
                    groups.setdefault(key, []).append(fact)
        return groups

    def _build_hyperedges(
        self, facts: tuple[StoredEvidence, ...], store: RDFKnowledgeGraphStore,
    ) -> dict[str, tuple[str, str, tuple[str, ...], tuple[StoredEvidence, ...], str, str]]:
        result = {}
        for (anchor, facet), members_list in self._groups(facts, store).items():
            members = tuple(members_list)
            predicates = tuple(sorted({str(item.predicate) for item in members}))
            anchor_name = self._entity_name(URIRef(anchor), store)
            facet_text = self._words(facet.removeprefix("property:"))
            facts_text = " ".join(self._fact_text(item, store) for item in members)
            description = (
                f"{anchor_name} {facet_text} data group. {facts_text}"
            )[:self.max_description_chars]
            member_keys = sorted("\x1f".join(self._fact_key(item)) for item in members)
            fingerprint = sha256("\x1e".join(member_keys).encode("utf-8")).hexdigest()
            edge_id = "hyperedge:" + sha256(
                f"{anchor}\x1f{facet}".encode("utf-8")
            ).hexdigest()[:24]
            result[edge_id] = (
                anchor, facet, predicates, members, description, fingerprint
            )
        return result

    def sync(self, facts: tuple[StoredEvidence, ...], store: RDFKnowledgeGraphStore) -> int:
        built = self._build_hyperedges(facts, store)
        pending = []
        entries: dict[str, Hyperedge] = {}
        for edge_id, data in built.items():
            anchor, facet, predicates, members, description, fingerprint = data
            row = self.database.execute(
                "SELECT fingerprint, description, vector FROM hyperedge_vectors "
                "WHERE hyperedge_id=?", (edge_id,),
            ).fetchone()
            if row is not None and row[0] == fingerprint and row[1] == description:
                entries[edge_id] = Hyperedge(
                    edge_id, anchor, facet, predicates, members, description,
                    fingerprint, json.loads(row[2]),
                )
            else:
                pending.append((edge_id, data))
        if pending:
            vectors = self.embeddings.embed_documents([
                self.document_prefix + data[4] for _, data in pending
            ])
            if len(vectors) != len(pending):
                raise ValueError("Embedding model returned the wrong number of hyperedge vectors")
            for (edge_id, data), vector in zip(pending, vectors):
                anchor, facet, predicates, members, description, fingerprint = data
                entries[edge_id] = Hyperedge(
                    edge_id, anchor, facet, predicates, members, description,
                    fingerprint, vector,
                )
                self.database.execute(
                    "INSERT INTO hyperedge_vectors VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(hyperedge_id) DO UPDATE SET "
                    "fingerprint=excluded.fingerprint, "
                    "description=excluded.description, vector=excluded.vector",
                    (edge_id, fingerprint, description, json.dumps(vector)),
                )
            self.database.commit()
        self.entries = entries
        return len(pending)

    def search(
        self,
        goals: list[str],
        store: RDFKnowledgeGraphStore,
        *,
        entry_limit: int,
        neighbor_limit: int,
        min_score: float,
    ) -> dict:
        """Rank hyperedges, return their full fact groups, then linked groups."""
        if entry_limit < 1 or neighbor_limit < 0:
            raise ValueError("entry_limit must be positive and neighbor_limit nonnegative")
        indexed = self.sync(store.evidence, store)
        queries = [goal.strip() for goal in goals if goal.strip()]
        if not queries or not self.entries:
            return {
                "entries": [], "facts": [], "indexed_count": indexed,
                "hyperedge_count": len(self.entries),
            }
        query_vectors = [
            self.embeddings.embed_query(self.query_prefix + query) for query in queries
        ]
        ranked = sorted(
            ((max(self._cosine(query, edge.vector) for query in query_vectors), edge)
             for edge in self.entries.values()),
            key=lambda item: item[0], reverse=True,
        )
        seeds = [(score, edge) for score, edge in ranked if score >= min_score][:entry_limit]
        selected: list[tuple[Hyperedge, float | None, str]] = [
            (edge, score, "hyperedge_entry") for score, edge in seeds
        ]
        selected_ids = {edge.id for _, edge in seeds}
        if neighbor_limit:
            seed_vertices = set().union(*(edge.vertices for _, edge in seeds)) if seeds else set()
            neighbors = [
                edge for _, edge in ranked
                if edge.id not in selected_ids and seed_vertices.intersection(edge.vertices)
            ][:neighbor_limit]
            selected.extend((edge, None, "hyperedge_neighbor") for edge in neighbors)

        facts: dict[tuple[str, str, str], dict] = {}
        for edge, score, role in selected:
            for fact in edge.members:
                key = self._fact_key(fact)
                facts.setdefault(key, self._record(fact, store, edge, score, role))
        return {
            "entries": [self._entry(edge, score) for score, edge in seeds],
            "facts": list(facts.values()),
            "indexed_count": indexed,
            "hyperedge_count": len(self.entries),
        }

    def _record(
        self, fact: StoredEvidence, store: RDFKnowledgeGraphStore,
        edge: Hyperedge, score: float | None, role: str,
    ) -> dict:
        return {
            "subject": str(fact.subject), "predicate": str(fact.predicate),
            "object": str(fact.object), "text": self._fact_text(fact, store),
            "evidence": fact.evidence, "source_url": fact.source_url,
            "role": role, "score": round(score, 6) if score is not None else None,
            "hyperedge_id": edge.id, "hyperedge_type": edge.facet,
            "hyperedge_description": edge.description,
        }

    @staticmethod
    def _entry(edge: Hyperedge, score: float) -> dict:
        return {
            "hyperedge_id": edge.id,
            "anchor_subject": edge.anchor_subject,
            "type": edge.facet,
            "predicates": list(edge.predicates),
            "member_count": len(edge.members),
            "score": round(score, 6),
            "description": edge.description,
        }

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding vectors must have equal dimensions")
        norms = sqrt(sum(x * x for x in left)) * sqrt(sum(x * x for x in right))
        return sum(a * b for a, b in zip(left, right)) / norms if norms else 0.0

    def close(self) -> None:
        self.database.close()
