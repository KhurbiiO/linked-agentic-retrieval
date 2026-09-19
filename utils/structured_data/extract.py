"""Parse schema.org JSON-LD and microdata from supplied page HTML."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from bs4 import BeautifulSoup
import extruct
from rdflib import Graph
from w3lib.html import get_base_url


@dataclass(slots=True)
class StructuredDataResult:
    url: str
    graph: Graph
    json_ld_documents: int = 0
    microdata_documents: int = 0
    errors: list[str] = field(default_factory=list)


class StructuredDataExtractor:
    """Parse embedded structured data from HTML supplied by the Controller."""

    def extract_html(self, html: str, url: str) -> StructuredDataResult:
        graph = Graph()
        errors: list[str] = []
        json_ld_count = 0
        soup = BeautifulSoup(html, "html.parser")

        # Match the explicit JSON-LD parsing used by test.py so malformed
        # blocks can be isolated without discarding valid blocks.
        for index, script in enumerate(soup.find_all("script", type="application/ld+json")):
            try:
                data = json.loads(script.string or script.get_text())
                # Keep parsing self-contained.  RDFLib otherwise dereferences
                # remote @context URLs such as https://schema.org, which makes
                # extraction slow and brittle and can fail in offline runs.
                document = self._schema_context_document(data)
                graph.parse(data=json.dumps(document), format="json-ld", publicID=url)
                json_ld_count += 1
            except Exception as exc:
                errors.append(f"json-ld[{index}]: {exc}")

        microdata_count = 0
        try:
            base_url = get_base_url(html, url)
            extracted = extruct.extract(
                html,
                base_url=base_url,
                syntaxes=["microdata"],
                uniform=True,
            )
            for index, item in enumerate(extracted.get("microdata", [])):
                try:
                    document = self._microdata_document(item)
                    graph.parse(data=json.dumps(document), format="json-ld", publicID=base_url)
                    microdata_count += 1
                except Exception as exc:
                    errors.append(f"microdata[{index}]: {exc}")
        except Exception as exc:
            errors.append(f"microdata: {exc}")

        return StructuredDataResult(
            url=url,
            graph=graph,
            json_ld_documents=json_ld_count,
            microdata_documents=microdata_count,
            errors=errors,
        )

    @classmethod
    def _microdata_document(cls, value: Any) -> Any:
        """Ensure extruct's normalized microdata has a schema.org context."""
        if isinstance(value, list):
            return [cls._microdata_document(item) for item in value]
        if not isinstance(value, dict):
            return value
        converted = {key: cls._microdata_document(item) for key, item in value.items()}
        converted["@context"] = {"@vocab": "https://schema.org/"}
        return converted

    @classmethod
    def _schema_context_document(cls, value: Any) -> Any:
        """Replace remote schema.org contexts with an equivalent local context."""
        if isinstance(value, list):
            return [cls._schema_context_document(item) for item in value]
        if not isinstance(value, dict):
            return value
        converted = {
            key: cls._schema_context_document(item)
            for key, item in value.items()
            if key != "@context"
        }
        converted["@context"] = {"@vocab": "https://schema.org/"}
        return converted
