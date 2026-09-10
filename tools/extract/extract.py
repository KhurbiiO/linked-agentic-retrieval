import json
import html
import re
from urllib.parse import parse_qsl, unquote, urljoin, urlparse

import requests
import extruct

from bs4 import BeautifulSoup
from w3lib.html import get_base_url

if __package__:
    from .scoring import CandidateScorer, WeightedContextScorer
else:
    # Support direct execution: python tools/extract/extract.py
    from scoring import CandidateScorer, WeightedContextScorer


class ExtractionResult(dict):
    """Public structured extraction with private HTML traversal metadata."""

    def __init__(self, *args, html_references=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.html_references = html_references or []


class StructuredDataExtractor:
    def __init__(
        self,
        timeout=30,
        use_custom_extraction=True,
        link_context_max_fields=12,
        link_context_max_chars=1000,
        link_context_child_depth=2,
        candidate_scorer: CandidateScorer | None = None,
        excluded_url_extensions=None,
    ):
        self.timeout = timeout
        self.use_custom_extraction = use_custom_extraction
        self.link_context_max_fields = link_context_max_fields
        self.link_context_max_chars = link_context_max_chars
        self.link_context_child_depth = link_context_child_depth
        self.candidate_scorer = candidate_scorer or WeightedContextScorer()
        self.excluded_url_extensions = tuple(
            extension.casefold() if extension.startswith(".") else f".{extension.casefold()}"
            for extension in (excluded_url_extensions or [])
        )

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/150.0 Safari/537.36"
            )
        }

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def extract(self, url):
        """
        Main pipeline.

        Input:
            URL

        Output:
            {
                "url": ...,
                "standard": {...},
                "embedded_json": [...],
                "schema_objects": [...],
                "all_typed_objects": [...]
            }
        """

        self._validate_url(url)
        html_text, final_url = self._download(url)

        standard = self._extract_standard(
            html_text,
            final_url
        )

        embedded = (
            self._extract_embedded_json(html_text)
            if self.use_custom_extraction
            else []
        )

        # Also recursively inspect JSON-LD found by extruct
        json_sources = []

        for item in standard.get("json-ld", []):
            json_sources.append({
                "source": "json-ld",
                "data": item
            })

        json_sources.extend(embedded)

        schema_objects = self._find_schema_objects(
            json_sources
        )

        typed_objects = self._find_typed_objects(
            json_sources
        )

        return ExtractionResult({
            "url": final_url,
            "standard": standard,
            "embedded_json": embedded,
            "schema_objects": schema_objects,
            "all_typed_objects": typed_objects
        }, html_references=self._extract_html_references(html_text, final_url))

    # --------------------------------------------------
    # Download
    # --------------------------------------------------

    def _download(self, url):

        response = requests.get(
            url,
            headers=self.headers,
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.text, response.url

    # --------------------------------------------------
    # Standard structured metadata
    # --------------------------------------------------

    def _extract_standard(self, html_text, url):

        base_url = get_base_url(
            html_text,
            url
        )

        try:
            return extruct.extract(
                html_text,
                base_url=base_url,
                syntaxes=[
                    "json-ld",
                    "microdata",
                    "rdfa",
                    "opengraph",
                    "microformat",
                    "dublincore"
                ],
                uniform=True
            )

        except Exception as e:
            # print("Extruct error:", e)
            return {}

    # --------------------------------------------------
    # Try JSON
    # --------------------------------------------------

    def _try_json(self, value):

        if not isinstance(value, str):
            return None

        value = value.strip()

        if not value:
            return None

        # Ignore obvious non-JSON
        if not value.startswith(("{", "[")):
            decoded = html.unescape(value).strip()

            if not decoded.startswith(("{", "[")):
                return None

            value = decoded

        # Repeated decode for nested HTML escaping
        for _ in range(3):

            try:
                return json.loads(value)

            except json.JSONDecodeError:
                decoded = html.unescape(value)

                if decoded == value:
                    break

                value = decoded.strip()

        return None

    # --------------------------------------------------
    # Extract arbitrary embedded JSON
    # --------------------------------------------------

    def _extract_embedded_json(self, html_text):

        soup = BeautifulSoup(
            html_text,
            "html.parser"
        )

        results = []

        # ----------------------------------------------
        # HTML attributes
        # ----------------------------------------------

        for tag in soup.find_all(True):

            for attr_name, attr_value in tag.attrs.items():

                values = (
                    attr_value
                    if isinstance(attr_value, list)
                    else [attr_value]
                )

                for value in values:

                    parsed = self._try_json(value)

                    if parsed is not None:

                        results.append({
                            "source": "attribute",
                            "tag": tag.name,
                            "attribute": attr_name,
                            "data": parsed
                        })

        # ----------------------------------------------
        # Script tags
        # ----------------------------------------------

        for index, script in enumerate(
            soup.find_all("script")
        ):

            content = (
                script.string
                or script.get_text()
            )

            if not content:
                continue

            parsed = self._try_json(content)

            if parsed is not None:

                results.append({
                    "source": "script",
                    "script_index": index,
                    "script_type": script.get("type"),
                    "script_id": script.get("id"),
                    "data": parsed
                })

        return results

    def _extract_html_references(self, html_text, page_url=""):
        """Capture HTML hrefs for optional traversal."""
        soup = BeautifulSoup(html_text, "html.parser")
        base_url = get_base_url(html_text, page_url) if page_url else page_url
        references = []

        for index, tag in enumerate(soup.find_all(href=True)):
            href = str(tag.get("href", "")).strip()
            if not href:
                continue
            references.append({
                "href": href,
                "base_url": base_url,
                "tag": tag.name,
                "index": index,
            })

        return references

    # --------------------------------------------------
    # Generic recursive JSON walker
    # --------------------------------------------------

    def _walk(self, obj, path=()):

        yield path, obj

        if isinstance(obj, dict):

            for key, value in obj.items():

                yield from self._walk(
                    value,
                    path + (key,)
                )

        elif isinstance(obj, list):

            for index, value in enumerate(obj):

                yield from self._walk(
                    value,
                    path + (index,)
                )

    # --------------------------------------------------
    # Find Schema / JSON-LD objects
    # --------------------------------------------------

    def _find_schema_objects(self, sources):

        results = []

        seen = set()

        for source in sources:

            for path, obj in self._walk(
                source["data"]
            ):

                if not isinstance(obj, dict):
                    continue

                if (
                    "@context" in obj
                    or "@type" in obj
                ):

                    fingerprint = self._fingerprint(obj)

                    if fingerprint in seen:
                        continue

                    seen.add(fingerprint)

                    results.append({
                        "source": source["source"],
                        "path": path,
                        "type": obj.get("@type"),
                        "context": obj.get("@context"),
                        "data": obj
                    })

        return results

    # --------------------------------------------------
    # Find every @type object
    # --------------------------------------------------

    def _find_typed_objects(self, sources):

        results = []

        seen = set()

        for source in sources:

            for path, obj in self._walk(
                source["data"]
            ):

                if not isinstance(obj, dict):
                    continue

                if "@type" not in obj:
                    continue

                fingerprint = self._fingerprint(obj)

                if fingerprint in seen:
                    continue

                seen.add(fingerprint)

                results.append({
                    "source": source["source"],
                    "path": path,
                    "type": obj["@type"],
                    "data": obj
                })

        return results

    # --------------------------------------------------
    # Deduplication
    # --------------------------------------------------

    def _fingerprint(self, obj):

        try:
            return json.dumps(
                obj,
                sort_keys=True,
                ensure_ascii=False
            )

        except Exception:
            return str(obj)

    # --------------------------------------------------
    # Convenience queries
    # --------------------------------------------------

    def get_type(self, result, wanted_type):
        """
        Get all Schema.org objects matching @type.
        """

        matches = []

        for entry in result["all_typed_objects"]:

            schema_type = entry["type"]

            if schema_type == wanted_type:

                matches.append(
                    entry["data"]
                )

            elif (
                isinstance(schema_type, list)
                and wanted_type in schema_type
            ):
                matches.append(
                    entry["data"]
                )

        return matches

    # --------------------------------------------------
    # Agent retrieval operations
    # --------------------------------------------------

    def traverse(self, result, search_terms, max_results=12, goal=""):
        """Rank scalar paths and values using the configured scoring strategy."""
        terms = [term.casefold().strip() for term in search_terms if term.strip()]
        source_url = str(result.get("url", ""))
        matches = []
        scalar_items = []
        evidence_sources = {
            "standard": result.get("standard", {}),
            "embedded_json": result.get("embedded_json", []),
        }

        for path, value in self._walk(evidence_sources):
            if value is None or isinstance(value, (dict, list)):
                continue

            rendered = self._render(value)
            json_path = self._format_path(path)
            naturalized_path = self._naturalize_json_path(path)
            naturalized_pair = (
                f"{naturalized_path}: {rendered}"
                if naturalized_path
                else rendered
            )
            scalar_items.append((json_path, rendered, naturalized_pair))

        scores = self.candidate_scorer.score_evidence_batch(
            items=[("", naturalized_pair) for _, _, naturalized_pair in scalar_items],
            source_url=source_url,
            goal=goal,
            search_terms=terms,
        )
        for (json_path, rendered, naturalized_pair), scored in zip(scalar_items, scores):
            if scored.total > 0:
                matches.append({
                    "source_url": source_url,
                    "json_path": json_path,
                    "value": rendered,
                    "naturalized_pair": naturalized_pair,
                    "score": scored.total,
                    "score_components": scored.components,
                })

        matches.sort(key=lambda item: (-item["score"], item["json_path"]))
        return matches[:max_results]

    @staticmethod
    def _naturalize_json_path(path):
        """Convert a structured-data path into readable relation words."""
        ignored = {"standard", "embedded", "json", "ld", "data"}
        words = []
        for part in path:
            if isinstance(part, int):
                continue
            text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(part))
            for word in re.findall(r"[^\W_]+", text.replace("_", " "), flags=re.UNICODE):
                normalized = word.casefold()
                if normalized not in ignored:
                    words.append(normalized)
        return " ".join(words)

    def discover_links(self, result, search_terms, max_links=20, goal=""):
        """Rank HTML hrefs using only the URL path and query string."""
        base_url = str(result.get("url", ""))
        terms = [term.casefold().strip() for term in search_terms if term.strip()]
        candidates = []

        for reference in getattr(result, "html_references", []):
            raw_value = reference["href"]
            if raw_value.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
                continue
            candidate = urljoin(reference.get("base_url") or base_url, raw_value)
            parsed = urlparse(candidate)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            if parsed.path.casefold().endswith(self.excluded_url_extensions):
                continue

            normalized = parsed._replace(fragment="").geturl()
            scoring_url = self._naturalize_url(parsed)
            json_path = f'html.{reference["tag"]}[{reference["index"]}].@href'
            candidates.append({
                "url": normalized,
                "scoring_url": scoring_url,
                "json_path": json_path,
                "goal": goal,
                "search_terms": terms,
                "parent_json_path": f'html.{reference["tag"]}[{reference["index"]}]',
            })

        scores = self.candidate_scorer.score_batch([
            {
                "url": item["scoring_url"],
                "json_path": "",
                "context": {},
                "goal": item["goal"],
                "search_terms": item["search_terms"],
            }
            for item in candidates
        ])
        found = {}
        for candidate, scored in zip(candidates, scores):
            normalized = candidate["url"]
            current = found.get(normalized)
            item = {
                "url": normalized,
                "json_path": candidate["json_path"],
                "parent_json_path": candidate["parent_json_path"],
                "anchor_text": None,
                "context": {},
                "score": scored.total,
                "score_components": scored.components,
            }
            if current is None or scored.total > current["score"]:
                found[normalized] = item

        links = sorted(found.values(), key=lambda item: (-item["score"], item["url"]))
        return links[:max_links]

    @staticmethod
    def _naturalize_url(parsed):
        """Turn a URL path and query into cleaner text for relevance scoring."""
        tracking_keys = {
            "fbclid", "gclid", "dclid", "msclkid", "ref", "source",
        }
        parts = []
        path = unquote(parsed.path or "")
        path = re.sub(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
            " ",
            path,
        )
        path = re.sub(r"\.(?:html?|php|aspx?)$", "", path, flags=re.IGNORECASE)
        parts.append(path)

        for key, value in parse_qsl(parsed.query, keep_blank_values=False):
            normalized_key = key.casefold()
            if normalized_key.startswith("utm_") or normalized_key in tracking_keys:
                continue
            if normalized_key in {"id", "uuid", "guid", "token"} and (
                value.isdigit() or len(value) >= 12
            ):
                continue
            parts.extend((unquote(key), unquote(value)))

        text = " ".join(parts)
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
        tokens = re.findall(r"[^\W_]+", text.replace("_", " "), flags=re.UNICODE)

        meaningful = []
        for token in tokens:
            if token.isdigit():
                continue
            if len(token) >= 12 and re.fullmatch(r"[0-9a-fA-F]+", token):
                continue
            if len(token) >= 16 and any(char.isdigit() for char in token):
                continue
            meaningful.append(token.casefold())

        return " ".join(meaningful)

    @staticmethod
    def _value_at_path(root, path):
        value = root
        for part in path:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and isinstance(part, int) and part < len(value):
                value = value[part]
            else:
                return None
        return value

    def _link_context(self, parent, excluded_key=None):
        """Return bounded scalar values from the URL's parent and nested children."""
        if not isinstance(parent, dict):
            return {}

        preferred = ("name", "title", "label", "heading", "description", "text", "type")
        keys = [key for key in preferred if key in parent]
        keys.extend(key for key in parent if key not in keys)
        context = {}
        consumed = 0

        for key in keys:
            if key == excluded_key:
                continue
            for context_key, value in self._flatten_context(
                parent[key], str(key), self.link_context_child_depth
            ):
                if len(context) >= self.link_context_max_fields:
                    return context
                rendered = self._render(value, limit=300)
                remaining = self.link_context_max_chars - consumed - len(context_key)
                if remaining <= 0:
                    return context
                rendered = rendered[:remaining]
                context[context_key] = rendered
                consumed += len(context_key) + len(rendered)

        return context

    def _flatten_context(self, value, prefix, depth):
        if isinstance(value, (str, int, float, bool)):
            yield prefix, value
        elif isinstance(value, list):
            if all(isinstance(item, (str, int, float, bool)) for item in value):
                yield prefix, ", ".join(str(item) for item in value)
            elif depth > 0:
                for index, child in enumerate(value):
                    yield from self._flatten_context(child, f"{prefix}[{index}]", depth - 1)
        elif isinstance(value, dict) and depth > 0:
            preferred = ("name", "title", "label", "heading", "description", "text", "type")
            keys = [key for key in preferred if key in value]
            keys.extend(key for key in value if key not in keys)
            for key in keys:
                yield from self._flatten_context(value[key], f"{prefix}.{key}", depth - 1)

    @staticmethod
    def _anchor_text(context):
        for wanted in ("name", "title", "label", "heading", "text"):
            for key, value in context.items():
                leaf = key.rsplit(".", 1)[-1].split("[", 1)[0]
                if leaf == wanted and value:
                    return value
        return None

    @staticmethod
    def _format_path(path):
        rendered = "$"
        for part in path:
            if isinstance(part, int):
                rendered += f"[{part}]"
            elif str(part).replace("_", "").replace("-", "").isalnum():
                rendered += f".{part}"
            else:
                rendered += f"[{part!r}]"
        return rendered

    @staticmethod
    def _render(value, limit=500):
        rendered = str(value).replace("\n", " ").strip()
        return rendered if len(rendered) <= limit else f"{rendered[:limit]}..."

    @staticmethod
    def _validate_url(url):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Extraction requires an absolute HTTP(S) URL: {url}")


if __name__ == "__main__":
    extractor = StructuredDataExtractor(use_custom_extraction=False)

    data = extractor.extract(
        "https://www.bbcgoodfood.com/recipes/salmon-beetroot-feta-lime-salsa"
    )

    with open(
        "tools/chicken.json",
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            data,
            output,
            indent=2,
            ensure_ascii=False,
        )

    # Count how many links were found
    print(f"Found {len(data.html_references)} HTML references.")
