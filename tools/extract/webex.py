import json
import html
from urllib.parse import urljoin, urlparse

import requests
import extruct

from bs4 import BeautifulSoup
from w3lib.html import get_base_url


class StructuredDataExtractor:
    def __init__(self, timeout=30, link_context_max_fields=12, link_context_max_chars=1000):
        self.timeout = timeout
        self.link_context_max_fields = link_context_max_fields
        self.link_context_max_chars = link_context_max_chars

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

        embedded = self._extract_embedded_json(
            html_text
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

        return {
            "url": final_url,
            "standard": standard,
            "embedded_json": embedded,
            "schema_objects": schema_objects,
            "all_typed_objects": typed_objects
        }

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

    def traverse(self, result, search_terms, max_results=12):
        """Rank scalar paths and values against retrieval terms."""
        terms = [term.casefold().strip() for term in search_terms if term.strip()]
        source_url = str(result.get("url", ""))
        matches = []

        for path, value in self._walk(result):
            if value is None or isinstance(value, (dict, list)):
                continue

            rendered = self._render(value)
            json_path = self._format_path(path)
            haystack = f"{json_path} {rendered}".casefold()
            score = sum(haystack.count(term) for term in terms)
            if score:
                matches.append({
                    "source_url": source_url,
                    "json_path": json_path,
                    "value": rendered,
                    "score": score
                })

        matches.sort(key=lambda item: (-item["score"], item["json_path"]))
        return matches[:max_results]

    def discover_links(self, result, search_terms, max_links=20):
        """Return ranked HTTP(S) links found in extracted structured data."""
        base_url = str(result.get("url", ""))
        terms = [term.casefold().strip() for term in search_terms if term.strip()]
        found = {}

        for path, value in self._walk(result):
            if not isinstance(value, str):
                continue

            raw_value = value.strip()
            json_path = self._format_path(path)
            path_hint = str(path[-1]).casefold() if path else ""
            is_link_field = any(hint in path_hint for hint in ("url", "href", "link"))
            if not (
                raw_value.startswith(("http://", "https://", "/", "./", "../"))
                or (is_link_field and raw_value and not any(char.isspace() for char in raw_value))
            ):
                continue

            candidate = urljoin(base_url, raw_value)
            parsed = urlparse(candidate)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue

            normalized = parsed._replace(fragment="").geturl()
            parent = self._value_at_path(result, path[:-1])
            context = self._link_context(parent, excluded_key=path[-1] if path else None)
            anchor_text = self._anchor_text(context)
            context_text = " ".join(f"{key} {value}" for key, value in context.items())
            score = sum(
                f"{json_path} {normalized} {context_text}".casefold().count(term)
                for term in terms
            )
            current = found.get(normalized)
            item = {
                "url": normalized,
                "json_path": json_path,
                "parent_json_path": self._format_path(path[:-1]),
                "anchor_text": anchor_text,
                "context": context,
                "score": score,
            }
            if current is None or score > current["score"]:
                found[normalized] = item

        links = sorted(found.values(), key=lambda item: (-item["score"], item["url"]))
        return links[:max_links]

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
        """Return bounded scalar siblings from the URL's immediate parent object."""
        if not isinstance(parent, dict):
            return {}

        preferred = ("name", "title", "label", "heading", "description", "text", "type")
        keys = [key for key in preferred if key in parent]
        keys.extend(key for key in parent if key not in keys)
        context = {}
        consumed = 0

        for key in keys:
            if key == excluded_key or len(context) >= self.link_context_max_fields:
                continue
            value = parent[key]
            if isinstance(value, (str, int, float, bool)):
                rendered = self._render(value, limit=300)
            elif isinstance(value, list) and all(
                isinstance(item, (str, int, float, bool)) for item in value
            ):
                rendered = self._render(", ".join(str(item) for item in value), limit=300)
            else:
                continue

            remaining = self.link_context_max_chars - consumed
            if remaining <= 0:
                break
            rendered = rendered[:remaining]
            context[str(key)] = rendered
            consumed += len(str(key)) + len(rendered)

        return context

    @staticmethod
    def _anchor_text(context):
        for key in ("name", "title", "label", "heading", "text"):
            if context.get(key):
                return context[key]
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
    extractor = StructuredDataExtractor()

    data = extractor.extract(
        "https://foodnetwork.co.uk/chefs"
    )
    
    with open(
        "tools/extracted_data_2.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )
