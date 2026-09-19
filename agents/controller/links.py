"""Find navigable URLs attached to link nodes in a supplied ARIA snapshot."""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit, urlunsplit


_LINK = re.compile(r'^([ \t]*)-\s+link\b(?:\s+"([^"]*)")?', re.I)
_DESTINATION = re.compile(r'^\s*-\s*/(?:url|href):\s*(\S.*?)\s*$', re.I)


def _http_url(value: str, base_url: str) -> str | None:
    raw = value.strip().strip('"\'')
    if not raw or raw.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
        return None
    absolute = urljoin(base_url, raw)
    parsed = urlsplit(absolute)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def aria_links(snapshot: str, base_url: str) -> list[dict[str, str]]:
    """Return distinct HTTP(S) destinations from ARIA link `/url` children."""
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    link_indent: int | None = None
    link_name = ""
    for line in snapshot.splitlines():
        match = _LINK.match(line)
        if match:
            link_indent = len(match.group(1))
            link_name = match.group(2) or ""
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if link_indent is None or indent <= link_indent:
            link_indent = None
            continue
        destination = _DESTINATION.match(line)
        if destination:
            url = _http_url(destination.group(1), base_url)
            if url and url not in seen:
                seen.add(url)
                links.append({"name": link_name, "url": url})
    return links
