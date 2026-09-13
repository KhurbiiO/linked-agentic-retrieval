"""Interactive Playwright page wrapper for exploring ARIA output."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Locator, Page, sync_playwright


class AriaPage:
    """Keep a browser page open while inspecting and refreshing its ARIA tree."""

    def __init__(
        self,
        *,
        headless: bool = True,
        timeout: float = 30,
        exclude_roles: Iterable[str] = (),
        exclude_patterns: Iterable[str] = (),
        exclude_properties: Iterable[str] = (),
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.headless = headless
        self.timeout = timeout
        self.timeout_ms = timeout * 1000
        self.playwright = None
        self.browser = None
        self.context = None
        self.page: Page | None = None
        self.exclude_roles = self._terms(exclude_roles)
        self.exclude_patterns = tuple(exclude_patterns)
        self.exclude_properties = self._terms(exclude_properties)
        self.raw_aria: str = ""
        self.aria: str = ""

    def start(self) -> AriaPage:
        """Start Chromium if it is not already running."""
        if self.page is not None:
            return self
        self.playwright = sync_playwright().start()
        try:
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.context = self.browser.new_context()
            self.page = self.context.new_page()
            self.page.set_default_timeout(self.timeout_ms)
            self.page.set_default_navigation_timeout(self.timeout_ms)
        except Exception:
            self.close()
            raise
        return self

    def navigate(self, url: str, *, wait_until: str = "domcontentloaded") -> AriaPage:
        """Navigate the persistent page and refresh its body ARIA snapshot."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be an absolute HTTP(S) URL")
        self.start()
        assert self.page is not None
        self.page.goto(url, wait_until=wait_until, timeout=self.timeout_ms)
        self.snapshot()
        return self

    def snapshot(
        self,
        selector: str = "body",
        *,
        mode: str = "ai",
        depth: int | None = None,
        boxes: bool = False,
        exclude_roles: Iterable[str] | None = None,
        exclude_patterns: Iterable[str] | None = None,
        exclude_properties: Iterable[str] | None = None,
    ) -> str:
        """Create, filter, and store an ARIA snapshot for a page locator."""
        if mode not in {"ai", "default"}:
            raise ValueError("mode must be 'ai' or 'default'")
        locator = self.locator(selector)
        self.raw_aria = locator.aria_snapshot(
            timeout=self.timeout_ms,
            mode=mode,
            depth=depth,
            boxes=boxes,
        )
        self.aria = self.filter(
            self.raw_aria,
            exclude_roles=exclude_roles,
            exclude_patterns=exclude_patterns,
            exclude_properties=exclude_properties,
        )
        return self.aria

    def locator(self, selector: str) -> Locator:
        """Return a Playwright locator for the current page."""
        if self.page is None:
            raise RuntimeError("browser is not started; call start() or navigate() first")
        return self.page.locator(selector)

    def filter(
        self,
        snapshot: str | None = None,
        *,
        exclude_roles: Iterable[str] | None = None,
        exclude_patterns: Iterable[str] | None = None,
        exclude_properties: Iterable[str] | None = None,
    ) -> str:
        """Filter an ARIA snapshot without navigating or taking a new snapshot.

        Excluded roles and matching lines remove their entire indented subtree.
        Excluded properties remove bracketed attributes such as ``[level=2]``.
        Patterns are case-insensitive regular expressions.
        """
        source = self.raw_aria if snapshot is None else snapshot
        roles = self.exclude_roles if exclude_roles is None else self._terms(exclude_roles)
        patterns = self.exclude_patterns if exclude_patterns is None else tuple(exclude_patterns)
        properties = (
            self.exclude_properties
            if exclude_properties is None
            else self._terms(exclude_properties)
        )
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        property_pattern = (
            re.compile(r"\s*\[([\w-]+)(?:=[^\]]*)?\]", re.IGNORECASE)
            if properties
            else None
        )

        output = []
        skipped_indent = None
        for line in source.splitlines():
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if skipped_indent is not None:
                if not stripped or indent > skipped_indent:
                    continue
                skipped_indent = None

            role_match = re.match(r"-\s+([\w-]+)", stripped)
            role = role_match.group(1).casefold() if role_match else ""
            if role in roles or any(pattern.search(line) for pattern in compiled_patterns):
                skipped_indent = indent
                continue

            if property_pattern is not None:
                line = property_pattern.sub(
                    lambda match: "" if match.group(1).casefold() in properties else match.group(0),
                    line,
                ).rstrip()
            output.append(line)

        return "\n".join(output)

    def roles(self, snapshot: str | None = None) -> list[str]:
        """Return the distinct accessibility roles present in a snapshot."""
        source = self.raw_aria if snapshot is None else snapshot
        return sorted({
            match.group(1).casefold()
            for line in source.splitlines()
            if (match := re.match(r"\s*-\s+([\w-]+)", line))
        })

    def properties(self, snapshot: str | None = None) -> dict[str, list[str]]:
        """Return bracketed ARIA property names and their distinct values."""
        source = self.raw_aria if snapshot is None else snapshot
        found: dict[str, set[str]] = {}
        for name, value in re.findall(r"\[([\w-]+)(?:=([^\]]*))?\]", source):
            found.setdefault(name.casefold(), set()).add(value)
        return {name: sorted(values) for name, values in sorted(found.items())}

    def pattern_matches(
        self,
        patterns: Iterable[str] | None = None,
        snapshot: str | None = None,
    ) -> dict[str, list[str]]:
        """Return the lines matched by each case-insensitive regex pattern."""
        source = self.raw_aria if snapshot is None else snapshot
        selected = self.exclude_patterns if patterns is None else tuple(patterns)
        return {
            pattern: [
                line
                for line in source.splitlines()
                if re.search(pattern, line, re.IGNORECASE)
            ]
            for pattern in selected
        }

    def describe(
        self,
        snapshot: str | None = None,
        *,
        patterns: Iterable[str] | None = None,
    ) -> dict[str, object]:
        """Return serializable role, property, and pattern information."""
        source = self.raw_aria if snapshot is None else snapshot
        selected_patterns = self.exclude_patterns if patterns is None else tuple(patterns)
        return {
            "roles": self.roles(source),
            "properties": self.properties(source),
            "patterns": list(selected_patterns),
            "pattern_matches": self.pattern_matches(selected_patterns, source),
            "excluded_roles": sorted(self.exclude_roles),
            "excluded_properties": sorted(self.exclude_properties),
        }

    @staticmethod
    def _terms(values: Iterable[str]) -> set[str]:
        if isinstance(values, str):
            values = (values,)
        return {value.strip().casefold() for value in values if value.strip()}

    def close(self) -> None:
        """Close all Playwright resources owned by this object."""
        try:
            if self.context is not None:
                self.context.close()
            elif self.browser is not None:
                self.browser.close()
        finally:
            if self.playwright is not None:
                self.playwright.stop()
            self.page = None
            self.context = None
            self.browser = None
            self.playwright = None

    def __enter__(self) -> AriaPage:
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def extract(
    url: str,
    timeout: float = 30,
    *,
    exclude_roles: Iterable[str] = (),
    exclude_patterns: Iterable[str] = (),
    exclude_properties: Iterable[str] = (),
) -> str:
    """Return one filtered body ARIA snapshot and close the browser."""
    with AriaPage(
        timeout=timeout,
        exclude_roles=exclude_roles,
        exclude_patterns=exclude_patterns,
        exclude_properties=exclude_properties,
    ) as aria_page:
        return aria_page.navigate(url).aria
