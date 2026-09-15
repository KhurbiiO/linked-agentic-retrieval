"""Validate browser actions and refresh evidence after confirmed page changes."""

from __future__ import annotations

import re
from time import monotonic
from typing import Any

from playwright.sync_api import Error as PlaywrightError


_ROLE_SELECTOR = re.compile(
    r"^role\s*=\s*([\w-]+)\s*,\s*name\s*=\s*(.+)$", re.IGNORECASE
)
_TARGET_STATE = """element => ({
    value: 'value' in element ? element.value : null,
    expanded: element.getAttribute('aria-expanded'),
    selected: element.getAttribute('aria-selected'),
    pressed: element.getAttribute('aria-pressed'),
    checked: 'checked' in element ? element.checked : element.getAttribute('aria-checked'),
    text: element.textContent,
})"""


class NavigationNotConfirmed(ValueError):
    """A requested page transition could not be confirmed before the deadline."""


def _target_fields(decision: Any) -> tuple[str, str, str]:
    role = (decision.role or "").strip().lower()
    name = (decision.name or "").strip()
    selector = (decision.selector or "").strip()
    repaired = _ROLE_SELECTOR.fullmatch(selector)
    if not role and repaired:
        role, name = repaired.group(1).lower(), repaired.group(2).strip("\"'")
        selector = ""
    return role, name, selector


def canonical_target_key(decision: Any) -> tuple[str, str, str, str, str]:
    """Normalize action spelling for failure history without resolving the DOM."""
    role, name, selector = _target_fields(decision)
    if role:
        selector = ""
    return decision.action, role, name, selector, decision.value or ""


def _remaining(deadline: float) -> float:
    remaining = (deadline - monotonic()) * 1000
    if remaining <= 0:
        raise ValueError("Browser action timed out before page state was confirmed")
    return remaining


def _visible_first(locator: Any) -> Any | None:
    for index in range(locator.count()):
        candidate = locator.nth(index)
        if candidate.is_visible():
            return candidate
    return None


def resolve_action_target(page: Any, decision: Any) -> tuple[Any, dict[str, str]]:
    """Resolve exact visible accessible names before falling back to CSS."""
    role, name, selector = _target_fields(decision)
    if role:
        if not name:
            raise ValueError(f"{decision.action} requires an accessible name with role={role!r}")
        target = _visible_first(page.get_by_role(role, name=name, exact=True))
        if target is None:
            raise ValueError(f"No visible {role!r} with exact accessible name {name!r}")
        return target, {"role": role, "name": name}
    if not selector:
        raise ValueError(f"{decision.action} requires role/name or a CSS selector")

    # Small models often put a visible link label in the selector field. Resolve
    # it only when that exact accessible label really exists on the current page.
    for candidate_role in ("link", "button", "textbox", "searchbox"):
        target = _visible_first(page.get_by_role(candidate_role, name=selector, exact=True))
        if target is not None:
            return target, {"role": candidate_role, "name": selector}
    if selector.lower().startswith("role="):
        raise ValueError("Unsupported role selector; provide separate role and name fields")
    try:
        target = _visible_first(page.locator(selector))
    except PlaywrightError as exc:
        raise ValueError(f"Invalid CSS selector {selector!r}; use role/name for visible text") from exc
    if target is None:
        raise ValueError(f"No visible element matches CSS selector {selector!r}")
    return target, {"selector": selector}


def _meaningful_aria(snapshot: str) -> str:
    """Ignore generated references, link metadata, and explicitly labelled ads."""
    lines: list[str] = []
    skipped_indent: int | None = None
    for line in snapshot.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if skipped_indent is not None:
            if not stripped or indent > skipped_indent:
                continue
            skipped_indent = None
        if re.search(r'\b(?:advertisement|advertising|ad banner)\b', stripped, re.I):
            skipped_indent = indent
            continue
        if re.match(r"-\s*/(?:url|href|box|bounds):", stripped):
            continue
        line = re.sub(r"\s*\[(?:ref|box|bounds)=[^\]]*\]", "", line)
        if line.strip():
            lines.append(line.rstrip())
    return "\n".join(lines)


def _body_snapshot(aria_page: Any, deadline: float) -> str:
    raw = aria_page.page.locator("body").aria_snapshot(timeout=_remaining(deadline), mode="ai")
    aria_page.raw_aria = raw
    aria_page.aria = aria_page.filter(raw)
    return raw


def _settle(aria_page: Any, deadline: float) -> bool:
    """Allow a short quiet period after DOM readiness, bounded by the caller."""
    page = aria_page.page
    page.wait_for_load_state("domcontentloaded", timeout=_remaining(deadline))
    settle_deadline = min(deadline, monotonic() + 1.5)
    previous: str | None = None
    stable_since = monotonic()
    while monotonic() < settle_deadline:
        current = _meaningful_aria(_body_snapshot(aria_page, deadline))
        now = monotonic()
        if current != previous or not current:
            previous = current
            stable_since = now
        elif now - stable_since >= 0.4:
            return True
        page.wait_for_timeout(min(100, max(1, (settle_deadline - now) * 1000)))
    return False


def refresh_stable_snapshot(aria_page: Any, *, timeout_ms: float) -> str:
    """Refresh full body ARIA after DOM readiness with a bounded quiet period."""
    if aria_page.page is None or timeout_ms <= 0:
        raise ValueError("A running page and positive snapshot timeout are required")
    _settle(aria_page, monotonic() + timeout_ms / 1000)
    return aria_page.aria


def execute_browser_action(
    aria_page: Any,
    decision: Any,
    *,
    action_timeout_ms: float,
    navigation_timeout_ms: float,
) -> dict[str, Any]:
    """Execute a bounded action and leave full, fresh body ARIA on ``aria_page``.

    Link clicks require a URL change or popup; other controls require an
    observable content/target state change. A successful click alone does not
    establish that navigation happened. No destination URL is extracted or set.
    """
    page = aria_page.page
    if page is None:
        raise ValueError("ARIA page is not running")
    if action_timeout_ms <= 0 or navigation_timeout_ms <= 0:
        raise ValueError("Action and navigation timeouts must be positive")
    action_deadline = monotonic() + action_timeout_ms / 1000
    before_url = page.url
    before_aria = _body_snapshot(aria_page, action_deadline)
    before_meaningful = _meaningful_aria(before_aria)
    metadata: dict[str, Any] = {
        "before_url": before_url,
        "target": {},
        "javascript_fallback": False,
    }

    def finish(status: str, *, stable: bool = True) -> dict[str, Any]:
        return {
            **metadata,
            "after_url": aria_page.page.url,
            "url_changed": aria_page.page.url != before_url,
            "aria_changed": _meaningful_aria(aria_page.raw_aria) != before_meaningful,
            "status": status,
            "snapshot_stable": stable,
        }

    if decision.action == "stop":
        return finish("stopped")
    if decision.action == "snapshot":
        if not decision.selector:
            raise ValueError("snapshot requires a focused CSS selector")
        metadata["focused_aria"] = page.locator(decision.selector).aria_snapshot(
            timeout=_remaining(action_deadline), mode="ai"
        )
        _body_snapshot(aria_page, action_deadline)
        return finish("inspected")

    popups: list[Any] = []
    def record_popup(popup: Any) -> None:
        popups.append(popup)

    page.on("popup", record_popup)
    try:
        requires_navigation = decision.action == "back"
        target = None
        before_target = None
        if decision.action == "back":
            page.go_back(wait_until="domcontentloaded", timeout=_remaining(action_deadline))
        else:
            if decision.action not in {"click", "fill", "press"}:
                raise ValueError(f"Unsupported controller action: {decision.action}")
            if decision.action in {"fill", "press"} and decision.value is None:
                raise ValueError(f"{decision.action} requires value")
            target, metadata["target"] = resolve_action_target(page, decision)
            before_target = target.evaluate(_TARGET_STATE, timeout=_remaining(action_deadline))
            if decision.action == "click":
                requires_navigation = target.evaluate(
                    """element => {
                        const link = element.closest('a[href]:not([download])');
                        return !!link && !link.hasAttribute('aria-controls')
                            && !link.hasAttribute('aria-expanded');
                    }""",
                    timeout=_remaining(action_deadline),
                )
                # Keep a small portion of the action budget for an intercepted
                # click fallback. It never applies to invalid/missing targets.
                try:
                    target.click(timeout=max(1, _remaining(action_deadline) * 0.8))
                except PlaywrightError as exc:
                    if page.url != before_url or popups:
                        pass  # Click dispatched; navigation can outlast its timeout.
                    elif re.search(r"intercepts pointer events|intercepted", str(exc), re.I):
                        if not target.is_visible():
                            raise ValueError("Intercepted click target is no longer visible") from exc
                        target.evaluate("element => element.click()", timeout=_remaining(action_deadline))
                        metadata["javascript_fallback"] = True
                    else:
                        raise
            elif decision.action == "fill":
                target.fill(decision.value, timeout=_remaining(action_deadline))
            else:
                target.press(decision.value, timeout=_remaining(action_deadline))

        navigation_deadline = monotonic() + navigation_timeout_ms / 1000
        status = ""
        while monotonic() < navigation_deadline:
            if popups:
                # Adopt only the popup emitted by this page during this action.
                aria_page.page = popups[0]
                status = "popup_opened"
                break
            if page.url != before_url:
                status = "navigated"
                break
            try:
                raw = _body_snapshot(aria_page, navigation_deadline)
            except (PlaywrightError, ValueError):
                # A navigation may destroy the document between URL inspection
                # and snapshot capture. Re-check the resulting page while budget
                # remains, never handing an old snapshot to the Builder.
                remaining_ms = (navigation_deadline - monotonic()) * 1000
                if remaining_ms <= 0:
                    break
                page.wait_for_timeout(min(100, remaining_ms))
                continue
            if not requires_navigation:
                if decision.action == "fill":
                    status = "filled"
                    break
                state_changed = False
                if target is not None:
                    try:
                        state_changed = (
                            target.evaluate(_TARGET_STATE, timeout=_remaining(navigation_deadline))
                            != before_target
                        )
                    except PlaywrightError:
                        # An element replaced by client-side rendering is not
                        # itself evidence; a body content change still is.
                        pass
                if state_changed or _meaningful_aria(raw) != before_meaningful:
                    status = "content_changed"
                    break
            remaining_ms = (navigation_deadline - monotonic()) * 1000
            if remaining_ms > 0:
                page.wait_for_timeout(min(100, remaining_ms))
        if not status:
            expected = "URL change or popup" if requires_navigation else "content or control-state change"
            raise NavigationNotConfirmed(
                f"{decision.action} did not produce a confirmed {expected} within "
                f"{navigation_timeout_ms:g} ms; current URL is {page.url!r}"
            )
        try:
            stable = _settle(aria_page, navigation_deadline)
        except (PlaywrightError, ValueError) as exc:
            raise NavigationNotConfirmed(
                "A page transition started, but its DOM/ARIA was not ready before the deadline"
            ) from exc
        return finish(status, stable=stable)
    finally:
        page.remove_listener("popup", record_popup)
