"""The prompter's eyes on the app: a headless browser on the relay tunnel's local port.

It sees what a person sees: the page's visible text and its controls by their accessible
names. It is confined to the tunnel's origin (every request to another host is aborted),
so the click-through cannot reach the host's own services. Playwright is imported when a
browser is opened, so the rest of the package does not require it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse


class BrowserError(RuntimeError):
    pass


class Browser(Protocol):
    async def reachable(self) -> bool: ...
    async def observe(self) -> str: ...
    async def act(self, action: dict) -> str: ...
    async def screenshot(self, path: Path) -> Path: ...
    async def close(self) -> None: ...


ACTIONS = ("goto", "click", "fill", "select", "press", "wait", "back", "reload")

#: Page script: visible controls with their accessible names, as a person would find them.
_CONTROLS_JS = """() => {
  const out = [];
  const vis = e => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  for (const e of document.querySelectorAll(
      'a[href],button,input,select,textarea,[role=button],[role=link],[role=tab],[role=option],[role=menuitem],summary')) {
    if (!vis(e)) continue;
    let name = e.getAttribute('aria-label') || '';
    if (!name && e.labels && e.labels.length) name = e.labels[0].innerText;
    if (!name) name = e.getAttribute('placeholder') || e.innerText || e.value || e.title || '';
    const row = {tag: e.tagName.toLowerCase(), name: name.trim().replace(/\\s+/g, ' ').slice(0, 80)};
    if (e.type && e.tagName === 'INPUT') row.type = e.type;
    if (e.tagName === 'INPUT' || e.tagName === 'TEXTAREA' || e.tagName === 'SELECT') row.value = String(e.value).slice(0, 60);
    if (e.tagName === 'SELECT') row.options = [...e.options].map(o => o.text.trim()).slice(0, 12);
    if (e.disabled) row.disabled = true;
    const tid = e.getAttribute('data-testid'); if (tid) row.testid = tid;
    out.push(row);
  }
  return out.slice(0, 80);
}"""


def render_observation(url: str, title: str, text: str, controls: list[dict],
                       *, text_cap: int = 3500) -> str:
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) > text_cap:
        text = text[:text_cap] + "\n[... page continues ...]"
    rows = []
    for c in controls:
        bits = [c["tag"] + (f"[{c['type']}]" if c.get("type") else ""), json.dumps(c["name"])]
        if c.get("testid"):
            bits.append(f"testid={c['testid']}")
        if "value" in c and c["value"]:
            bits.append(f"value={json.dumps(c['value'])}")
        if c.get("options"):
            bits.append("options=" + json.dumps(c["options"]))
        if c.get("disabled"):
            bits.append("disabled")
        rows.append("  " + " ".join(bits))
    return (f"URL: {url}\nTITLE: {title}\nVISIBLE TEXT:\n{text}\nCONTROLS:\n"
            + ("\n".join(rows) if rows else "  (none)"))


async def http_reachable(base_url: str, timeout_s: float = 10.0) -> bool:
    """Does the app answer an HTTP request through the tunnel?"""
    import asyncio
    u = urlparse(base_url)
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(u.hostname, u.port or 80), timeout_s)
        try:
            writer.write(b"GET / HTTP/1.0\r\nHost: app\r\n\r\n")
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout_s)
        finally:
            writer.close()
        return line.startswith(b"HTTP/")
    except (OSError, asyncio.TimeoutError):
        return False


@dataclass
class PlaywrightBrowser:
    """Headless Chromium confined to `base_url`."""

    base_url: str
    executable_path: str | None = None
    timeout_ms: int = 8000

    def __post_init__(self) -> None:
        self._pw = None
        self._browser = None
        self._page = None
        self._origin = urlparse(self.base_url).netloc

    async def _ensure(self) -> None:
        if self._page is not None:
            return
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:                    # pragma: no cover - env dependent
            raise BrowserError("playwright is not installed (pip install playwright)") from exc
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True, executable_path=self.executable_path)
        ctx = await self._browser.new_context(viewport={"width": 1100, "height": 900})
        origin = self._origin

        async def guard(route):
            if urlparse(route.request.url).netloc == origin or route.request.url.startswith(
                    ("data:", "about:")):
                await route.continue_()
            else:
                await route.abort()

        await ctx.route("**/*", guard)
        ctx.set_default_timeout(self.timeout_ms)
        self._page = await ctx.new_page()
        await self._page.goto(self.base_url)

    async def reachable(self) -> bool:
        return await http_reachable(self.base_url)

    async def observe(self) -> str:
        await self._ensure()
        page = self._page
        last: Exception | None = None
        for _ in range(4):
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
                text = await page.inner_text("body")
                controls = await page.evaluate(_CONTROLS_JS)
                return render_observation(page.url, await page.title(), text, controls)
            except Exception as exc:                       # noqa: BLE001
                # a click that navigates destroys the page's context mid-read: look again
                if "context was destroyed" not in str(exc) and "navigat" not in str(exc).lower():
                    raise
                last = exc
                await page.wait_for_timeout(300)
        raise last

    def _locate(self, target: str):
        page = self._page
        if target.startswith("testid="):
            return page.get_by_test_id(target[7:]).first
        for locator in (page.get_by_role("button", name=target), page.get_by_role("link", name=target),
                        page.get_by_label(target), page.get_by_placeholder(target),
                        page.get_by_text(target)):
            yield_first = locator.first
            yield yield_first

    async def act(self, action: dict) -> str:
        await self._ensure()
        page = self._page
        kind = action.get("type")
        target, value = str(action.get("target", "")), str(action.get("value", ""))
        try:
            if kind == "goto":
                path = target if target.startswith("/") else "/" + target
                url = self.base_url.rstrip("/") + path
                try:
                    await page.goto(url)
                except Exception as exc:                  # noqa: BLE001
                    if "ERR_ABORTED" not in str(exc):
                        raise
                    # a navigation the app started (sign-in redirect) superseded this one
                    await self._settle()
                    await page.goto(url)
            elif kind == "back":
                await page.go_back()
            elif kind == "reload":
                await page.reload()
            elif kind == "wait":
                await page.wait_for_timeout(min(int(value or 800), 3000))
            elif kind == "press":
                await page.keyboard.press(value or target)
            elif kind in ("click", "fill", "select"):
                locator = await self._resolve(target)
                if kind == "click":
                    await locator.click()
                elif kind == "fill":
                    await locator.fill(value)
                else:
                    await locator.select_option(label=value)
            else:
                return f"unknown action {kind!r}"
            await self._settle()
        except Exception as exc:                      # noqa: BLE001 - the model reads the failure
            return f"action failed: {str(exc).splitlines()[0][:200]}"
        return "ok"

    async def _settle(self) -> None:
        """Let the page finish what the action started; a page that never idles is fine."""
        try:
            await self._page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:                             # noqa: BLE001
            pass
        await self._page.wait_for_timeout(250)

    async def _resolve(self, target: str):
        page = self._page
        if target.startswith("testid="):
            return page.get_by_test_id(target[7:]).first
        for locator in (page.get_by_role("option", name=target, exact=True),
                        page.get_by_role("menuitem", name=target, exact=True),
                        page.get_by_role("button", name=target), page.get_by_role("link", name=target),
                        page.get_by_label(target), page.get_by_placeholder(target),
                        page.get_by_text(target)):
            if await locator.count():
                return locator.first
        raise BrowserError(f"nothing on the page is named {target!r}")

    async def screenshot(self, path: Path) -> Path:
        await self._ensure()
        path.parent.mkdir(parents=True, exist_ok=True)
        await self._page.screenshot(path=str(path), full_page=False)
        return path

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._pw is not None:
            await self._pw.stop()
        self._page = self._browser = self._pw = None
