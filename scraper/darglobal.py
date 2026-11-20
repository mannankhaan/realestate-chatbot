"""DarGlobal (darglobal.co.uk) scraper.

The site is behind Imperva/Incapsula: plain HTTP gets a challenge page and copied cookies do not
work (TLS fingerprinting), so pages are rendered with Playwright driving installed Google Chrome.
The sitemap itself is fetched over plain HTTP. robots.txt is honoured; ~1 page / 2.5 s, one tab.

Usage:
    python -m scraper.darglobal              # full run (resumable)
    python -m scraper.darglobal --limit 5    # smoke test
    python -m scraper.darglobal --headed     # fallback if headless gets blocked
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import re
import sys
from urllib.parse import urlsplit

import httpx
from playwright.async_api import Page, async_playwright

from .common import (
    RAW_DIR,
    USER_AGENT,
    Counters,
    JsonlWriter,
    RateLimiter,
    Robots,
    clean_whitespace,
    load_done_ids,
    now_iso,
    setup_logging,
)

HOST = "https://darglobal.co.uk"  # www.darglobal.co.uk 301-redirects here (verified 2026-11-20)
SITEMAP = f"{HOST}/sitemap.xml"

# Transactional / thin / duplicate pages that add nothing for a Q&A bot.
SKIP_PATTERNS = [
    r"/pay-online", r"/landing-?page", r"/win-a-trip", r"/become-an?-", r"/get-in-touch",
    r"/thank-you", r"/bookYourUnit", r"/internship-application", r"/privacy-policy",
    r"/terms-", r"/broker-contest", r"/success$", r"/new-home-page", r"/campaigns$",
]
INFO_PAGES = {  # root-level pages that are about the company, not a single project
    "", "about", "why-invest", "tokenization", "faq", "hospitality", "commercial",
    "development-management", "investor", "dg-circle", "exclusive-membership", "insignia",
    "careers", "internships", "aston-martin-media", "live-all-in", "luxury-golf-communities",
    "sohar-islamic", "darglobal-exclusive-off-plan-investment-catalogue", "darglobal-insignia-card",
    "invest-crypto", "exclusive-plots", "blog", "press", "insights", "partners", "projects",
    "one-of-one",
}
BOILERPLATE_KEYS = {"globalData", "thankyouData", "thankyouPageData", "otherProjectData", "isMobile",
                    "locale", "inventoryDataLength"}
LIST_SECTIONS = {"blog", "press", "insights", "projects", "partners", "one-of-one"}

REMOVE_JS = """() => document.querySelectorAll(
  'header,nav,footer,script,style,noscript,iframe,svg,[class*=cookie],[id*=cookie],[class*=Cookie],'
  + '[class*=modal],[class*=Modal],[role=dialog]'
).forEach(e => e.remove())"""

log = setup_logging("darglobal")


class BlockedError(Exception):
    pass


def normalise(url: str) -> str:
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    return f"{HOST}{path}" if path else HOST


def section_of(url: str) -> str:
    path = urlsplit(url).path.strip("/")
    first = path.split("/")[0]
    if not path:
        return "home"
    if first in LIST_SECTIONS:
        return first if "/" in path else f"{first}-index"
    if first.lower() in INFO_PAGES:
        return "company"
    return "project"


def select_urls(locs: list[str], robots: Robots) -> list[str]:
    seen: set[str] = set()
    out = []
    for raw in locs:
        u = normalise(raw)
        if u in seen or any(re.search(p, u, re.I) for p in SKIP_PATTERNS):
            continue
        if not robots.allowed(u):
            continue
        seen.add(u)
        out.append(u)
    # projects and company pages first so a partial run already covers the important content
    order = {"home": 0, "project": 0, "company": 1, "projects-index": 1, "projects": 1,
             "one-of-one": 1, "one-of-one-index": 1, "partners": 2, "partners-index": 2,
             "insights": 3, "insights-index": 3, "press": 4, "press-index": 4, "blog": 5, "blog-index": 5}
    return sorted(out, key=lambda u: order.get(section_of(u), 9))


async def fetch_page(page: Page, url: str) -> dict:
    html = ""
    for attempt in range(3):
        await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        await page.wait_for_timeout(2_500 + attempt * 6_000)
        html = await page.content()
        if "Incapsula" not in html and "__NEXT_DATA__" in html:
            break
        log.info("challenge/no data on %s (attempt %d)", url, attempt + 1)
    else:
        raise BlockedError(url)
    next_data = json.loads(await page.locator("script#__NEXT_DATA__").inner_text())
    meta_desc = await page.locator('meta[name="description"]').first.get_attribute("content") \
        if await page.locator('meta[name="description"]').count() else ""
    await page.evaluate(REMOVE_JS)
    main = page.locator("main")
    text = await (main.first.inner_text() if await main.count() else page.inner_text("body"))
    if "Application error" in text[:200]:
        raise BlockedError("client-side application error page")
    page_props = next_data.get("props", {}).get("pageProps", {}) or {}
    # Content lives in pageData (listing pages) or projectDetailsData (project pages); keep all
    # content keys, drop site-wide boilerplate (country dial-code list, thank-you pages, ...).
    content = {k: v for k, v in page_props.items() if k not in BOILERPLATE_KEYS and v not in (None, "", [], {})}
    return {
        "url": url,
        "final_url": page.url,
        "site": "darglobal",
        "section": section_of(url),
        "title": await page.title(),
        "description": meta_desc or "",
        "text": clean_whitespace(text),
        "next_page_data": content,
        "fetched_at": now_iso(),
    }


async def run(limit: int | None, headed: bool, rate_s: float) -> int:
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30) as c:
        robots = Robots(c.get(f"{HOST}/robots.txt").text)
        locs = [html.unescape(x) for x in re.findall(r"<loc>(.*?)</loc>", c.get(SITEMAP).text)]  # &apos; etc.
    urls = select_urls(locs, robots)
    log.info("sitemap: %d locs -> %d selected", len(locs), len(urls))

    out = RAW_DIR / "darglobal.jsonl"
    fail_path = RAW_DIR / "darglobal_failures.jsonl"
    done = load_done_ids(out)
    todo = [u for u in urls if u not in done][: limit or None]
    c = Counters()
    c.skipped = len(urls) - len([u for u in urls if u not in done])
    rl = RateLimiter(rate_s)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not headed, channel="chrome")

        async def new_context():
            ctx = await browser.new_context(
                locale="en-GB", viewport={"width": 1440, "height": 900}, user_agent=USER_AGENT
            )
            await ctx.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in {"image", "media", "font"}
                else route.continue_(),
            )
            pg = await ctx.new_page()
            await pg.goto(HOST + "/", wait_until="domcontentloaded", timeout=60_000)  # warm-up
            await pg.wait_for_timeout(6_000)
            return ctx, pg

        ctx, page = await new_context()
        consecutive_blocks = 0
        with JsonlWriter(out) as jw, JsonlWriter(fail_path) as fw:
            for i, u in enumerate(todo, 1):
                await rl.await_wait()
                try:
                    rec = await fetch_page(page, u)
                    if len(rec["text"]) < 80:
                        raise BlockedError(f"too little text ({len(rec['text'])} chars)")
                    jw.write(rec)
                    c.fetched += 1
                    consecutive_blocks = 0
                except Exception as e:  # noqa: BLE001
                    c.failed += 1
                    consecutive_blocks += 1
                    log.warning("failed %s: %s", u, str(e)[:200])
                    fw.write({"url": u, "reason": str(e)[:300], "at": now_iso()})
                    if consecutive_blocks >= 3:  # fallback: fresh context + slower rate
                        log.warning("3 consecutive failures -> new browser context, slowing down")
                        await ctx.close()
                        ctx, page = await new_context()
                        rl.min_interval_s = max(rl.min_interval_s, 5.0)
                        consecutive_blocks = 0
                if i % 25 == 0:
                    log.info("[%d/%d] %s", i, len(todo), c)
        await ctx.close()
        await browser.close()
    log.info("done: %s -> %s", c, out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--rate", type=float, default=2.5)
    a = ap.parse_args(argv)
    return asyncio.run(run(a.limit, a.headed, a.rate))


if __name__ == "__main__":
    sys.exit(main())
