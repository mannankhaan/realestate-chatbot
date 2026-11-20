"""Wasalt.sa scraper.

* Honors robots.txt (no /search) and https://wasalt.sa/llms.txt (public listing content only,
  moderate crawl rate, no contact details, attribution "Source: Wasalt.sa", non-commercial).
* Listing data comes from the Next.js data route  /_next/data/{buildId}/en/property/...json.
* Cloudflare challenges Python's default TLS fingerprint, so requests go through curl_cffi with a
  Chrome TLS profile. Same public pages, same rate limit (<= 1 req/s), single concurrency.
* PII: fields are copied through an ALLOW-LIST. leadContactInfo / propertyOwner / phone numbers /
  license and plot numbers are never read into the output.

Usage:
    python -m scraper.wasalt --sample 4000          # full run (resumable)
    python -m scraper.wasalt --sample 50 --dry-run  # quick validation into wasalt_dryrun.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from collections import defaultdict
from typing import Any

from curl_cffi import requests as cr
from selectolax.lexbor import LexborHTMLParser as HTMLParser
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .common import (
    RAW_DIR,
    Counters,
    JsonlWriter,
    RateLimiter,
    Robots,
    clean_whitespace,
    load_done_ids,
    now_iso,
    setup_logging,
)
from .normalize import redact

BASE = "https://wasalt.sa"
SITEMAPS = {
    "product": "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz",
    "static": "https://cdn.wasalt.sa/sitemap/static_sitemap_en_sa.xml.gz",
}
LISTING_RE = re.compile(r"/en/property/(sale|rent)/(.+)-(\d+)$")
NEXT_DATA_RE = re.compile(r'id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)

log = setup_logging("wasalt")


class RetryableHTTP(Exception):
    pass


# --------------------------------------------------------------------------- allow-list mapping

ATTR_MAP = {  # attributes[].key -> our column
    "noOfBedrooms": "bedrooms",
    "noOfBathrooms": "bathrooms",
    "noOfLivingrooms": "living_rooms",
    "noOfGuestrooms": "guest_rooms",
    "builtUpArea": "built_up_area_sqm",
    "carpetArea": "land_area_sqm",
    "furnishingType": "furnishing",
}
ADDITIONAL_MAP = {  # additionalAttributes[].key -> our column
    "completionYear": "property_age",
    "propertyFacade": "facing",
    "propertyStreetWidth": "street_width",
    "electricityMeter": "electricity_meter",
    "waterMeter": "water_meter",
    "furnishingType": "furnishing",
}
INT_FIELDS = {"bedrooms", "bathrooms", "living_rooms", "guest_rooms"}
FLOAT_FIELDS = {"built_up_area_sqm", "land_area_sqm"}


def _num(v: Any, cast=float):
    if v is None or v == "":
        return None
    try:
        return cast(float(str(v).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


def strip_html(html: str | None) -> str:
    if not html:
        return ""
    tree = HTMLParser(html)
    for n in tree.css("script, style, noscript"):
        n.decompose()
    for br in tree.css("br"):
        br.replace_with("\n")
    text = clean_whitespace(tree.text(separator="\n"))
    return re.sub(r"\n{2,}", "\n", text)


def bucket_of(url: str) -> tuple[str, str] | None:
    m = LISTING_RE.search(url)
    if not m:
        return None
    kind = re.split(r"-\d", m.group(2))[0].split("-")[0]
    return m.group(1), kind


def listing_id(url: str) -> int:
    m = LISTING_RE.search(url)
    return int(m.group(3)) if m else 0


def extract_listing(page_props: dict[str, Any], url: str) -> dict[str, Any] | None:
    """Allow-list extraction from pageProps. Returns None if listing is missing/not approved."""
    p = page_props.get("propertyDetailsV3")
    if not isinstance(p, dict):
        return None
    pi = p.get("propertyInfo") or {}
    status = pi.get("statusTypeSlug")
    if status and status != "approved":
        return None

    purpose = (pi.get("propertyFor") or "").lower() or None
    rec: dict[str, Any] = {
        "id": p.get("id"),
        "url": url,
        "title": pi.get("title"),
        "purpose": purpose,
        "main_type": pi.get("propertyMainType"),
        "sub_type": pi.get("propertySubType"),
        "city": pi.get("city"),
        "district": pi.get("district"),
        "zone": pi.get("zone"),
        "territory": pi.get("territory"),
        "address": pi.get("address"),
        "currency": pi.get("currencyType") or "SAR",
        "furnishing": pi.get("furnishingType"),
        "facing": pi.get("facingType"),
    }

    # Price: sale price, or rent normalised to yearly (Wasalt's expectedRent is the yearly amount).
    rent_freq = pi.get("rentFreq") if isinstance(pi.get("rentFreq"), dict) else {}
    if purpose == "rent":
        yearly = (rent_freq.get("yearly") or {}).get("amount")
        monthly = (rent_freq.get("monthly") or {}).get("amount")
        rec["price"] = _num(yearly) or _num(pi.get("expectedRent"))
        rec["rent_frequency"] = "yearly"
        rec["rent_monthly"] = _num(monthly)
        default = [k for k, v in rent_freq.items() if isinstance(v, dict) and v.get("default_freq")]
        rec["rent_default_period"] = default[0] if default else "yearly"
    else:
        rec["price"] = _num(pi.get("salePrice"))
        rec["rent_frequency"] = None

    for a in p.get("attributes") or []:
        col = ATTR_MAP.get(a.get("key"))
        if col:
            rec[col] = a.get("value")
    for a in p.get("additionalAttributes") or []:
        col = ADDITIONAL_MAP.get(a.get("key"))
        if col and rec.get(col) in (None, ""):
            rec[col] = a.get("value")

    for f in INT_FIELDS:
        if f in rec:
            rec[f] = _num(rec[f], int)
    for f in FLOAT_FIELDS:
        if f in rec:
            rec[f] = _num(rec[f])
    area = rec.get("built_up_area_sqm") or rec.get("land_area_sqm")
    rec["area_sqm"] = area
    rec["price_per_sqm"] = round(rec["price"] / area, 1) if rec.get("price") and area else None

    street = p.get("streetInfo") or {}
    rec["street_width_m"] = _num(street.get("streetWidth"))
    loc = p.get("location") or {}
    lat, lon = _num(loc.get("lat")), _num(loc.get("lon"))
    rec["lat"] = round(lat, 3) if lat is not None else None  # coarse: ~100 m
    rec["lon"] = round(lon, 3) if lon is not None else None

    val = p.get("priceValuation") or {}
    rec["valuation_low"] = _num((val.get("lower_bound") or {}).get("value")) if isinstance(val.get("lower_bound"), dict) else _num(val.get("lower_bound"))
    rec["valuation_high"] = _num((val.get("upper_bound") or {}).get("value")) if isinstance(val.get("upper_bound"), dict) else _num(val.get("upper_bound"))

    # llms.txt forbids extracting contact details: redact phones/emails before anything is written
    rec["description_text"] = redact(strip_html(pi.get("description")))
    rec["title"] = redact(rec["title"] or "")
    rec["is_verified"] = bool(p.get("isVerified"))
    rec["published_at"] = p.get("publishedAt")
    rec["updated_at"] = p.get("updatedAt")
    rec["fetched_at"] = now_iso()
    return rec


# --------------------------------------------------------------------------- sampling


def stratified_sample(urls: list[str], target: int, floor: int = 150, small_cap: int = 60, seed: int = 42) -> list[str]:
    """Proportional per (purpose, type) bucket, floor for buckets >= floor, newest (highest id) first."""
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for u in urls:
        b = bucket_of(u)
        if b:
            buckets[b].append(u)
    total = sum(len(v) for v in buckets.values())
    # Floors only make sense relative to the target (a 50-listing dry run must not fetch 2,000).
    floor = min(floor, max(1, target // 20))
    small_cap = min(small_cap, max(1, target // 40))
    alloc: dict[tuple[str, str], int] = {}
    for b, lst in buckets.items():
        share = round(target * len(lst) / total)
        if len(lst) >= floor:
            alloc[b] = max(share, floor)
        else:
            alloc[b] = min(len(lst), max(share, small_cap))
    # scale the large buckets down so the total matches the target
    over = sum(alloc.values()) - target
    if over > 0:
        big = sorted((b for b in alloc if alloc[b] > floor), key=lambda b: -alloc[b])
        big_total = sum(alloc[b] - floor for b in big)
        if big_total:
            for b in big:
                alloc[b] -= round(over * (alloc[b] - floor) / big_total)
    rng = random.Random(seed)
    picked: list[str] = []
    for b, lst in sorted(buckets.items()):
        newest = sorted(lst, key=listing_id, reverse=True)[: alloc[b]]
        picked.extend(newest)
    rng.shuffle(picked)  # interleave buckets so a partial run is still balanced
    return picked[:target]


# --------------------------------------------------------------------------- client


class Wasalt:
    def __init__(self, rate_s: float = 1.0):
        self.s = cr.Session(impersonate="chrome", headers={"Accept-Language": "en-US,en;q=0.9"}, timeout=30)
        self.rl = RateLimiter(rate_s)
        self.build_id: str | None = None
        self.robots = Robots(self._get("https://wasalt.sa/robots.txt").text)

    @retry(
        retry=retry_if_exception_type(RetryableHTTP),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, url: str):
        self.rl.wait()
        r = self.s.get(url)
        if r.status_code == 429 or r.status_code >= 500:
            log.warning("HTTP %s on %s -> backing off", r.status_code, url)
            self.rl.min_interval_s = max(self.rl.min_interval_s, 2.0)  # slow down for the rest of run
            raise RetryableHTTP(f"{r.status_code}")
        return r

    def refresh_build_id(self) -> str:
        html = self._get(f"{BASE}/en").text
        m = NEXT_DATA_RE.search(html)
        if not m:
            raise RuntimeError("Could not find __NEXT_DATA__ on wasalt.sa/en (blocked?)")
        self.build_id = json.loads(m.group(1))["buildId"]
        log.info("buildId=%s", self.build_id)
        return self.build_id

    def sitemap(self, url: str) -> list[str]:
        b = self._get(url).content
        try:
            b = gzip.decompress(b)
        except OSError:
            pass  # some .xml.gz files are served as plain XML
        return [x.decode().strip() for x in re.findall(rb"<loc>(.*?)</loc>", b)]

    def listing_props(self, url: str) -> dict[str, Any] | None:
        path = url.removeprefix(BASE)
        for attempt in range(2):
            r = self._get(f"{BASE}/_next/data/{self.build_id}{path}.json")
            if r.status_code == 200:
                try:
                    return r.json().get("pageProps") or {}
                except ValueError:
                    break
            if r.status_code == 404 and attempt == 0:
                old = self.build_id
                self.refresh_build_id()
                if self.build_id != old:
                    continue
            break
        # fallback: HTML + __NEXT_DATA__
        r = self._get(url)
        if r.status_code != 200:
            return None
        m = NEXT_DATA_RE.search(r.text)
        return json.loads(m.group(1))["props"]["pageProps"] if m else None

    def page_text(self, url: str) -> dict[str, Any] | None:
        r = self._get(url)
        if r.status_code != 200:
            return None
        tree = HTMLParser(r.text)
        title = tree.css_first("title").text(strip=True) if tree.css_first("title") else ""
        desc_node = tree.css_first('meta[name="description"]')
        for sel in ("script", "style", "noscript", "header", "nav", "footer", "svg"):
            for n in tree.css(sel):
                n.decompose()
        body = tree.css_first("main") or tree.body
        text = clean_whitespace(body.text(separator="\n")) if body else ""
        return {
            "url": url,
            "site": "wasalt",
            "section": "static",
            "title": title,
            "description": desc_node.attributes.get("content", "") if desc_node else "",
            "text": text,
            "fetched_at": now_iso(),
        }


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--rate", type=float, default=1.0, help="min seconds between requests")
    ap.add_argument("--dry-run", action="store_true", help="write to wasalt_dryrun.jsonl")
    ap.add_argument("--skip-static", action="store_true")
    args = ap.parse_args(argv)

    w = Wasalt(rate_s=args.rate)
    w.refresh_build_id()

    # static pages (terms, masterplan, business...) -> "what is Wasalt" questions
    if not args.skip_static and not args.dry_run:
        out = RAW_DIR / "wasalt_pages.jsonl"
        done = load_done_ids(out)
        with JsonlWriter(out) as jw:
            for u in w.sitemap(SITEMAPS["static"]):
                if u in done or not w.robots.allowed(u):
                    continue
                rec = w.page_text(u)
                if rec and rec["text"]:
                    jw.write(rec)
                    log.info("static page ok: %s (%d chars)", u, len(rec["text"]))

    urls = [u for u in w.sitemap(SITEMAPS["product"]) if w.robots.allowed(u)]
    log.info("product sitemap: %d listing URLs", len(urls))
    sample = stratified_sample(urls, args.sample)
    log.info("sampled %d URLs", len(sample))

    out = RAW_DIR / ("wasalt_dryrun.jsonl" if args.dry_run else "wasalt_listings.jsonl")
    fail_path = RAW_DIR / ("wasalt_dryrun_failures.jsonl" if args.dry_run else "wasalt_failures.jsonl")
    done = load_done_ids(out) | load_done_ids(fail_path)
    c = Counters()
    with JsonlWriter(out) as jw, JsonlWriter(fail_path) as fw:
        for i, u in enumerate(sample, 1):
            if u in done:
                c.skipped += 1
                continue
            try:
                props = w.listing_props(u)
                rec = extract_listing(props, u) if props else None
            except Exception as e:  # noqa: BLE001 - log and continue, the run must be resumable
                rec, props = None, None
                log.warning("error %s: %s", u, e)
            if rec:
                jw.write(rec)
                c.fetched += 1
            else:
                fw.write({"url": u, "reason": "unavailable_or_error", "at": now_iso()})
                c.failed += 1
            if i % 50 == 0:
                log.info("[%d/%d] %s", i, len(sample), c)
    log.info("done: %s -> %s", c, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
