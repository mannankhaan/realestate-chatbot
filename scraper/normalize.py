"""raw -> processed.

Outputs (data/processed/):
  documents.jsonl   text docs for RAG (DarGlobal pages, Wasalt static pages, one fact sheet per listing)
  listings.jsonl    Wasalt structured listings (allow-listed fields only)
  projects.jsonl    DarGlobal structured projects
  scrape_report.json  counts, dates, failures (served at /api/stats)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from typing import Any

from .common import PROCESSED_DIR, RAW_DIR, clean_whitespace, now_iso, read_jsonl, setup_logging

log = setup_logging("normalize")

# ----------------------------------------------------------------------------- PII redaction

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RES = [
    re.compile(r"(?<![\w])(?:\+|00)\d[\d\s()-]{7,}\d"),                    # international
    re.compile(r"(?<![\d,.])(?:966|0)?5\d(?:[\s-]?\d){7}(?![\d,])"),       # KSA mobile 05xxxxxxxx
    re.compile(r"(?<![\d,.])(?:800|920)[\s-]?\d{2,3}[\s-]?\d{3,4}(?![\d,])"),  # KSA toll-free / unified
    re.compile(r"(?<![\d,.])\d{3,4}[\s-]\d{3}[\s-]\d{3,4}(?![\d,])"),      # 3-3-4 style
]
LINK_RE = re.compile(r"(?:https?://)?(?:wa\.me|api\.whatsapp\.com|chat\.whatsapp\.com)/\S*", re.I)
FORBIDDEN_KEYS = ("ownername", "resemployee", "leadcontactinfo", "propertyowner")


def redact(text: str) -> str:
    if not text:
        return text
    text = LINK_RE.sub("[redacted]", text)
    text = EMAIL_RE.sub("[redacted]", text)
    for rx in PHONE_RES:
        text = rx.sub("[redacted]", text)
    return text


# ----------------------------------------------------------------------------- helpers

UI_LINES = {
    "download brochure", "book unit", "gallery", "view all images", "live feed", "enquire now",
    "register your interest", "read more", "load more", "view all", "explore", "learn more",
    "submit", "next", "previous", "share", "back", "view project", "know more", "discover more",
}


def strapi_text(blocks: Any) -> str:
    """Flatten Strapi rich-text blocks (or plain strings) into text."""
    if blocks is None:
        return ""
    if isinstance(blocks, str):
        return blocks
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for n in node:
                walk(n)
        elif isinstance(node, dict):
            if "text" in node and isinstance(node["text"], str):
                out.append(node["text"])
            for k in ("children",):
                if k in node:
                    walk(node[k])
            if node.get("type") in ("paragraph", "list-item", "heading", "list"):
                out.append("\n")

    walk(blocks)
    return clean_whitespace("".join(out))


def find_dicts(o: Any, pred) -> Iterable[dict]:
    if isinstance(o, dict):
        if pred(o):
            yield o
        for v in o.values():
            yield from find_dicts(v, pred)
    elif isinstance(o, list):
        for v in o:
            yield from find_dicts(v, pred)


def faq_text(page_data: Any) -> str:
    qas = []
    for d in find_dicts(page_data, lambda d: "question" in d and "answer" in d):
        q, a = strapi_text(d.get("question")), strapi_text(d.get("answer"))
        if q and a:
            qas.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(dict.fromkeys(qas))


def remove_boilerplate(pages: list[dict], threshold: float = 0.3) -> dict[str, str]:
    """Drop lines that repeat across > threshold of pages (nav/footer leftovers) and UI labels."""
    freq: Counter[str] = Counter()
    for p in pages:
        freq.update({ln.strip() for ln in p["text"].split("\n") if ln.strip()})
    limit = max(3, int(threshold * len(pages)))
    common = {ln for ln, c in freq.items() if c > limit}
    log.info("boilerplate: %d lines repeated on >%d pages", len(common), limit)
    out = {}
    for p in pages:
        kept = [
            ln for ln in p["text"].split("\n")
            if ln.strip() not in common and ln.strip().lower() not in UI_LINES
        ]
        out[p["url"]] = clean_whitespace("\n".join(kept))
    return out


def doc_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode(), usedforsecurity=False).hexdigest()[:16]


# ----------------------------------------------------------------------------- DarGlobal

LABELS = {
    "PROPERTY TYPE": "type", "UNIT TYPE": "unit_types", "UNIT TYPES": "unit_types", "STATUS": "status",
    "AREA (SQFT)": "area_sqft", "AREA (SQM)": "area_sqm", "AREA": "area_sqft", "BEDROOMS": "bedrooms",
    "LOCATION": "location", "COMPLETION": "completion", "HANDOVER": "completion",
    "STARTING PRICE": "starting_price", "PRICE": "starting_price", "FEATURING": "featuring",
    "DESIGNED BY": "designed_by", "IN COLLABORATION WITH": "collaboration", "BRAND": "collaboration",
}
COUNTRY_ALIASES = {
    "UAE": "UAE", "UNITED ARAB EMIRATES": "UAE", "DUBAI": "UAE", "ABU DHABI": "UAE",
    "KSA": "Saudi Arabia", "SAUDI ARABIA": "Saudi Arabia", "JEDDAH": "Saudi Arabia", "RIYADH": "Saudi Arabia",
    "OMAN": "Oman", "MUSCAT": "Oman", "QATAR": "Qatar", "DOHA": "Qatar", "SPAIN": "Spain",
    "MARBELLA": "Spain", "UK": "United Kingdom", "UNITED KINGDOM": "United Kingdom", "LONDON": "United Kingdom",
    "BOSNIA": "Bosnia and Herzegovina", "BOSNIA AND HERZEGOVINA": "Bosnia and Herzegovina",
    "MALDIVES": "Maldives", "USA": "USA", "UNITED STATES": "USA", "MIAMI": "USA", "GREECE": "Greece",
    "ITALY": "Italy", "MONTENEGRO": "Montenegro", "SERBIA": "Serbia", "BAHRAIN": "Bahrain",
    "AL MARJAN ISLAND": "UAE", "RAS AL KHAIMAH": "UAE", "MARBELLA ": "Spain", "BENAHAVIS": "Spain",
    "BENAHAVÍS": "Spain", "AIDA": "Oman", "DOHA, QATAR": "Qatar",
}


def parse_location(loc: str) -> tuple[str | None, str | None]:
    if not loc:
        return None, None
    parts = [p.strip() for p in re.split(r"[|,]", loc) if p.strip()]
    country = None
    for p in reversed(parts):
        country = COUNTRY_ALIASES.get(p.upper())
        if country:
            break
    city = parts[0].title() if parts else None
    return city, country


def project_from_page(rec: dict, text: str) -> dict | None:
    pd = rec.get("next_page_data") or {}
    det = (pd.get("projectDetailsData") or {}).get("attributes")
    if rec["section"] != "project" or not det:
        return None
    banner = (det.get("ProjectDetails") or {}).get("ProjectBanner") or {}
    about = (det.get("ProjectDetails") or {}).get("AboutProject") or {}
    p: dict[str, Any] = {
        "name": det.get("title") or banner.get("Heading") or rec["title"],
        "url": rec["url"],
        "location": banner.get("Location"),
        "summary": strapi_text(about.get("description"))[:600],
    }
    # "LABEL\nvalue" pairs from the rendered fact box
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for i, ln in enumerate(lines[:-1]):
        key = LABELS.get(ln.upper().rstrip(":"))
        if key and key not in p and len(lines[i + 1]) < 160:
            p[key] = lines[i + 1]
    if not p.get("location"):
        m = re.search(r"\n([A-Z][A-Z .'-]+ \| [A-Z][A-Z .'-]+)\n", "\n" + text + "\n")
        p["location"] = m.group(1).title() if m else None
    p["city"], p["country"] = parse_location(p.get("location") or "")
    feats = (det.get("ProjectDetails") or {}).get("Amenities") or {}
    am = [strapi_text(a.get("title") or a.get("name") or "") for a in feats.get("amenitiesList") or [] if isinstance(a, dict)]
    p["amenities"] = [a for a in am if a][:15]
    return p


def projects_from_index(rec: dict) -> list[dict]:
    """/projects/properties-in-<country> pages: Tabs[].TabData[] with ProjectTitle, Location, ..."""
    out = []
    for d in find_dicts(rec.get("next_page_data") or {}, lambda d: "ProjectTitle" in d):
        link = d.get("Link") or ""
        if isinstance(link, dict):
            link = link.get("href") or link.get("url") or ""
        url = link if link.startswith("http") else ("https://darglobal.co.uk" + link if link.startswith("/") else None)
        out.append({
            "name": strapi_text(d.get("ProjectTitle")),
            "url": url.rstrip("/") if url else None,
            "location": strapi_text(d.get("Location")) or None,
            "bedrooms": strapi_text(d.get("Bedrooms")) or None,
            "type": strapi_text(d.get("ProjectType")) or None,
        })
    return out


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def merge_projects(page_projects: list[dict], index_projects: list[dict]) -> list[dict]:
    by_url = {p["url"].lower(): p for p in page_projects}
    by_name = {norm_name(p["name"]): p for p in page_projects}
    for ip in index_projects:
        target = (by_url.get((ip.get("url") or "").lower()) or by_name.get(norm_name(ip["name"])))
        if target:
            for k, v in ip.items():
                if v and not target.get(k):
                    target[k] = v
        elif ip["name"]:
            ip["city"], ip["country"] = parse_location(ip.get("location") or "")
            by_name[norm_name(ip["name"])] = ip
            page_projects.append(ip)
    # stable de-dupe by name
    seen, out = set(), []
    for p in page_projects:
        k = norm_name(p["name"])
        if k and k not in seen:
            seen.add(k)
            out.append(p)
    return out


# ----------------------------------------------------------------------------- Wasalt

def fmt_num(x: float | int | None) -> str:
    if x is None:
        return ""
    return f"{x:,.0f}" if float(x).is_integer() or x > 100 else f"{x:,.1f}"


def listing_fact_sheet(r: dict) -> str:
    where = ", ".join(x for x in [r.get("district"), r.get("territory") or r.get("city")] if x)
    if r.get("city") and r.get("city") not in where:
        where += f", {r['city']}"
    purpose = "for rent" if r.get("purpose") == "rent" else "for sale"
    head = f"{r.get('sub_type') or 'Property'} {purpose} in {where}."
    bits = []
    if r.get("bedrooms") is not None:
        bits.append(f"{r['bedrooms']} bedrooms")
    if r.get("bathrooms") is not None:
        bits.append(f"{r['bathrooms']} bathrooms")
    if r.get("area_sqm"):
        bits.append(f"{fmt_num(r['area_sqm'])} sqm")
    price = ""
    if r.get("price"):
        price = f"{'Rent' if r.get('purpose') == 'rent' else 'Price'} {fmt_num(r['price'])} {r.get('currency') or 'SAR'}"
        price += "/year." if r.get("purpose") == "rent" else "."
    extras = []
    for k, label in (("furnishing", "Furnishing"), ("property_age", "Age"), ("facing", "Facing"),
                     ("street_width", "Street width"), ("main_type", "Usage")):
        if r.get(k):
            extras.append(f"{label}: {r[k]}")
    desc = redact((r.get("description_text") or "")[:500])
    parts = [head, (", ".join(bits) + ".") if bits else "", price, "; ".join(extras), desc,
             "Source: Wasalt.sa"]
    return clean_whitespace("\n".join(p for p in parts if p))


LISTING_COLS = [
    "id", "url", "title", "purpose", "main_type", "sub_type", "city", "district", "zone", "territory",
    "address", "price", "currency", "rent_frequency", "rent_monthly", "price_per_sqm", "bedrooms",
    "bathrooms", "living_rooms", "guest_rooms", "area_sqm", "built_up_area_sqm", "land_area_sqm",
    "furnishing", "property_age", "facing", "street_width_m", "lat", "lon", "valuation_low",
    "valuation_high", "is_verified", "published_at", "updated_at", "description_text", "fetched_at", "price_flag",
]


def clean_listing(r: dict) -> dict:
    out = {k: r.get(k) for k in LISTING_COLS}  # allow-list again (defence in depth)
    for k in ("title", "description_text", "address"):
        if out.get(k):
            out[k] = redact(clean_whitespace(str(out[k])))
    if out.get("description_text"):
        out["description_text"] = out["description_text"][:1500]
    if out.get("city"):
        out["city"] = CITY_FIX.get(out["city"], out["city"])
    # ~8% of rentals carry a "yearly" rent below 12k SAR (< 1k/month), which is almost certainly a monthly
    # figure typed into the yearly field. Keep the listed value, but flag it so stats and sorting skip it.
    out["price_flag"] = None
    if (out.get("purpose") == "rent" and out.get("sub_type") not in ("Room",) and out.get("price")
            and out["price"] < MIN_PLAUSIBLE_YEARLY_RENT):
        out["price_flag"] = "possibly_monthly"
    return out


MIN_PLAUSIBLE_YEARLY_RENT = 12_000


CITY_FIX = {"Aldammam": "Dammam", "Makkah Al Mukarramah": "Makkah", "Al Madinah Al Munawwarah": "Madinah",
            "Alkhobar": "Khobar", "Al Khobar": "Khobar", "Albasar": "Al Basar",
            # Wasalt's machine transliterations -> common English spellings users will type
            "Bariduh": "Buraydah", "Alttayif": "Taif", "Eanizah": "Unaizah", "Tbwk": "Tabuk", "Earear": "Arar",
            "Alzahran": "Dhahran", "Harimla'": "Huraymila", "Al-Majma'Ah": "Al Majmaah", "Almuzahimih": "Al Muzahimiyah",
            "Al Mazahmyah": "Al Muzahimiyah", "Hayil": "Hail", "Al-Zulfi": "Al Zulfi", "Rafha'": "Rafha",
            "Daba'": "Duba", "Al-Ula": "AlUla"}


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-darglobal", type=int, default=300)
    ap.add_argument("--min-listings", type=int, default=3000)
    args = ap.parse_args(argv)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dg_raw = read_jsonl(RAW_DIR / "darglobal.jsonl")
    wl_raw = read_jsonl(RAW_DIR / "wasalt_listings.jsonl")
    wp_raw = read_jsonl(RAW_DIR / "wasalt_pages.jsonl")

    # de-dupe DarGlobal by url, then by text hash
    # (same page is often listed under two URL casings, e.g. /Trump-International-Hotel-Tower-Dubai and
    #  /trump-international-hotel-and-tower-dubai -> also de-dupe on title + text prefix)
    seen_url, seen_hash, dg = set(), set(), []
    for r in dg_raw:
        h = hashlib.sha1(r["text"].encode(), usedforsecurity=False).hexdigest()
        near = hashlib.sha1(((r.get("title") or "") + r["text"][:400]).lower().encode(), usedforsecurity=False).hexdigest()
        if r["url"].lower() in seen_url or h in seen_hash or near in seen_hash or "Incapsula" in r["text"]:
            continue
        seen_url.add(r["url"].lower()); seen_hash.update((h, near)); dg.append(r)
    cleaned = remove_boilerplate(dg)

    docs: list[dict] = []
    page_projects, index_projects = [], []
    for r in dg:
        text = cleaned[r["url"]]
        faq = faq_text(r.get("next_page_data"))
        if faq and faq[:80] not in text:
            text += "\n\nFrequently Asked Questions\n" + faq
        if r.get("description") and r["description"] not in text:
            text = r["description"] + "\n" + text
        text = redact(text)
        if len(text) < 80:
            continue
        title = re.sub(r"\s*[|–-]\s*DarGlobal.*$", "", r["title"] or "").strip() or r["url"].rsplit("/", 1)[-1]
        published = None
        pdata = r.get("next_page_data") or {}
        for d in find_dicts(pdata, lambda d: "publishedAt" in d):
            published = d.get("date") or d.get("Date") or d.get("publishedAt")
            break
        docs.append({"doc_id": doc_id("dg", r["url"]), "site": "darglobal", "url": r["url"], "title": title,
                     "section": r["section"], "text": text, "published_at": published})
        p = project_from_page(r, text)
        if p:
            page_projects.append(p)
        if r["section"] in ("projects", "projects-index", "home", "one-of-one", "one-of-one-index"):
            index_projects.extend(projects_from_index(r))
    projects = merge_projects(page_projects, index_projects)
    for p in projects:
        for k, v in list(p.items()):
            if isinstance(v, str):
                p[k] = redact(v)

    for r in wp_raw:
        text = redact(clean_whitespace(r["text"]))
        if len(text) < 80:
            continue
        title = re.sub(r"\s*\|\s*Wasalt.*$", "", r["title"] or "").strip()
        docs.append({"doc_id": doc_id("wp", r["url"]), "site": "wasalt", "url": r["url"], "title": title,
                     "section": "about-wasalt", "text": text, "published_at": None})

    listings, seen_ids = [], set()
    for r in wl_raw:
        if r.get("id") in seen_ids or not r.get("price"):
            continue
        seen_ids.add(r.get("id"))
        cl = clean_listing(r)
        listings.append(cl)
        docs.append({"doc_id": doc_id("wl", str(cl["id"])), "site": "wasalt", "url": cl["url"],
                     "title": cl["title"], "section": f"listing-{cl['purpose']}",
                     "text": listing_fact_sheet(cl), "published_at": cl.get("published_at")})

    def dump(name: str, rows: list[dict]) -> None:
        with (PROCESSED_DIR / name).open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump("documents.jsonl", docs)
    dump("listings.jsonl", listings)
    dump("projects.jsonl", projects)

    dg_fail = read_jsonl(RAW_DIR / "darglobal_failures.jsonl")
    wl_fail = read_jsonl(RAW_DIR / "wasalt_failures.jsonl")
    dg_ok = {d["url"] for d in docs if d["site"] == "darglobal"}
    report = {
        "generated_at": now_iso(),
        "scrape_date": max((r.get("fetched_at") or "" for r in dg_raw + wl_raw), default="")[:10],
        "darglobal": {
            "pages": len(dg_ok),
            "by_section": dict(Counter(d["section"] for d in docs if d["site"] == "darglobal").most_common()),
            "projects": len(projects),
            "failed_urls": len({f["url"] for f in dg_fail} - dg_ok),
            "method": "Playwright + installed Google Chrome (site behind Imperva/Incapsula)",
        },
        "wasalt": {
            "listings": len(listings),
            "static_pages": len(wp_raw),
            "by_purpose_type": dict(Counter(f"{r['purpose']}/{r['sub_type']}" for r in listings).most_common()),
            "by_city": dict(Counter(r["city"] for r in listings).most_common(20)),
            "unavailable_or_failed": len({f["url"] for f in wl_fail}),
            "sitemap_listing_urls": 60000,
            "sampling": "stratified by purpose x type, newest-first, seed 42",
            "method": "Next.js /_next/data JSON routes, <=1 req/s, field allow-list",
        },
        "documents": len(docs),
        "compliance": [
            "robots.txt honoured on both sites (Wasalt /search never requested)",
            "Wasalt llms.txt honoured: public listing content only, moderate rate, attribution 'Source: Wasalt.sa', non-commercial",
            "No owner/agent names, phone numbers, emails or license numbers stored (allow-list + regex redaction)",
        ],
    }
    (PROCESSED_DIR / "scrape_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- sanity checks (fail loudly)
    blob = "\n".join((PROCESSED_DIR / n).read_text(encoding="utf-8").lower()
                     for n in ("documents.jsonl", "listings.jsonl", "projects.jsonl"))
    bad = [k for k in FORBIDDEN_KEYS if k in blob]
    problems = []
    if bad:
        problems.append(f"forbidden PII keys present: {bad}")
    if len(dg_ok) < args.min_darglobal:
        problems.append(f"only {len(dg_ok)} DarGlobal docs (< {args.min_darglobal})")
    if len(listings) < args.min_listings:
        problems.append(f"only {len(listings)} listings (< {args.min_listings})")
    log.info("documents=%d listings=%d projects=%d darglobal_pages=%d", len(docs), len(listings), len(projects), len(dg_ok))
    if problems:
        for p in problems:
            log.error("SANITY CHECK FAILED: %s", p)
        return 1
    log.info("sanity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
