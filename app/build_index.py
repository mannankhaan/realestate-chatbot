"""processed data -> index/ (vectors.npy, chunks.jsonl, bm25.pkl, listings.sqlite).

Runs during `docker build` so the container starts instantly. Must stay well under 10 min on CPU.
    python -m app.build_index
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import re
import shutil
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .config import get_settings
from .textutil import chunk_text, tokenize

log = logging.getLogger("build_index")

LISTING_SCHEMA = """
CREATE TABLE listings (
  id INTEGER PRIMARY KEY, url TEXT NOT NULL, title TEXT, purpose TEXT, main_type TEXT, sub_type TEXT,
  city TEXT, district TEXT, zone TEXT, territory TEXT, address TEXT,
  price REAL, currency TEXT, rent_frequency TEXT, rent_monthly REAL, price_per_sqm REAL,
  bedrooms INTEGER, bathrooms INTEGER, living_rooms INTEGER, guest_rooms INTEGER,
  area_sqm REAL, built_up_area_sqm REAL, land_area_sqm REAL,
  furnishing TEXT, property_age TEXT, facing TEXT, street_width_m REAL, lat REAL, lon REAL,
  valuation_low REAL, valuation_high REAL, is_verified INTEGER, published_at TEXT, updated_at TEXT,
  description_text TEXT, fetched_at TEXT, price_flag TEXT
);
CREATE INDEX ix_city ON listings(city COLLATE NOCASE);
CREATE INDEX ix_purpose_type ON listings(purpose, sub_type COLLATE NOCASE);
CREATE INDEX ix_price ON listings(price);
CREATE INDEX ix_bedrooms ON listings(bedrooms);
CREATE TABLE projects (
  name TEXT, url TEXT, location TEXT, city TEXT, country TEXT, type TEXT, unit_types TEXT, status TEXT,
  bedrooms TEXT, area_sqft TEXT, area_sqm TEXT, completion TEXT, starting_price TEXT, featuring TEXT,
  collaboration TEXT, designed_by TEXT, summary TEXT, amenities TEXT
);
CREATE TABLE city_stats (
  city TEXT, purpose TEXT, sub_type TEXT, n INTEGER, min_price REAL, median_price REAL, max_price REAL,
  avg_price_per_sqm REAL
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""
PROJECT_COLS = ["name", "url", "location", "city", "country", "type", "unit_types", "status", "bedrooms",
                "area_sqft", "area_sqm", "completion", "starting_price", "featuring", "collaboration",
                "designed_by", "summary", "amenities"]


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_chunks(docs: list[dict]) -> list[dict]:
    chunks = []
    for d in docs:
        is_listing = d["section"].startswith("listing")
        parts = [d["text"]] if is_listing else chunk_text(d["text"])
        site_label = "DarGlobal" if d["site"] == "darglobal" else "Wasalt"
        for i, part in enumerate(parts):
            header = f"{d['title']} — {site_label} {d['section']}"
            chunks.append({
                "id": len(chunks), "doc_id": d["doc_id"], "chunk_no": i, "site": d["site"],
                "section": d["section"], "title": d["title"], "url": d["url"],
                "text": f"{header}\n{part}",
            })
    return chunks


ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]+")


def embed_text(text: str, max_chars: int = 1600) -> str:
    """Text fed to the (English) embedding model. Arabic script is dropped: bge-small's vocabulary
    explodes it into hundreds of useless sub-tokens, which made embedding ~10x slower. BM25 still
    indexes the full text, Arabic included."""
    t = ARABIC_RE.sub(" ", text)
    t = re.sub(r"[^\S\n]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t).strip()
    return t[:max_chars]


def _key(model_name: str, text: str) -> str:
    return hashlib.sha1(f"{model_name}|{text}".encode(), usedforsecurity=False).hexdigest()


def embed(texts: list[str], model_name: str, cache_path: Path | None = None, batch: int = 32) -> np.ndarray:
    """Embed with an optional on-disk cache keyed by sha1(model, text).

    The cache (data/processed/embeddings_cache.npz, float16) is committed, so `docker build` only
    embeds chunks whose text changed; with no cache everything is computed from scratch."""
    keys = [_key(model_name, t) for t in texts]
    cached: dict[str, np.ndarray] = {}
    if cache_path and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        cached = dict(zip(z["keys"].tolist(), z["vecs"].astype(np.float32)))
    missing = [i for i, k in enumerate(keys) if k not in cached]
    log.info("embedding cache: %d hits, %d to compute", len(texts) - len(missing), len(missing))
    if missing:
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name)
        new = model.embed([texts[i] for i in missing], batch_size=batch)
        for n, (i, v) in enumerate(zip(missing, new), 1):
            cached[keys[i]] = np.asarray(v, dtype=np.float32)
            if n % 500 == 0:
                log.info("  embedded %d/%d", n, len(missing))
    vecs = np.stack([cached[k] for k in keys]).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs


def write_cache(cache_path: Path, texts: list[str], vecs: np.ndarray, model_name: str) -> None:
    keys = np.array([_key(model_name, t) for t in texts])
    np.savez_compressed(cache_path, keys=keys, vecs=vecs.astype(np.float16))


def build_sqlite(path: Path, listings: list[dict], projects: list[dict], report: dict) -> None:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(LISTING_SCHEMA)
    cols = [c[1] for c in con.execute("PRAGMA table_info(listings)")]
    con.executemany(
        f"INSERT OR IGNORE INTO listings ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        [[(int(r[c]) if c == "is_verified" else r.get(c)) for c in cols] for r in listings],
    )
    con.executemany(
        f"INSERT INTO projects ({','.join(PROJECT_COLS)}) VALUES ({','.join('?' * len(PROJECT_COLS))})",
        [[(", ".join(p[c]) if isinstance(p.get(c), list) else p.get(c)) for c in PROJECT_COLS] for p in projects],
    )
    # precomputed stats (SQLite has no MEDIAN)
    groups: dict[tuple, list[dict]] = {}
    for r in listings:
        for key in ((r["city"], r["purpose"], r["sub_type"]), (r["city"], r["purpose"], None),
                    (None, r["purpose"], r["sub_type"])):
            groups.setdefault(key, []).append(r)
    rows = []
    for (city, purpose, sub), rs in groups.items():
        prices = [r["price"] for r in rs if r.get("price") and not r.get("price_flag")]
        ppsqm = [r["price_per_sqm"] for r in rs if r.get("price_per_sqm")]
        if prices:
            rows.append((city, purpose, sub, len(prices), min(prices), statistics.median(prices), max(prices),
                         round(statistics.mean(ppsqm), 1) if ppsqm else None))
    con.executemany("INSERT INTO city_stats VALUES (?,?,?,?,?,?,?,?)", rows)
    con.executemany("INSERT INTO meta VALUES (?,?)", [("scrape_date", report.get("scrape_date", "")),
                                                      ("report", json.dumps(report))])
    con.commit()
    con.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-cache", action="store_true",
                    help="save embeddings to data/processed/embeddings_cache.npz (commit it)")
    ap.add_argument("--no-cache", action="store_true", help="ignore the embedding cache")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = get_settings()
    t0 = time.perf_counter()
    docs = read_jsonl(s.data_dir / "documents.jsonl")
    listings = read_jsonl(s.data_dir / "listings.jsonl")
    projects = read_jsonl(s.data_dir / "projects.jsonl")
    report = json.loads((s.data_dir / "scrape_report.json").read_text(encoding="utf-8"))

    tmp = s.index_dir.with_name(s.index_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    chunks = build_chunks(docs)
    log.info("%d docs -> %d chunks (%.1fs)", len(docs), len(chunks), time.perf_counter() - t0)

    t = time.perf_counter()
    cache_path = s.data_dir / "embeddings_cache.npz"
    etexts = [embed_text(c["text"]) for c in chunks]
    vecs = embed(etexts, s.embed_model, None if args.no_cache else cache_path)
    np.save(tmp / "vectors.npy", vecs)
    if args.write_cache:
        write_cache(cache_path, etexts, vecs, s.embed_model)
        log.info("wrote embedding cache %s (%.1f MB)", cache_path, cache_path.stat().st_size / 1e6)
    log.info("embedded %s with %s in %.1fs", vecs.shape, s.embed_model, time.perf_counter() - t)

    with (tmp / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    t = time.perf_counter()
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    with (tmp / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("bm25 built in %.1fs", time.perf_counter() - t)

    build_sqlite(tmp / "listings.sqlite", listings, projects, report)
    log.info("sqlite: %d listings, %d projects", len(listings), len(projects))

    # make sure both ONNX models are downloaded into the image, even when every embedding came from the
    # cache, so the container never downloads anything at runtime
    from fastembed import TextEmbedding

    TextEmbedding(s.embed_model)
    if s.use_reranker:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        TextCrossEncoder(s.rerank_model)
        log.info("re-ranker %s cached", s.rerank_model)

    try:
        if s.index_dir.exists():
            shutil.rmtree(s.index_dir)
        tmp.rename(s.index_dir)
    except OSError as e:
        raise SystemExit(f"Could not replace {s.index_dir} ({e}). Is the app still running and holding the old "
                         f"index open? Stop it and move {tmp} to {s.index_dir}.") from e
    log.info("index built in %.1fs -> %s", time.perf_counter() - t0, s.index_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
