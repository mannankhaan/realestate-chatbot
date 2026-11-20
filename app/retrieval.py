"""Retrieval layer.

* HybridRetriever: BM25 + dense (bge-small) fused with Reciprocal Rank Fusion, then a cross-encoder
  re-ranker. Used for unstructured questions (DarGlobal projects, news, Wasalt pages).
* ListingStore: exact, parameterised SQL over Wasalt listings (price / beds / city filters that a
  vector index cannot do), with progressive filter relaxation, plus aggregate stats.
"""
from __future__ import annotations

import json
import logging
import pickle
import re
import sqlite3
import statistics
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .textutil import BGE_QUERY_PREFIX, tokenize

log = logging.getLogger("retrieval")

RRF_K = 60
TITLE_BOOST_SECTIONS = {"project", "company", "projects", "one-of-one", "about-wasalt"}


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class Hit:
    chunk_id: int
    score: float
    text: str
    url: str
    title: str
    site: str
    section: str
    doc_id: str

    def as_source(self) -> dict[str, Any]:
        return {"url": self.url, "title": self.title, "site": self.site, "section": self.section,
                "snippet": self.text[:240]}


class HybridRetriever:
    def __init__(self, index_dir: Path, embed_model: str, rerank_model: str | None = None,
                 embedder: Any | None = None, reranker: Any | None = None):
        with (index_dir / "chunks.jsonl").open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]
        self.vectors = np.load(index_dir / "vectors.npy")
        with (index_dir / "bm25.pkl").open("rb") as f:
            self.bm25 = pickle.load(f)  # noqa: S301 - our own file, built at image build time
        self.site = np.array([c["site"] for c in self.chunks])
        self._embed_model_name = embed_model
        self._rerank_model_name = rerank_model
        self._embedder = embedder
        self._reranker = reranker
        self._lock = threading.Lock()

    # lazy model loading keeps unit tests fast and lets tests inject fakes
    @property
    def embedder(self):
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(self._embed_model_name)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None and self._rerank_model_name:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(self._rerank_model_name)
        return self._reranker

    def embed_query(self, query: str) -> np.ndarray:
        with self._lock:
            v = np.asarray(next(iter(self.embedder.embed([BGE_QUERY_PREFIX + query]))), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def search(self, query: str, k: int = 6, site: str | None = None, pool: int = 30,
               max_per_doc: int = 2, rerank: bool = True, exclude_listings: bool = False) -> list[Hit]:
        if not query.strip():
            return []
        mask = np.ones(len(self.chunks), dtype=bool)
        if site in ("darglobal", "wasalt"):
            mask &= self.site == site
        if exclude_listings:
            mask &= np.array([not c["section"].startswith("listing") for c in self.chunks])
        if not mask.any():
            return []

        bm = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        bm[~mask] = -np.inf
        bm_rank = [int(i) for i in np.argsort(-bm)[:pool] if np.isfinite(bm[i]) and bm[i] > 0]

        dense = self.vectors @ self.embed_query(query)
        dense[~mask] = -np.inf
        dense_rank = [int(i) for i in np.argsort(-dense)[:pool] if np.isfinite(dense[i])]  # masked = -inf

        fused = sorted(rrf([bm_rank, dense_rank]).items(), key=lambda x: -x[1])
        hits, per_doc = [], {}
        for idx, score in fused:
            c = self.chunks[idx]
            if per_doc.get(c["doc_id"], 0) >= max_per_doc:
                continue
            per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
            hits.append(Hit(idx, score, c["text"], c["url"], c["title"], c["site"], c["section"], c["doc_id"]))
            if len(hits) >= max(k * 3, 12):
                break

        if rerank and self.reranker is not None and len(hits) > 1:
            try:
                with self._lock:
                    scores = list(self.reranker.rerank(query, [h.text[:1200] for h in hits]))
                for h, s in zip(hits, scores):
                    h.score = float(s)
                hits.sort(key=lambda h: -h.score)
            except Exception as e:  # noqa: BLE001 - re-ranking is an optimisation, never fatal
                log.warning("re-ranker failed, using RRF order: %s", e)
        # Metadata boost: when the query names an entity whose *title* matches a project/company page,
        # the canonical page should beat press releases that merely mention it.
        q_tokens = set(tokenize(query))
        if q_tokens:
            margin = max((abs(h.score) for h in hits), default=1.0) or 1.0
            for h in hits:
                overlap = len(q_tokens & set(tokenize(h.title))) / len(q_tokens)
                if overlap >= 0.6 and h.section in TITLE_BOOST_SECTIONS:
                    h.score += margin * overlap
            hits.sort(key=lambda h: -h.score)
        return hits[:k]


# ----------------------------------------------------------------------------- listings

SORTS = {
    "price_asc": "(price_flag IS NOT NULL), price ASC",  # suspect prices never win "cheapest"
    "price_desc": "price DESC",
    "newest": "published_at DESC",
    "area_desc": "area_sqm DESC",
}
LISTING_FIELDS = ("id, url, title, purpose, sub_type, city, district, territory, price, currency, "
                  "rent_frequency, bedrooms, bathrooms, area_sqm, furnishing, property_age, price_per_sqm, "
                  "published_at, price_flag")


@dataclass
class Filters:
    purpose: str | None = None
    sub_type: str | None = None
    city: str | None = None
    district: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    min_bedrooms: int | None = None
    max_bedrooms: int | None = None
    min_area: float | None = None
    max_area: float | None = None
    furnishing: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Filters:
        d = d or {}
        out = cls()
        for f in out.__dataclass_fields__:
            v = d.get(f)
            if v in (None, "", "null", "any"):
                continue
            if f in ("min_price", "max_price", "min_area", "max_area"):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
            elif f in ("min_bedrooms", "max_bedrooms"):
                try:
                    v = int(float(v))
                except (TypeError, ValueError):
                    continue
            else:
                v = str(v).strip()[:80]
            setattr(out, f, v)
        if out.purpose:
            p = out.purpose.lower()
            out.purpose = "rent" if p.startswith("rent") or p == "lease" else "sale" if p in ("sale", "buy", "sell") else None
        return out

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ListingResult:
    rows: list[dict[str, Any]]
    total: int
    filters: dict[str, Any]
    relaxed: list[str] = field(default_factory=list)


def canon_furnishing(v: str) -> str:
    s = v.lower().replace(" ", "").replace("-", "")
    if s.startswith("semi"):
        return "Semi-Furnished"
    if s.startswith("un") or s.startswith("not"):
        return "Un-Furnished"
    return "Furnished"


class ListingStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        con = self._con()
        self.cities = [r[0] for r in con.execute(
            "SELECT city FROM listings WHERE city IS NOT NULL GROUP BY city ORDER BY COUNT(*) DESC")]
        self.sub_types = [r[0] for r in con.execute(
            "SELECT sub_type FROM listings WHERE sub_type IS NOT NULL GROUP BY sub_type ORDER BY COUNT(*) DESC")]
        self.districts = [r[0] for r in con.execute(
            "SELECT district FROM listings WHERE district IS NOT NULL GROUP BY district")]
        self.count = con.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        self.project_count = con.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        row = con.execute("SELECT value FROM meta WHERE key='scrape_date'").fetchone()
        self.scrape_date = row[0] if row else ""

    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
            con.row_factory = sqlite3.Row
            self._local.con = con
        return con

    # -- canonicalisation: map free text to values that exist in the DB
    def canon_city(self, city: str | None) -> str | None:
        if not city:
            return None
        c = re.sub(r"[^a-z]", "", city.lower())
        aliases = {"mecca": "makkah", "makkah": "makkah", "medina": "madinah", "madinah": "madinah",
                   "alkhobar": "khobar", "aldammam": "dammam", "jedda": "jeddah", "riyad": "riyadh"}
        c = aliases.get(c, c)
        for known in self.cities:
            k = re.sub(r"[^a-z]", "", known.lower())
            if k == c or k.endswith(c) or c.endswith(k):
                return known
        return city

    def canon_sub_type(self, st: str | None) -> str | None:
        if not st:
            return None
        s = st.lower().strip().rstrip("s")
        aliases = {"flat": "apartment", "plot": "land", "house": "villa", "resthouse": "rest house",
                   "istiraha": "rest house", "office": "office", "shop": "shop"}
        s = aliases.get(s.replace(" ", ""), s)
        for known in self.sub_types:
            if known.lower() == s or known.lower().rstrip("s") == s:
                return known
        for known in self.sub_types:
            if s in known.lower():
                return known
        return st

    def _where(self, f: Filters) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if f.purpose:
            clauses.append("purpose = ?"); params.append(f.purpose)
        if f.sub_type:
            clauses.append("sub_type = ? COLLATE NOCASE"); params.append(self.canon_sub_type(f.sub_type))
        if f.city:
            clauses.append("city = ? COLLATE NOCASE"); params.append(self.canon_city(f.city))
        if f.district:
            clauses.append("(district LIKE ? OR zone LIKE ? OR territory LIKE ?)")
            params += [f"%{f.district}%"] * 3
        if f.min_price is not None:
            clauses.append("price >= ?"); params.append(f.min_price)
        if f.max_price is not None:
            clauses.append("price <= ?"); params.append(f.max_price)
        if f.min_bedrooms is not None:
            clauses.append("bedrooms >= ?"); params.append(f.min_bedrooms)
        if f.max_bedrooms is not None:
            clauses.append("bedrooms <= ?"); params.append(f.max_bedrooms)
        if f.min_area is not None:
            clauses.append("area_sqm >= ?"); params.append(f.min_area)
        if f.max_area is not None:
            clauses.append("area_sqm <= ?"); params.append(f.max_area)
        if f.furnishing:
            # exact (case-insensitive): LIKE %Furnished% would also match Semi-/Un-Furnished
            clauses.append("furnishing = ? COLLATE NOCASE"); params.append(canon_furnishing(f.furnishing))
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def _query(self, f: Filters, sort: str | None, limit: int) -> tuple[list[dict], int]:
        where, params = self._where(f)
        con = self._con()
        total = con.execute(f"SELECT COUNT(*) FROM listings{where}", params).fetchone()[0]
        order = SORTS.get(sort or "", "published_at DESC")
        rows = con.execute(f"SELECT {LISTING_FIELDS} FROM listings{where} ORDER BY {order} LIMIT ?",
                           [*params, int(limit)]).fetchall()
        return [dict(r) for r in rows], total

    def search(self, filters: Filters | dict, sort: str | None = None, limit: int = 8) -> ListingResult:
        f = filters if isinstance(filters, Filters) else Filters.from_dict(filters)
        limit = max(1, min(int(limit or 8), 20))
        rows, total = self._query(f, sort, limit)
        relaxed: list[str] = []
        # Progressive relaxation when nothing matches: tell the model exactly what was loosened.
        steps = [
            ("district", lambda g: setattr(g, "district", None)),
            ("price ±20%", lambda g: (setattr(g, "max_price", g.max_price * 1.2 if g.max_price else None),
                                      setattr(g, "min_price", g.min_price * 0.8 if g.min_price else None))),
            ("bedrooms ±1", lambda g: (setattr(g, "min_bedrooms", max(0, g.min_bedrooms - 1) if g.min_bedrooms else None),
                                       setattr(g, "max_bedrooms", g.max_bedrooms + 1 if g.max_bedrooms else None))),
            ("area", lambda g: (setattr(g, "min_area", None), setattr(g, "max_area", None))),
            ("furnishing", lambda g: setattr(g, "furnishing", None)),
        ]
        g = Filters(**f.__dict__)
        for name, apply in steps:
            if total:
                break
            before = g.as_dict()
            apply(g)
            if g.as_dict() == before:
                continue
            relaxed.append(name)
            rows, total = self._query(g, sort, limit)
        return ListingResult(rows=rows, total=total, filters=g.as_dict(), relaxed=relaxed if total else relaxed)

    def stats(self, filters: Filters | dict, group_by: str | None = None) -> dict[str, Any]:
        f = filters if isinstance(filters, Filters) else Filters.from_dict(filters)
        where, params = self._where(f)
        group_col = {"city": "city", "sub_type": "sub_type", "district": "district", "purpose": "purpose"}.get(group_by or "")
        con = self._con()
        where = (where + " AND " if where else " WHERE ") + "price_flag IS NULL"  # exclude suspect prices
        sel = f"SELECT {group_col + ',' if group_col else ''} price, price_per_sqm, area_sqm, bedrooms FROM listings{where}"
        groups: dict[str, list[sqlite3.Row]] = {}
        for r in con.execute(sel, params):
            groups.setdefault(str(r[group_col]) if group_col else "all", []).append(r)

        def summarise(rs: list[sqlite3.Row]) -> dict[str, Any]:
            prices = [r["price"] for r in rs if r["price"]]
            pps = [r["price_per_sqm"] for r in rs if r["price_per_sqm"]]
            areas = [r["area_sqm"] for r in rs if r["area_sqm"]]
            if not prices:
                return {"count": 0}
            return {
                "count": len(prices),
                "min_price": round(min(prices)),
                "median_price": round(statistics.median(prices)),
                "mean_price": round(statistics.mean(prices)),
                "max_price": round(max(prices)),
                "median_price_per_sqm": round(statistics.median(pps)) if pps else None,
                "median_area_sqm": round(statistics.median(areas)) if areas else None,
            }

        out = {k: summarise(v) for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:15]}
        return {"filters": f.as_dict(), "group_by": group_col, "groups": out or {"all": {"count": 0}},
                "currency": "SAR", "note": "rent prices are SAR per year" if f.purpose == "rent" else None}

    # -- DarGlobal projects
    def projects(self, country: str | None = None, query: str | None = None) -> list[dict[str, Any]]:
        con = self._con()
        sql, params = "SELECT * FROM projects", []
        conds = []
        if country:
            conds.append("(country LIKE ? OR location LIKE ? OR city LIKE ?)")
            params += [f"%{country}%"] * 3
        if query:
            conds.append("(name LIKE ? OR summary LIKE ? OR type LIKE ? OR collaboration LIKE ?)")
            params += [f"%{query}%"] * 4
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        return [dict(r) for r in con.execute(sql + " ORDER BY name", params)]

    def find_project(self, name: str) -> dict[str, Any] | None:
        if not name:
            return None
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        best, best_score = None, 0.0
        for p in self.projects():
            pk = re.sub(r"[^a-z0-9]", "", (p["name"] or "").lower())
            if not pk:
                continue
            if pk == key:
                return p
            words = set(re.findall(r"[a-z0-9]+", name.lower())) - {"the", "by", "residences", "dar", "global", "darglobal"}
            pwords = set(re.findall(r"[a-z0-9]+", (p["name"] or "").lower()))
            score = len(words & pwords) / max(1, len(words)) + (0.5 if key in pk or pk in key else 0)
            if score > best_score:
                best, best_score = p, score
        return best if best_score >= 0.5 else None
