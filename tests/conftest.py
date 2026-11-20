"""Shared fixtures: a tiny in-memory dataset + index built with a deterministic fake embedder,
so tests need no model downloads and no network."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.build_index import build_chunks, build_sqlite, embed_text  # noqa: E402
from app.textutil import BGE_QUERY_PREFIX, tokenize  # noqa: E402

DIM = 64


class FakeEmbedder:
    """Hashed bag-of-words -> unit vector. Similar texts share tokens -> high cosine."""

    def embed(self, texts, batch_size=32):
        for t in texts:
            t = t.replace(BGE_QUERY_PREFIX, "")
            v = np.zeros(DIM, dtype=np.float32)
            for tok in tokenize(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
            n = np.linalg.norm(v)
            yield v / n if n else v


class FakeReranker:
    def rerank(self, query, docs):
        q = set(tokenize(query))
        for d in docs:
            yield float(len(q & set(tokenize(d))))


DOCS = [
    {"doc_id": "dg1", "site": "darglobal", "url": "https://darglobal.co.uk/w-residences", "title": "W Residences Downtown Dubai",
     "section": "project", "text": "W Residences Downtown Dubai by DarGlobal. Branded residences with 1 to 3 bedroom apartments. "
                                   "STATUS Under Development. AREA (SQFT) 700 - 3,000."},
    {"doc_id": "dg2", "site": "darglobal", "url": "https://darglobal.co.uk/trump-tower-jeddah", "title": "Trump Tower Jeddah",
     "section": "project", "text": "Trump Tower Jeddah is a DarGlobal project in Jeddah, Saudi Arabia, on the Corniche."},
    {"doc_id": "dg3", "site": "darglobal", "url": "https://darglobal.co.uk/about", "title": "About DarGlobal",
     "section": "company", "text": "DarGlobal is a London-listed luxury real estate developer building in Dubai, Oman, Saudi Arabia, Qatar, Spain and the UK."},
    {"doc_id": "wp1", "site": "wasalt", "url": "https://wasalt.sa/en/s/terms-of-use", "title": "Terms of use",
     "section": "about-wasalt", "text": "Wasalt is a Saudi real estate platform for buying and renting property."},
]
LISTINGS = [
    {"id": i, "url": f"https://wasalt.sa/en/property/{p}/{t.lower()}-{i}", "title": f"{t} {i}", "purpose": p,
     "main_type": "Residential", "sub_type": t, "city": c, "district": d, "zone": d, "territory": None, "address": None,
     "price": price, "currency": "SAR", "rent_frequency": "yearly" if p == "rent" else None, "rent_monthly": None,
     "price_per_sqm": round(price / area, 1), "bedrooms": beds, "bathrooms": 2, "living_rooms": None, "guest_rooms": None,
     "area_sqm": area, "built_up_area_sqm": area, "land_area_sqm": None, "furnishing": None, "property_age": "New",
     "facing": None, "street_width_m": None, "lat": 24.7, "lon": 46.6, "valuation_low": None, "valuation_high": None,
     "is_verified": False, "published_at": f"2026-11-{10 + i % 9:02d}", "updated_at": None,
     "description_text": "Nice property", "fetched_at": "2026-11-20"}
    for i, (p, t, c, d, price, beds, area) in enumerate([
        ("rent", "Apartment", "Riyadh", "Al Malqa", 75000, 3, 160),
        ("rent", "Apartment", "Riyadh", "Al Yasmin", 60000, 3, 140),
        ("rent", "Apartment", "Riyadh", "Al Narjis", 95000, 3, 180),
        ("rent", "Apartment", "Riyadh", "Al Malqa", 50000, 2, 110),
        ("rent", "Villa", "Riyadh", "Al Malqa", 140000, 5, 400),
        ("sale", "Villa", "Jeddah", "Obhur", 2_000_000, 5, 450),
        ("sale", "Villa", "Jeddah", "Al Shati", 3_000_000, 6, 600),
        ("sale", "Land", "Dammam", "Al Faisaliyah", 900_000, None, 600),
    ], start=1)
]
PROJECTS = [
    {"name": "W Residences Downtown Dubai", "url": "https://darglobal.co.uk/w-residences", "location": "Dubai, UAE",
     "city": "Dubai", "country": "UAE", "type": "Branded residences", "status": "Under Development",
     "unit_types": "1-3 bedroom apartments", "summary": "Branded W residences", "amenities": ["Pool"]},
    {"name": "Trump Tower Jeddah", "url": "https://darglobal.co.uk/trump-tower-jeddah", "location": "Jeddah, KSA",
     "city": "Jeddah", "country": "Saudi Arabia", "type": "Branded tower", "status": "Off-plan", "summary": "Trump tower"},
]


@pytest.fixture(scope="session")
def tiny_index(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("index")
    docs = DOCS + [{"doc_id": f"wl{r['id']}", "site": "wasalt", "url": r["url"], "title": r["title"],
                    "section": f"listing-{r['purpose']}", "text": f"{r['sub_type']} for {r['purpose']} in {r['district']}, {r['city']}"}
                   for r in LISTINGS]
    chunks = build_chunks(docs)
    with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    vecs = np.stack(list(FakeEmbedder().embed([embed_text(c["text"]) for c in chunks]))).astype(np.float32)
    np.save(d / "vectors.npy", vecs)
    with (d / "bm25.pkl").open("wb") as f:
        pickle.dump(BM25Okapi([tokenize(c["text"]) for c in chunks]), f)
    build_sqlite(d / "listings.sqlite", LISTINGS, PROJECTS, {"scrape_date": "2026-11-20"})
    return d


@pytest.fixture()
def retriever(tiny_index):
    from app.retrieval import HybridRetriever

    return HybridRetriever(tiny_index, "fake", "fake", embedder=FakeEmbedder(), reranker=FakeReranker())


@pytest.fixture()
def store(tiny_index):
    from app.retrieval import ListingStore

    return ListingStore(tiny_index / "listings.sqlite")
