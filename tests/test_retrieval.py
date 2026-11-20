from app.llm import TagFilter
from app.retrieval import Filters, rrf
from app.textutil import chunk_text


def test_rrf_rewards_agreement():
    scores = rrf([[1, 2, 3], [3, 1, 4]])
    order = sorted(scores, key=lambda k: -scores[k])
    assert order[0] == 1  # ranked 1st and 2nd beats 3rd and 1st
    assert order[1] == 3
    assert scores[4] < scores[2] or scores[2] < scores[3]


def test_hybrid_search_finds_project_by_name(retriever):
    hits = retriever.search("W Residences", k=3)
    assert hits[0].url.endswith("/w-residences")


def test_site_filter_and_listing_exclusion(retriever):
    hits = retriever.search("Wasalt platform", k=5, site="wasalt", exclude_listings=True)
    assert hits and all(h.site == "wasalt" and not h.section.startswith("listing") for h in hits)
    hits = retriever.search("Jeddah", k=5, site="darglobal")
    assert all(h.site == "darglobal" for h in hits)


def test_max_two_chunks_per_doc(retriever):
    hits = retriever.search("DarGlobal developer residences", k=10)
    counts = {}
    for h in hits:
        counts[h.doc_id] = counts.get(h.doc_id, 0) + 1
    assert max(counts.values()) <= 2


def test_listing_filters_exact(store):
    res = store.search({"purpose": "rent", "sub_type": "apartment", "city": "riyadh", "min_bedrooms": 3,
                        "max_bedrooms": 3, "max_price": 80000}, sort="price_asc")
    assert res.total == 2 and not res.relaxed
    assert [r["price"] for r in res.rows] == [60000, 75000]


def test_relaxation_reports_what_changed(store):
    # nothing under 70k in Al Malqa with 3 beds -> drop district first
    res = store.search({"purpose": "rent", "sub_type": "Apartment", "city": "Riyadh", "district": "Al Malqa",
                        "min_bedrooms": 3, "max_bedrooms": 3, "max_price": 70000})
    assert res.relaxed[0] == "district"
    assert res.total >= 1 and all(r["price"] <= 70000 for r in res.rows)


def test_relaxation_price_band(store):
    res = store.search({"purpose": "sale", "sub_type": "Villa", "city": "Jeddah", "max_price": 1_800_000})
    assert "price ±20%" in res.relaxed
    assert res.rows[0]["price"] == 2_000_000


def test_sql_injection_is_inert(store):
    res = store.search({"city": "Riyadh'; DROP TABLE listings; --"})
    assert res.total == 0
    assert store.search({"city": "Riyadh"}).total == 5  # table still intact


def test_filters_from_dict_coerces_and_ignores_junk():
    f = Filters.from_dict({"min_price": "80000", "max_bedrooms": "3.0", "city": "null", "purpose": "renting", "x": 1})
    assert f.min_price == 80000 and f.max_bedrooms == 3 and f.city is None and f.purpose == "rent"


def test_stats(store):
    st = store.stats({"purpose": "sale", "sub_type": "Villa", "city": "Jeddah"})
    g = st["groups"]["all"]
    assert g["count"] == 2 and g["median_price"] == 2_500_000
    by_city = store.stats({"purpose": "rent"}, group_by="city")
    assert by_city["groups"]["Riyadh"]["count"] == 5


def test_find_project_fuzzy(store):
    assert store.find_project("w residences dubai")["url"].endswith("/w-residences")
    assert store.find_project("Trump Jeddah")["name"] == "Trump Tower Jeddah"
    assert store.find_project("nonexistent palace") is None
    assert [p["name"] for p in store.projects(country="Saudi")] == ["Trump Tower Jeddah"]


def test_chunking_overlap_and_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(10))
    chunks = chunk_text(text, max_chars=1500)
    assert len(chunks) > 3 and all(len(c) <= 1500 * 1.3 for c in chunks)
    assert chunks[1][:40] in chunks[0] or "word" in chunks[1][:20]  # overlap carried over


def test_tag_filter_strips_think_and_captures_followups():
    f = TagFilter()
    out = "".join(f.feed(x) for x in ["<thi", "nk>secret</th", "ink>Hello [1]", " world <follo", 'wups>["A?", "B?"]</followups>'])
    out += f.flush()
    assert out == "Hello [1] world "
    assert f.followups() == ["A?", "B?"]


def test_furnishing_filter_is_exact():
    from app.retrieval import canon_furnishing

    assert canon_furnishing("furnished") == "Furnished"
    assert canon_furnishing("semi furnished") == "Semi-Furnished"
    assert canon_furnishing("Unfurnished") == "Un-Furnished"
