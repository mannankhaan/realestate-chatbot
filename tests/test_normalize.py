import json
from pathlib import Path

from scraper.normalize import clean_listing, listing_fact_sheet, redact, remove_boilerplate, strapi_text
from scraper.wasalt import bucket_of, extract_listing, stratified_sample, strip_html

FIX = Path(__file__).parent / "fixtures"


def load_fixture():
    return json.loads((FIX / "wasalt_listing_pageprops.json").read_text(encoding="utf-8"))


def test_attribute_mapper_beds_baths_area():
    rec = extract_listing(load_fixture(), "https://wasalt.sa/en/property/rent/apartment-160-sqm-1")
    assert rec["bedrooms"] == 3
    assert rec["bathrooms"] == 2
    assert rec["built_up_area_sqm"] == 160.5
    assert rec["area_sqm"] == 160.5
    assert rec["purpose"] == "rent"
    assert rec["price"] == 75000  # yearly
    assert rec["rent_monthly"] == 6250
    assert rec["city"] == "Riyadh" and rec["district"] == "Al Malqa"
    assert rec["property_age"] == "New"
    assert rec["lat"] == 24.812  # rounded to 3 dp
    assert rec["price_per_sqm"] == round(75000 / 160.5, 1)


def test_pii_allow_list_drops_contact_and_owner_fields():
    rec = extract_listing(load_fixture(), "https://wasalt.sa/en/property/rent/apartment-160-sqm-1")
    blob = json.dumps(rec).lower()
    for forbidden in ("leadcontactinfo", "propertyowner", "fake agent", "0551234567", "agent@example.com",
                      "licence", "1234567890", "plannumber", "landnumber"):
        assert forbidden not in blob, forbidden
    cleaned = clean_listing(rec)
    assert "0551234567" not in json.dumps(cleaned)


def test_unapproved_listing_is_skipped():
    pp = load_fixture()
    pp["propertyDetailsV3"]["propertyInfo"]["statusTypeSlug"] = "expired"
    assert extract_listing(pp, "u") is None
    assert extract_listing({}, "u") is None


def test_html_stripping():
    assert strip_html("<p>Hello<br>world</p><script>x</script>") .startswith("Hello\nworld")
    assert strip_html(None) == ""


def test_phone_and_email_redaction():
    t = "Call 0551234567 or +966 55 123 4567, mail a.b@x.com, wa.me/966551234567. Price 1,350,000 SAR, 750 sqm."
    out = redact(t)
    assert "0551234567" not in out and "+966" not in out and "@" not in out and "wa.me" not in out
    assert "1,350,000" in out and "750 sqm" in out  # prices/areas survive


def test_boilerplate_removal():
    pages = [{"url": f"u{i}", "text": f"MENU\nCookie banner\nUnique content {i}\nBOOK UNIT"} for i in range(10)]
    out = remove_boilerplate(pages)
    assert out["u3"] == "Unique content 3"


def test_strapi_rich_text():
    blocks = [{"type": "paragraph", "children": [{"text": "Hello "}, {"text": "world"}]}]
    assert strapi_text(blocks) == "Hello world"


def test_fact_sheet_has_attribution():
    rec = extract_listing(load_fixture(), "https://wasalt.sa/en/property/rent/apartment-160-sqm-1")
    sheet = listing_fact_sheet(clean_listing(rec))
    assert "Apartment for rent in Al Malqa" in sheet
    assert "75,000 SAR/year" in sheet
    assert sheet.endswith("Source: Wasalt.sa")


def test_stratified_sample_respects_target_and_buckets():
    urls = [f"https://wasalt.sa/en/property/sale/apartment-100-sqm-{i}" for i in range(500)]
    urls += [f"https://wasalt.sa/en/property/rent/villa-300-sqm-{i}" for i in range(500, 520)]
    s = stratified_sample(urls, 50)
    assert len(s) == 50
    assert any(bucket_of(u) == ("rent", "villa") for u in s)  # rare bucket represented
    assert s == stratified_sample(urls, 50)  # deterministic


def test_implausible_yearly_rent_is_flagged_not_changed():
    base = {"purpose": "rent", "sub_type": "Apartment", "city": "Riyadh", "title": "t", "description_text": ""}
    flagged = clean_listing({**base, "price": 2600})
    assert flagged["price"] == 2600 and flagged["price_flag"] == "possibly_monthly"
    assert clean_listing({**base, "price": 45000})["price_flag"] is None
    assert clean_listing({**base, "sub_type": "Room", "price": 2600})["price_flag"] is None
    assert clean_listing({**base, "purpose": "sale", "price": 9000})["price_flag"] is None
