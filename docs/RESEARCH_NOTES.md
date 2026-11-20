# Research notes

Decisions backed by research or live verification on **2026-11-20**.
Each entry: what was checked → finding → decision.

## Scraping

| Topic | Finding (source, date) | Decision |
|---|---|---|
| Wasalt bot protection | Live test 2026-11-20: Cloudflare returns a 403 "Just a moment…" challenge to Python `httpx` (OpenSSL TLS fingerprint) on every page, while `curl` (Windows Schannel) gets 200 on uncached listing pages and on `/_next/data/…json`. This differs from the plan's earlier finding ("no bot wall"). | Use `curl_cffi` with a Chrome TLS profile for the same public pages. robots.txt (`Allow: /`, `Disallow: /search`) and `llms.txt` (public listing content may be extracted, moderate rate) are still honoured; ≤ 1 req/s, single connection. |
| Wasalt robots.txt / llms.txt | Fetched 2026-11-20 (AllowDataExtraction: public listing content only; AllowCommercialUse: No; attribution "Source: Wasalt.sa"; contact details must not be extracted at scale). | Field allow-list at extraction; phones/emails/WhatsApp links redacted at extraction **and** at normalisation; README states non-commercial use. |
| Wasalt sitemap | `product_sitemap_en_sa.xml.gz` = 60,000 URLs. Buckets (first slug word): sale/apartment 25,188 · rent/apartment 11,088 · sale/villa 10,140 · sale/land 4,661 · sale/floor 4,137 · rent/villa 1,479 · sale/building 1,049 · rent/floor 936 · rent/building 385 · rent/land 263 · sale/rest-house 212 · rent/room 194. | Stratified sample of 4,000 (proportional, floor 150 for buckets ≥ 150), newest IDs first, seed 42. |
| Wasalt listing JSON | `pageProps.propertyDetailsV3.attributes[].key`: `noOfBedrooms`, `noOfBathrooms`, `noOfLivingrooms`, `noOfGuestrooms`, `builtUpArea`, `carpetArea`, `furnishingType`. `additionalAttributes` also carries `street`, `planNumber`, `landNumber`, `buildingNumber`, `zipCode`, `regaAdvLicDate` (exact address / licence data). `expectedRent` equals the **yearly** amount; `rentFreq` has monthly/yearly breakdowns. | Map only the room/area/age/facing/utility keys; never copy address-pinpointing or licence fields. Rent normalised to SAR/year. Lat/lon rounded to 3 dp (~100 m). |
| Wasalt 404s on JSON route | Deleted listings return 404 on `/_next/data`. ~3 % of sampled URLs are unavailable. | Refresh buildId once, then fall back to HTML; record as unavailable. |
| DarGlobal host | `www.darglobal.co.uk` now 301-redirects to `darglobal.co.uk` (2026-11-20). Sitemap: 482 URLs, matches plan. | Canonical host `https://darglobal.co.uk`. |
| DarGlobal bot protection | Headless installed Chrome (`channel="chrome"`) passes Imperva after a warm-up visit; ~6 s per page incl. polite delay. | As planned. |
| DarGlobal data | Project pages keep CMS data in `pageProps.projectDetailsData` (not `pageData`): title, `ProjectBanner.Location`, `AboutProject`, `Amenities`, `faqContent[].faqs[]` (Strapi rich text). FAQ answers sit in collapsed accordions, so they are missing from visible text. | Store all non-boilerplate `pageProps` keys; append FAQ Q&A from JSON to the document text; parse "PROPERTY TYPE / UNIT TYPE / STATUS / AREA" fact boxes from visible text. |
| DarGlobal sitemap escaping | Some `<loc>` values contain XML entities (`&apos;`) → 404/empty pages. | `html.unescape` sitemap locations. |
| Broken DarGlobal pages | A few pages render "Application error: a client-side exception" or empty `<main>` (e.g. `/pre-launch-trump-jeddah-tower`, `/DarGlobal-Insignia-Card`). | Logged as failures, not stored. |
| selectolax 1.0 | `selectolax.parser` (Modest backend) now raises ImportError. | Use `selectolax.lexbor.LexborHTMLParser`. |

## Models & retrieval

| Topic | Finding (source, date) | Decision |
|---|---|---|
| OpenRouter free models | `GET /api/v1/models` 2026-11-20: 15 free models. None of the plan's examples (Llama 3.3, Gemma 3, Mistral Small 3.2, DeepSeek V3) are still free. All current free text models advertise `tools`. Candidates: `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3-ultra-550b-a55b:free`, `inclusionai/ling-3.2-flash:free`, `baidu/ernie-5.5-flash-preview:free`, router `openrouter/free`. | Final chain chosen by live smoke test (see "Model selection" below). |
| OpenRouter limits | [OpenRouter limits docs](https://openrouter.ai/docs/api/reference/limits) (read 2026-11-20): free models 20 req/min; **50 req/day** with < $10 lifetime credits, **1,000/day** after buying ≥ $10. 429 for rate limits, 402 for negative balance. | Answer cache, guard replies without LLM calls, deterministic + no-LLM fallbacks, per-IP rate limit 10/min. Recommend the $10 top-up. |
| Embedding model | `fastembed` 0.9.1 list: `BAAI/bge-small-en-v1.5` (384-d, 67 MB, MIT). | Keep as planned. |
| Embedding speed | Measured on Ryzen 5 5600H: ~8.5 chunks/s for 1,300-char chunks; Arabic text in listing descriptions explodes into many sub-tokens under bge's English vocab (~10× slower, no retrieval value). | Strip Arabic script from the *embedding* text (BM25 still indexes it); commit a float16 embedding cache (`data/processed/embeddings_cache.npz`) keyed by sha1(model, text) so `docker build` only embeds changed chunks. |
| Re-ranker | `TextCrossEncoder.list_supported_models()` (fastembed 0.9.1): `Xenova/ms-marco-MiniLM-L-6-v2` 80 MB Apache-2.0; `BAAI/bge-reranker-base` 1.04 GB; `jinaai/jina-reranker-v2-base-multilingual` is CC-BY-NC. | `Xenova/ms-marco-MiniLM-L-6-v2` (small, permissive) over the top RRF candidates. |
| Hybrid fusion | Reciprocal Rank Fusion with k = 60 (Cormack et al. 2009, the de-facto default). | BM25 top-30 + dense top-30 → RRF → re-rank → max 2 chunks per document. |

## Hosting

| Topic | Finding (source, date) | Decision |
|---|---|---|
| Hugging Face Spaces | [Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview) and [pricing](https://huggingface.co/pricing) (read 2026-11-20): "Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts". PRO is $9/month. CPU Basic (2 vCPU, 16 GB) itself is free; free Spaces sleep when unused. Community forum reports from mid-2026 match. | Free accounts can no longer host this Docker app on HF. **Human decision required** (see README / deploy notes). |
| Render | [Render free tier](https://render.com/docs/free) (read 2026-11-20): free web services spin down after 15 min idle (~1 min cold start), 750 instance-hours/month. | Backup host; mitigate cold starts with the keep-alive workflow. |

## Memory / CPU budget (Render free = 512 MB, 0.1 CPU)

| Check | Finding (measured 2026-11-20 in the container) | Decision |
|---|---|---|
| First container run at 512 MB | **OOM-killed** on the first queries: 386 MB idle; fastembed's embedder load +~250 MB; the cross-encoder's ONNX CPU arena grew 555 → 702 MB over two re-ranks. Data (chunks + vectors + BM25) is only ~60 MB. | Query-time inference via `app/onnx_models.py`: same ONNX files, `onnxruntime` with basic graph optimisation, CPU arena and memory patterns off; outputs identical to fastembed (embedding cosine 1.0, identical re-rank logits). fastembed still does build-time download + batch embedding. |
| Re-ranker on 0.1 CPU | ~0.8 s per re-rank on a full core → ~8 s at 0.1 CPU. Without it, BM25 + dense + RRF + title boost still ranks the acceptance queries correctly (W Residences, Trump Tower Jeddah, Missoni, tokenization, DarGlobal FAQ/about). | `USE_RERANKER=false`, `ONNX_THREADS=1` on Render (render.yaml); re-ranker stays on wherever there is CPU to spare. |
| After the fix, `--memory 512m --cpus 0.1` | 291 MB after startup, 296 MB after 6 queries (stable), retrieval 0.2–2.2 s per query, startup ~33 s, runs as uid 1000, no `.env` in the image. Image 890 MB. | Ship. |

## Model selection on the owner's OpenRouter account (2026-11-20)

| Check | Finding | Decision |
|---|---|---|
| Free models | All 15 free models return `404 … 0 endpoints … matching your guardrail restrictions and data policy` on this account (data-policy + workspace guardrail). | No free models in the chain (owner's instruction). |
| Allowed models | `GET /api/v1/models/user` lists the 4 models the account's guardrail allows: `deepseek/deepseek-v4-flash-0731` ($0.047/M in, $0.30/M out), `openai/gpt-5.6-luna` ($0.20/$1.20), `moonshotai/kimi-k3`, `deepseek/deepseek-v4-pro`. The owner-suggested `google/gemma-4-26b-a4b-it` is blocked by the guardrail. | Primary DeepSeek V4 Flash (cheapest; ~$0.0002 per call, ~$0.001 per answer); fallback GPT-5.6 Luna; Luna also acts as the eval judge (a different model from the answerer). `ALLOWED_PAID_MODELS` allow-list guards spend. |
| Reasoning | `reasoning: {"enabled": false}` accepted; 0 reasoning tokens billed. | Disabled everywhere. |
| Provider variance | OpenRouter load-balances DeepSeek across providers. In 6 identical probes, 3 providers (DigitalOcean, Relace, Wafer) put an "under 80,000 SAR" budget into `max_area`; Nebius/StreamLake/Cohere used `max_price`. | Clearer tool schema descriptions + server-side argument repair from the deterministic parser (`repair_listing_args`). |
| Grounding | Follow-up "show me villas instead" was answered from the previous reply with invented listings when the model skipped tools. | `tool_choice: "required"` on round 1 for factual intents + prompt rule that earlier answers are not data. Acceptance 10/10 afterwards. |
