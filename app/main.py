"""FastAPI app: static chat UI + SSE chat API.

    uvicorn app.main:app --port 7860
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from .agent import Engine
from .cache import AnswerCache
from .config import get_settings
from .llm import OpenRouter
from .retrieval import HybridRetriever, ListingStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("app")

STATIC = Path(__file__).parent / "static"
SUGGESTIONS = [
    "Show me 3-bedroom apartments for rent in Riyadh under 80,000 SAR",
    "Tell me about W Residences Dubai — unit sizes and status",
    "Which DarGlobal projects are in Saudi Arabia?",
    "What's the average price of villas for sale in Jeddah?",
    "Any land for sale in Dammam?",
    "Compare DarGlobal's Trump-branded projects",
]


def client_ip(request: Request) -> str:
    """Real client IP behind the hosting proxy (HF Spaces / Render set X-Forwarded-For)."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


settings = get_settings()
limiter = Limiter(key_func=client_ip)


def build_engine(s=settings) -> Engine:
    if not (s.index_dir / "listings.sqlite").exists():
        raise RuntimeError(f"Index not found in {s.index_dir}. Run `python -m app.build_index` first.")
    embedder = reranker = None
    try:  # lean query-time ONNX sessions (~75 MB vs ~250 MB via fastembed); see app/onnx_models.py
        from .onnx_models import OnnxEmbedder, OnnxReranker

        embedder = OnnxEmbedder(s.embed_model, threads=s.onnx_threads)
        reranker = OnnxReranker(s.rerank_model, threads=s.onnx_threads) if s.use_reranker else None
    except Exception as e:  # noqa: BLE001 - fall back to fastembed's loader
        log.warning("lean ONNX loader unavailable (%s); using fastembed", e)
    retriever = HybridRetriever(s.index_dir, s.embed_model, s.rerank_model if s.use_reranker else None,
                                embedder=embedder, reranker=reranker)
    store = ListingStore(s.index_dir / "listings.sqlite")
    llm = OpenRouter(s.openrouter_api_key, s.models, s.openrouter_base_url, referer=s.app_public_url,
                     connect_timeout=s.connect_timeout_s, first_token_timeout=s.first_token_timeout_s,
                     allowed_paid=[m.strip() for m in s.allowed_paid_models.split(",") if m.strip()])
    return Engine(s, retriever, store, llm, AnswerCache(s.cache_size, s.cache_ttl_s))


async def prewarm(engine: Engine) -> None:
    """Answer the UI's suggested questions in the background so a reviewer's first clicks are instant."""
    await asyncio.sleep(5)
    for q in SUGGESTIONS:
        try:
            async for _ in engine.answer(q, []):
                pass
        except Exception as e:  # noqa: BLE001 - best effort only
            log.info("prewarm skipped (%s)", e)
            return
        await asyncio.sleep(6)
    log.info("prewarm done: %d cached answers", len(engine.cache))


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.perf_counter()
    engine = getattr(app.state, "engine", None) or build_engine()
    app.state.engine = engine
    await asyncio.to_thread(engine.retriever.embed_query, "warm up")  # load ONNX models now, not on 1st request
    if settings.use_reranker:
        await asyncio.to_thread(lambda: engine.retriever.reranker)
    log.info("ready in %.1fs: %d chunks, %d listings, %d projects, models=%s, llm_configured=%s",
             time.perf_counter() - t, len(engine.retriever.chunks), engine.store.count, engine.store.project_count,
             engine.llm.models, engine.llm.configured)
    task = asyncio.create_task(prewarm(engine)) if settings.prewarm_cache and engine.llm.configured else None
    yield
    if task:
        task.cancel()
    await engine.llm.aclose()


app = FastAPI(
    title="Property Guide — DarGlobal × Wasalt RAG chatbot",
    description="Agentic RAG over public DarGlobal and Wasalt data. `/api/chat` streams Server-Sent Events.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limited(request: Request, exc: RateLimitExceeded):
    return JSONResponse({"detail": "Too many requests — please wait a minute and try again."}, status_code=429)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Internal error — please try again."}, status_code=500)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    if request.url.path in ("/", "/index.html"):
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'self' https://huggingface.co https://*.hf.space",
        )
    return resp


class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=6000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Msg] = Field(default_factory=list, max_length=10)


class Feedback(BaseModel):
    message_id: str = Field(max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=500)


_LINE_SEPARATORS = {ord(chr(0x2028)): "\\u2028", ord(chr(0x2029)): "\\u2029", ord(chr(0x85)): "\\u0085"}


def sse(event: str, data) -> str:
    # ensure_ascii=False keeps Arabic compact; escape Unicode line separators that some SSE/line-based
    # clients (e.g. Python's str.splitlines) would treat as newlines and split a data line in two
    payload = json.dumps(data, ensure_ascii=False).translate(_LINE_SEPARATORS)
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/api/chat", summary="Ask a question (Server-Sent Events stream)")
@limiter.limit(settings.rate_limit)
async def chat(request: Request, body: ChatRequest):
    """Streams events: `status`, `meta` (sources), `cards`, `token`, `followups`, `done`, `error`."""
    engine: Engine = request.app.state.engine
    history = [m.model_dump() for m in body.history]

    async def gen():
        try:
            async for e in engine.answer(body.message.strip(), history):
                yield sse(e["event"], e["data"])
                if await request.is_disconnected():
                    break
        except Exception:  # noqa: BLE001 - never leak a stack trace to the client
            log.exception("chat stream failed")
            yield sse("error", {"message": "Something went wrong while answering. Please try again."})
            yield sse("done", {"error": True})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/feedback", summary="Thumbs up/down on an answer")
async def feedback(body: Feedback):
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **body.model_dump()}
    with (settings.log_dir / "feedback.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logging.getLogger("feedback").info(json.dumps(rec))
    return {"ok": True}


@app.get("/api/health", summary="Liveness + dataset counts")
async def health(request: Request):
    e: Engine = request.app.state.engine
    return {"status": "ok", "chunks": len(e.retriever.chunks), "listings": e.store.count,
            "projects": e.store.project_count, "models": e.llm.models, "llm_configured": e.llm.configured,
            "scrape_date": e.store.scrape_date, "cache_entries": len(e.cache), "repo_url": settings.repo_url or None}


@app.get("/api/stats", summary="Scrape report (dataset transparency)")
async def stats():
    p = settings.data_dir / "scrape_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@app.get("/api/suggestions", summary="Suggested starter questions")
async def suggestions():
    return {"items": SUGGESTIONS}


@app.get("/", include_in_schema=False)
async def index():
    # no-cache: browsers revalidate the page (cheap, via ETag) so a new ?v= asset version is picked up
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
