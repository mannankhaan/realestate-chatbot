"""Answer engine.

Order of strategies for each question:
  0. cache hit -> replay
  1. cheap guards (prompt injection, greetings) -> canned reply, no LLM call
  2. agentic RAG: the model calls tools (search_listings, listing_stats, search_knowledge, get_project,
     list_projects) for up to N rounds, then streams a cited answer. Models are tried in order.
  3. deterministic pipeline: query parser -> retrieval -> one LLM call without tools
  4. no-LLM answer rendered straight from the data (quota exhausted / all models down)
Every path emits the same SSE event types: status, meta, cards, token, followups, done, error.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .cache import AnswerCache, cache_key
from .llm import LLMError, OpenRouter, RateLimited, StreamResult, TagFilter
from .prompts import (
    CHITCHAT_REPLY,
    REFUSAL_INJECTION,
    STATUS_TEXT,
    SYSTEM_PROMPT,
    TOOLS,
    dumps,
    fmt_price,
    listing_card,
    listing_line,
    project_card,
    project_text,
)
from .query_parser import PARSE_PROMPT, Parsed, merge_llm, parse_llm_json, regex_parse
from .retrieval import Filters, HybridRetriever, ListingStore

log = logging.getLogger("agent")
reqlog = logging.getLogger("request")

TOOL_RESULT_MAX_CHARS = 7000
PRE_TOOL_HOLD_CHARS = 160
FACTUAL_INTENTS = {"listing_search", "listing_stats", "darglobal_info", "wasalt_info"}
Event = dict[str, Any]


def repair_listing_args(tool: str, args: dict[str, Any], parsed: Parsed) -> dict[str, Any]:
    """Correct well-known model slips in listing-tool arguments using the deterministic parser.

    Observed on some OpenRouter providers: a budget ("under 80,000 SAR") sent as max_area, and filters the
    user stated (rent/sale, city, type) dropped on retries. The parser is reliable for exactly these fields,
    so we trust it when the model contradicts or omits them."""
    if tool not in ("search_listings", "listing_stats") or parsed.intent not in ("listing_search", "listing_stats"):
        return args
    f, out = parsed.filters, dict(args)
    for bound in ("min", "max"):
        price, area = f.get(f"{bound}_price"), out.get(f"{bound}_area")
        if price is not None and out.get(f"{bound}_price") is None and area is not None and float(area) == float(price):
            out[f"{bound}_price"] = out.pop(f"{bound}_area")
        if price is not None and out.get(f"{bound}_price") is None:
            out[f"{bound}_price"] = price
    for k in ("purpose", "city", "sub_type", "furnishing"):
        if f.get(k) and not out.get(k):
            out[k] = f[k]
    if out != args:
        log.info("repaired %s args %s -> %s", tool, args, out)
    return out


def ev(name: str, data: Any) -> Event:
    return {"event": name, "data": data}


@dataclass
class Turn:
    """Mutable state for one question: numbered sources, cards, tools used."""
    sources: list[dict[str, Any]] = field(default_factory=list)
    cards: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # tool summaries, e.g. '33 listings match ...'
    url_to_n: dict[str, int] = field(default_factory=dict)

    def cite(self, url: str, title: str, site: str, snippet: str = "", key: str | None = None,
             snippet_chars: int = 300) -> int:
        """Number a source once per turn. `key` defaults to the URL; pass a distinct key when one URL backs
        several different results (e.g. two stats queries)."""
        key = key or url
        if key in self.url_to_n:
            return self.url_to_n[key]
        n = len(self.sources) + 1
        self.sources.append({"n": n, "url": url, "title": title, "site": site, "snippet": snippet[:snippet_chars]})
        self.url_to_n[key] = n
        return n


class Engine:
    def __init__(self, settings, retriever: HybridRetriever, store: ListingStore, llm: OpenRouter,
                 cache: AnswerCache):
        self.s = settings
        self.retriever = retriever
        self.store = store
        self.llm = llm
        self.cache = cache
        self.system_prompt = SYSTEM_PROMPT.format(scrape_date=store.scrape_date or "recently",
                                                  listing_count=f"{store.count:,}")

    # ------------------------------------------------------------------ tools (sync; run in threads)

    def tool_search_listings(self, args: dict[str, Any], turn: Turn) -> str:
        f = Filters.from_dict(args)
        res = self.store.search(f, sort=args.get("sort"), limit=int(args.get("limit") or 6))
        if not res.rows:
            return dumps({"matches": 0, "filters": res.filters, "relaxed": res.relaxed,
                          "note": "No listings match, even after relaxing filters. Say so and suggest alternatives."})
        lines = []
        for r in res.rows:
            n = turn.cite(r["url"], r["title"], "wasalt", listing_line(0, r)[4:])
            lines.append(listing_line(n, r))
            turn.cards.append(listing_card(r, n))
        head = f"{res.total} Wasalt listings match {dumps(res.filters)}; showing {len(res.rows)}."
        if res.relaxed:
            head += f" NOTE: no exact matches, so these filters were relaxed: {', '.join(res.relaxed)}."
        turn.notes.append(head)
        return head + "\n" + "\n".join(lines)

    def tool_listing_stats(self, args: dict[str, Any], turn: Turn) -> str:
        f = Filters.from_dict(args)
        st = self.store.stats(f, group_by=args.get("group_by"))
        body = dumps(st)
        scope = " ".join(str(x) for x in (f.purpose, f.sub_type, f"in {f.city}" if f.city else None,
                                         f.district, f"by {args['group_by']}" if args.get("group_by") else None) if x)
        title = f"Wasalt stats: {scope or 'all listings'} (sample of {self.store.count:,} listings)"
        # each stats query is its own source (same URL), with the full numbers kept for transparency/eval
        n = turn.cite("https://wasalt.sa/en", title, "wasalt", body, key=f"stats:{body}", snippet_chars=1500)
        return f"[{n}] Statistics computed over the collected Wasalt listings sample:\n{body}"

    def tool_search_knowledge(self, args: dict[str, Any], turn: Turn) -> str:
        site = args.get("site") if args.get("site") in ("darglobal", "wasalt") else None
        hits = self.retriever.search(str(args.get("query") or "")[:300], k=6, site=site,
                                     rerank=self.s.use_reranker, exclude_listings=site != "wasalt")
        if not hits:
            return "No relevant documents found."
        out = []
        for h in hits:
            n = turn.cite(h.url, h.title, h.site, h.text.split("\n", 1)[-1])
            out.append(f"[{n}] {'DarGlobal' if h.site == 'darglobal' else 'Wasalt'} | {h.title} | {h.url}\n{h.text[:1400]}")
        return "\n\n".join(out)

    def tool_get_project(self, args: dict[str, Any], turn: Turn) -> str:
        name = str(args.get("name") or "")
        p = self.store.find_project(name)
        if not p:
            # fall back to search so near-misses still find the page
            return self.tool_search_knowledge({"query": name, "site": "darglobal"}, turn)
        n = turn.cite(p["url"], p["name"], "darglobal", project_text(p))
        turn.cards.append(project_card(p, n))
        body = [c["text"] for c in self.retriever.chunks if c["url"] == p["url"]][:3]
        return f"[{n}] DarGlobal project | {p['name']} | {p['url']}\nFacts: {project_text(p)}\n" + "\n".join(body)[:4500]

    def tool_list_projects(self, args: dict[str, Any], turn: Turn) -> str:
        country = args.get("country")
        if country and country.lower() in ("ksa", "saudi", "saudi arabia"):
            country = "Saudi"
        rows = self.store.projects(country=country, query=args.get("keyword"))
        if not rows and args.get("keyword"):
            rows = self.store.projects(country=country)
        if not rows:
            return "No DarGlobal projects found for that filter."
        out = []
        for p in rows[:25]:
            n = turn.cite(p["url"], p["name"], "darglobal", project_text(p))
            turn.cards.append(project_card(p, n))
            out.append(f"[{n}] {p['name']} | {project_text(p)[:350]} | {p['url']}")
        return f"{len(rows)} DarGlobal projects:\n" + "\n".join(out)

    TOOL_FUNCS = {
        "search_listings": tool_search_listings, "listing_stats": tool_listing_stats,
        "search_knowledge": tool_search_knowledge, "get_project": tool_get_project,
        "list_projects": tool_list_projects,
    }

    async def run_tool(self, name: str, args: dict[str, Any], turn: Turn) -> str:
        fn = self.TOOL_FUNCS.get(name)
        if not fn:
            return f"Unknown tool {name}."
        turn.tools_used.append(name)
        try:
            out = await asyncio.to_thread(fn, self, args, turn)
        except Exception as e:  # noqa: BLE001 - tool errors are reported to the model, not the user
            log.exception("tool %s failed", name)
            out = f"Tool error: {type(e).__name__}"
        return out[:TOOL_RESULT_MAX_CHARS]

    # ------------------------------------------------------------------ main entry

    async def answer(self, message: str, history: list[dict[str, str]] | None = None) -> AsyncIterator[Event]:
        t0 = time.perf_counter()
        rid = uuid.uuid4().hex[:12]
        history = [h for h in (history or []) if h.get("role") in ("user", "assistant") and h.get("content")][-10:]
        info: dict[str, Any] = {"request_id": rid, "path": None, "model": None, "tools": [], "cached": False}

        key = cache_key(message, history)
        hit = self.cache.get(key)
        if hit:
            info.update(path="cache", model=hit.get("model"), cached=True)
            yield ev("status", {"text": "⚡ Instant answer (cached)"})
            async for e in self._replay(hit):
                yield e
            yield self._done(info, t0)
            return

        parsed = regex_parse(message, history, self.store.cities, self.store.districts)
        if parsed.is_injection or parsed.intent == "chitchat":
            info["path"] = "guard"
            text = REFUSAL_INJECTION if parsed.is_injection else CHITCHAT_REPLY
            yield ev("token", {"text": text})
            yield ev("followups", {"items": self._heuristic_followups(parsed)})
            yield self._done(info, t0)
            return

        turn = Turn()
        result: dict[str, Any] | None = None
        if self.llm.configured:
            async for e in self._agentic(message, history, parsed, turn, info):
                if e["event"] == "_result":
                    result = e["data"]
                else:
                    yield e
            if result is None and not info.get("emitted"):
                turn = Turn()
                async for e in self._pipeline(message, history, parsed, turn, info):
                    if e["event"] == "_result":
                        result = e["data"]
                    else:
                        yield e
        if result is None and not info.get("emitted"):
            info["path"] = "no_llm"
            turn = Turn()
            async for e in self._no_llm(message, parsed, turn):
                yield e
            info["tools"] = turn.tools_used
        if result:
            if not result["followups"]:
                result["followups"] = self._heuristic_followups(parsed)
            yield ev("followups", {"items": result["followups"]})
            if result.get("cacheable"):
                self.cache.set(key, result)
        elif info.get("emitted") and info["path"] != "no_llm":
            yield ev("followups", {"items": self._heuristic_followups(parsed)})
        yield self._done(info, t0)

    # ------------------------------------------------------------------ strategies

    async def _agentic(self, message: str, history: list[dict], parsed: Parsed, turn: Turn,
                       info: dict[str, Any]) -> AsyncIterator[Event]:
        hint = (f"Keyword-parser hint (may be incomplete or wrong): intent={parsed.intent}, "
                f"filters={dumps(parsed.filters)}, sort={parsed.sort}")
        base = [{"role": "system", "content": self.system_prompt}, *history,
                {"role": "system", "content": hint}, {"role": "user", "content": message}]
        for model in self.llm.available_models():
            messages = list(base)
            turn.__init__()  # reset sources between models
            malformed = 0
            emitted = ""
            filt = TagFilter()
            try:
                for rnd in range(self.s.max_tool_rounds + 1):
                    tools = TOOLS if rnd < self.s.max_tool_rounds else None
                    # Grounding: factual questions must look data up on the first round (models otherwise
                    # sometimes "answer" follow-ups from their previous reply and invent listings).
                    choice = "required" if rnd == 0 and parsed.intent in FACTUAL_INTENTS else "auto"
                    res: StreamResult | None = None
                    # Some models say "I'll search for..." before calling a tool. While tools are available,
                    # hold the first ~160 chars back: dropped if a tool call follows, streamed otherwise.
                    pending, live = "", tools is None
                    async for kind, val in self.llm.stream(messages, model, tools=tools, tool_choice=choice,
                                                           temperature=self.s.temperature):
                        if kind == "token":
                            out = filt.feed(val)
                            if not out:
                                continue
                            if not live:
                                pending += out
                                if len(pending) < PRE_TOOL_HOLD_CHARS:
                                    continue
                                out, pending, live = pending, "", True
                            emitted += out
                            info["emitted"] = True
                            yield ev("token", {"text": out})
                        else:
                            res = val
                    assert res is not None
                    info.setdefault("usage", []).append(res.usage)
                    # A full answer was already streamed this round: any extra tool calls are redundant
                    # (acting on them made some models write the whole answer a second time).
                    answered_live = live and tools is not None and rnd > 0
                    if res.tool_calls and answered_live:
                        log.info("%s streamed an answer and also called tools %s; ignoring the calls",
                                 model, [tc.name for tc in res.tool_calls])
                        res.tool_calls = []
                    if pending and not res.tool_calls:
                        emitted += pending
                        info["emitted"] = True
                        yield ev("token", {"text": pending})
                    if not res.tool_calls:
                        tail = filt.flush()
                        if tail:
                            emitted += tail
                            yield ev("token", {"text": tail})
                        if not emitted.strip():
                            raise LLMError(f"{model}: empty answer")
                        info.update(path="agentic", model=model, tools=turn.tools_used)
                        yield ev("_result", {"text": emitted, "sources": turn.sources, "cards": turn.cards,
                                             "followups": filt.followups(), "model": model,
                                             "cacheable": True})
                        return
                    # tool round
                    bad = [tc for tc in res.tool_calls if tc.parsed_args() is None or tc.name not in self.TOOL_FUNCS]
                    if bad:
                        malformed += 1
                        log.warning("%s malformed tool call(s): %s", model, [(b.name, b.arguments[:80]) for b in bad])
                        if malformed >= 2:
                            raise LLMError(f"{model}: malformed tool calls twice")
                    messages.append({"role": "assistant", "content": res.text or None, "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments or "{}"}}
                        for tc in res.tool_calls]})
                    for tc in res.tool_calls:
                        args = tc.parsed_args()
                        if args is None or tc.name not in self.TOOL_FUNCS:
                            content = "Invalid tool call: arguments must be a JSON object for a known tool."
                        else:
                            yield ev("status", {"text": STATUS_TEXT.get(tc.name, "Working…"), "tool": tc.name,
                                                "args": args})
                            args = repair_listing_args(tc.name, args, parsed)
                            content = await self.run_tool(tc.name, args, turn)
                        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": content})
                    yield ev("meta", {"sources": turn.sources, "intent": parsed.intent, "filters": parsed.filters,
                                  "notes": turn.notes})
                    if turn.cards:
                        yield ev("cards", {"items": turn.cards[:12]})
            except (LLMError, RateLimited) as e:
                log.warning("agentic: %s", e)
                if isinstance(e, RateLimited):
                    self.llm.cooldown[model] = time.time() + 60
                if emitted.strip():
                    # we already streamed part of an answer: finish gracefully instead of restarting
                    note = "\n\n_(The AI model was interrupted — the answer above may be incomplete.)_"
                    yield ev("token", {"text": note})
                    info.update(path="agentic_partial", model=model, tools=turn.tools_used)
                    yield ev("_result", {"text": emitted + note, "sources": turn.sources, "cards": turn.cards,
                                         "followups": [], "model": model, "cacheable": False})
                    return
                yield ev("status", {"text": "🔁 Switching to a backup model…"})
                continue

    async def _pipeline(self, message: str, history: list[dict], parsed: Parsed, turn: Turn,
                        info: dict[str, Any]) -> AsyncIterator[Event]:
        """Deterministic fallback: parse -> retrieve -> single LLM call without tools."""
        if not parsed.confident:
            try:
                raw, _ = await self.llm.complete(
                    [{"role": "system", "content": PARSE_PROMPT},
                     *[{"role": h["role"], "content": h["content"][:500]} for h in history[-4:]],
                     {"role": "user", "content": message}], json_mode=True, max_tokens=300)
                parsed = merge_llm(parsed, parse_llm_json(raw))
            except LLMError as e:
                log.info("LLM parse unavailable: %s", e)
        if parsed.intent == "out_of_scope":
            text = ("Sorry, that's outside what I can help with. I only answer real-estate questions using "
                    "public data from **DarGlobal** projects and **Wasalt** listings in Saudi Arabia.")
            info.update(path="pipeline_refusal", emitted=True)
            yield ev("token", {"text": text})
            yield ev("_result", {"text": text, "sources": [], "cards": [], "followups": [], "model": None,
                                 "cacheable": False})
            return
        context = await self._gather(parsed, message, turn)
        yield ev("meta", {"sources": turn.sources, "intent": parsed.intent, "filters": parsed.filters,
                                  "notes": turn.notes})
        if turn.cards:
            yield ev("cards", {"items": turn.cards[:12]})
        messages = [{"role": "system", "content": self.system_prompt}, *history,
                    {"role": "user", "content": f"CONTEXT (numbered sources):\n{context[:self.s.max_context_chars]}"
                                                f"\n\nQUESTION: {message}"}]
        for model in self.llm.available_models():
            filt, emitted = TagFilter(), ""
            try:
                async for kind, val in self.llm.stream(messages, model, tools=None, temperature=self.s.temperature):
                    if kind == "token":
                        out = filt.feed(val)
                        if out:
                            emitted += out
                            info["emitted"] = True
                            yield ev("token", {"text": out})
                    else:
                        info.setdefault("usage", []).append(val.usage)
                tail = filt.flush()
                if tail:
                    emitted += tail
                    yield ev("token", {"text": tail})
                if not emitted.strip():
                    raise LLMError("empty answer")
                info.update(path="pipeline", model=model, tools=turn.tools_used)
                yield ev("_result", {"text": emitted, "sources": turn.sources, "cards": turn.cards,
                                     "followups": filt.followups(), "model": model, "cacheable": True})
                return
            except (LLMError, RateLimited) as e:
                log.warning("pipeline: %s", e)
                if emitted.strip():
                    info.update(path="pipeline_partial", model=model)
                    yield ev("_result", {"text": emitted, "sources": turn.sources, "cards": turn.cards,
                                         "followups": [], "model": model, "cacheable": False})
                    return
                if isinstance(e, RateLimited):
                    self.llm.cooldown[model] = time.time() + 60

    async def _gather(self, parsed: Parsed, message: str, turn: Turn) -> str:
        q = parsed.rewritten_query or message
        parts: list[str] = []
        if parsed.intent == "listing_search":
            parts.append(await self.run_tool("search_listings", {**parsed.filters, "sort": parsed.sort, "limit": 6}, turn))
        elif parsed.intent == "listing_stats":
            parts.append(await self.run_tool("listing_stats", parsed.filters, turn))
            parts.append(await self.run_tool("search_listings", {**parsed.filters, "limit": 3}, turn))
        elif parsed.intent == "darglobal_info":
            country = next((c for c in ("Saudi", "UAE", "Oman", "Qatar", "Spain", "United Kingdom", "Maldives")
                            if c.lower() in message.lower() or (c == "Saudi" and "ksa" in message.lower())), None)
            if country or "projects" in message.lower():
                parts.append(await self.run_tool("list_projects", {"country": country}, turn))
            parts.append(await self.run_tool("search_knowledge", {"query": q, "site": "darglobal"}, turn))
        elif parsed.intent == "wasalt_info":
            parts.append(await self.run_tool("search_knowledge", {"query": q, "site": "wasalt"}, turn))
        elif parsed.intent != "out_of_scope":
            parts.append(await self.run_tool("search_knowledge", {"query": q, "site": "any"}, turn))
        return "\n\n".join(parts)

    async def _no_llm(self, message: str, parsed: Parsed, turn: Turn) -> AsyncIterator[Event]:
        yield ev("status", {"text": "📄 AI model busy — showing results straight from the data"})
        await self._gather(parsed, message, turn)
        yield ev("meta", {"sources": turn.sources, "intent": parsed.intent, "filters": parsed.filters,
                                  "notes": turn.notes})
        if turn.cards:
            yield ev("cards", {"items": turn.cards[:12]})
        lines = ["The AI model is busy right now (free-tier limits), so here are the most relevant results "
                 "straight from the collected data:\n"]
        if parsed.intent in ("listing_search", "listing_stats"):
            if parsed.intent == "listing_stats":
                st = self.store.stats(Filters.from_dict(parsed.filters))
                g = st["groups"].get("all", {})
                if g.get("count"):
                    lines.append(f"**{g['count']} listings** — median {fmt_price(g['median_price'])}, range "
                                 f"{fmt_price(g['min_price'])} – {fmt_price(g['max_price'])}.\n")
            for c in turn.cards:
                if c["kind"] == "listing":
                    lines.append(f"- [{c['title']}]({c['url']}) — {c['location']} — **{c['price']}** [{c['n']}]")
            lines.append("\nPrices may have changed — please check the live listing. Source: Wasalt.sa")
        else:
            for s in turn.sources[:5]:
                lines.append(f"- [{s['title']}]({s['url']}) [{s['n']}]")
        if len(lines) == 1:
            lines.append("I couldn't find anything relevant in the collected data. Please try again in a minute.")
        yield ev("token", {"text": "\n".join(lines)})
        yield ev("followups", {"items": self._heuristic_followups(parsed)})

    # ------------------------------------------------------------------ helpers

    async def _replay(self, hit: dict[str, Any]) -> AsyncIterator[Event]:
        if hit.get("sources"):
            yield ev("meta", {"sources": hit["sources"]})
        if hit.get("cards"):
            yield ev("cards", {"items": hit["cards"][:12]})
        text = hit["text"]
        for i in range(0, len(text), 60):
            yield ev("token", {"text": text[i:i + 60]})
            await asyncio.sleep(0.004)
        yield ev("followups", {"items": hit.get("followups") or []})

    def _heuristic_followups(self, parsed: Parsed) -> list[str]:
        f = parsed.filters
        city = f.get("city") or "Riyadh"
        if parsed.intent == "listing_search":
            other = "villas" if f.get("sub_type") != "Villa" else "apartments"
            return [f"Show me {other} instead", f"What's the average price in {city}?", "Show the cheapest ones first"]
        if parsed.intent == "listing_stats":
            return [f"Show me listings in {city}", "Compare prices by city", "What about rentals?"]
        if parsed.intent == "darglobal_info":
            return ["Which DarGlobal projects are in Saudi Arabia?", "Tell me about W Residences Dubai",
                    "Compare DarGlobal's Trump-branded projects"]
        return ["3-bedroom apartments for rent in Riyadh under 80k SAR", "What is DarGlobal and where do they build?",
                "Average price of villas for sale in Jeddah"]

    def _done(self, info: dict[str, Any], t0: float) -> Event:
        latency = int((time.perf_counter() - t0) * 1000)
        usage = info.get("usage") or []
        tokens = sum((u or {}).get("total_tokens", 0) for u in usage)
        cost = round(sum(float((u or {}).get("cost") or 0) for u in usage), 6)
        reqlog.info(json.dumps({"request_id": info["request_id"], "path": info["path"], "model": info["model"],
                                "tools": info["tools"], "latency_ms": latency, "tokens": tokens, "cost_usd": cost,
                                "llm_calls": len(usage), "cached": info["cached"],
                                "process_spend_usd": round(self.llm.spent_usd, 6)}))
        return ev("done", {"request_id": info["request_id"], "model": info["model"], "path": info["path"],
                           "tools_used": info["tools"], "latency_ms": latency, "cached": info["cached"],
                           "tokens": tokens, "llm_calls": len(usage)})
