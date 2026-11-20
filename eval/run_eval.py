"""Evaluate the chatbot end-to-end over its HTTP API (local or production).

Metrics
  * retrieval hit@5      expected source URL fragment appears among the first 5 cited sources
  * keyword recall       share of expected keywords present in the answer
  * refusal accuracy     out-of-scope / injection questions refused; in-scope questions answered
  * LLM-as-judge         faithfulness (claims supported by the cited sources) and relevance, 1-5,
                         scored by a *different* free model than the one that answered
  * latency              p50 / p95 server latency

Usage:
    python eval/run_eval.py --base-url http://localhost:7860
    python eval/run_eval.py --base-url https://<user>-realestate-chatbot.hf.space --no-judge
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
REFUSAL_MARKERS = re.compile(
    r"can't share|cannot share|can't help with|outside (?:what|the scope)|only (?:answer|help with|discuss)|"
    r"not able to|don't have (?:that|any|information)|not in (?:my|the collected) data|unrelated|"
    r"sorry|لا أستطيع|عذرا", re.I)
JUDGE_PROMPT = """You are grading a real-estate assistant. Given the QUESTION, the SOURCES it retrieved and its ANSWER,
score two things from 1 (bad) to 5 (excellent):
- faithfulness: every factual claim in the answer is supported by the SOURCES (no invented numbers or facts).
  An answer that correctly says the information is unavailable is faithful.
- relevance: the answer directly addresses the question.
Reply with ONLY JSON: {"faithfulness": n, "relevance": n, "reason": "<one sentence>"}"""


def load_env() -> dict[str, str]:
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
            if m:
                env[m.group(1)] = m.group(2).strip()
    return {**env, **{k: v for k, v in os.environ.items() if k.startswith("OPENROUTER")}}


def sse_lines(resp: httpx.Response):
    """Split an SSE stream on '\\n' only (as browsers do). httpx.iter_lines uses str.splitlines semantics,
    which also breaks on Unicode line separators that can legitimately appear inside JSON strings."""
    buf = ""
    for chunk in resp.iter_text():
        buf += chunk
        *lines, buf = buf.split("\n")
        for line in lines:
            yield line.rstrip("\r")
    if buf:
        yield buf


def ask(client: httpx.Client, base: str, message: str, history: list[dict]) -> dict:
    """POST a question; retry once if a pooled keep-alive connection was closed by the server meanwhile."""
    try:
        return _ask(client, base, message, history)
    except (httpx.RemoteProtocolError, httpx.ConnectError):
        return _ask(client, base, message, history)


def _ask(client: httpx.Client, base: str, message: str, history: list[dict]) -> dict:
    t0 = time.perf_counter()
    out = {"text": "", "sources": [], "cards": [], "done": {}, "status": [], "error": None}
    with client.stream("POST", f"{base}/api/chat", json={"message": message, "history": history},
                       timeout=180) as r:
        if r.status_code != 200:
            out["error"] = f"HTTP {r.status_code}: {r.read()[:200]!r}"
            return out
        event = None
        for line in sse_lines(r):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event:
                d = json.loads(line[5:].strip())
                if event == "token":
                    out["text"] += d["text"]
                elif event == "meta":
                    out["sources"] = d.get("sources", [])
                    out["notes"] = d.get("notes", [])
                elif event == "cards":
                    out["cards"] = d.get("items", [])
                elif event == "status":
                    out["status"].append(d.get("text"))
                elif event == "done":
                    out["done"] = d
                elif event == "error":
                    out["error"] = d.get("message")
    out["wall_ms"] = int((time.perf_counter() - t0) * 1000)
    return out


_DOCS: dict[str, str] | None = None
SCRAPE_DATE = "unknown"  # set from /api/health at the start of a run


def source_text(url: str) -> str:
    """Full text of a scraped document (what the model's tools returned), keyed by URL."""
    global _DOCS
    if _DOCS is None:
        _DOCS = {}
        p = ROOT / "data" / "processed" / "documents.jsonl"
        if p.exists():
            with p.open(encoding="utf-8") as f:  # file iteration splits on real newlines only (not U+2028)
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        _DOCS.setdefault(d["url"], d["text"])
    return _DOCS.get(url, "")


def evidence(res: dict, per_source: int = 3000, max_sources: int = 12) -> str:
    """Give the judge the same evidence the model had: the documents it actually cited (full text, capped),
    falling back to the source snippet (e.g. computed statistics)."""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", res["text"])}
    chosen = [s for s in res["sources"] if s["n"] in cited] or res["sources"]
    blocks = []
    for s in chosen[:max_sources]:
        body = s.get("snippet", "") if s["url"] == "https://wasalt.sa/en" else (source_text(s["url"]) or s.get("snippet", ""))
        blocks.append(f"[{s['n']}] {s.get('title')} | {s['url']}\n{body[:per_source]}")
    # facts the model legitimately had besides the documents: scrape date (system prompt) and tool summaries
    # such as "33 Wasalt listings match {...}; showing 6"
    header = f"SYSTEM FACTS: data collected on {SCRAPE_DATE}."
    if res.get("notes"):
        header += "\nTOOL SUMMARIES:\n" + "\n".join(res["notes"])
    return header + "\n\n" + "\n\n".join(blocks)


def judge(client: httpx.Client, key: str, model: str, q: str, res: dict) -> dict:
    sources = evidence(res)
    payload = {"model": model, "temperature": 0, "max_tokens": 200, "reasoning": {"enabled": False},
               "usage": {"include": True},
               "messages": [{"role": "system", "content": JUDGE_PROMPT},
                            {"role": "user", "content": f"QUESTION: {q}\n\nSOURCES:\n{sources or '(none)'}\n\nANSWER:\n{res['text'][:3000]}"}]}
    for attempt in range(3):
        r = client.post("https://openrouter.ai/api/v1/chat/completions", json=payload, timeout=90,
                        headers={"Authorization": f"Bearer {key}", "X-Title": "Real Estate Chatbot eval"})
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code != 200:
            return {}
        text = (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}
    return {}


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:7860")
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "report.md"))
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between questions (free-tier limits)")
    ap.add_argument("--judge-model", default="openai/gpt-5.6-luna")  # a different model from the answerer
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    with open(args.questions, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        items = items[: args.limit]
    key = load_env().get("OPENROUTER_API_KEY", "")
    use_judge = not args.no_judge and bool(key)
    base = args.base_url.rstrip("/")
    rows = []
    with httpx.Client() as client:
        health = client.get(f"{base}/api/health", timeout=30).json()
        global SCRAPE_DATE
        SCRAPE_DATE = health.get("scrape_date") or SCRAPE_DATE
        for i, it in enumerate(items, 1):
            history: list[dict] = []
            for prev in it.get("history", []):  # multi-turn: replay earlier turns first
                pr = ask(client, base, prev, history)
                history += [{"role": "user", "content": prev}, {"role": "assistant", "content": pr["text"][:3000]}]
                time.sleep(args.sleep)
            res = ask(client, base, it["question"], history)
            text_l = res["text"].lower()
            kws = it.get("expected_keywords", [])
            recall = sum(1 for k in kws if k.lower() in text_l) / len(kws) if kws else None
            frag = it.get("expected_source_contains")
            hit5 = None
            if frag:
                hit5 = any(frag.lower() in s["url"].lower() for s in res["sources"][:5])
            refused = bool(REFUSAL_MARKERS.search(res["text"])) and not res["cards"]
            refusal_ok = refused if it.get("should_refuse") else not (refused and not res["sources"])
            j = judge(client, key, args.judge_model, it["question"], res) if use_judge and res["text"] else {}
            row = {"id": it.get("id", i), "category": it.get("category", ""), "question": it["question"],
                   "hit5": hit5, "recall": recall, "refusal_ok": refusal_ok, "faith": j.get("faithfulness"),
                   "rel": j.get("relevance"), "judge_reason": j.get("reason"), "latency_ms": res["done"].get("latency_ms"), "model": res["done"].get("model"),
                   "path": res["done"].get("path"), "tools": res["done"].get("tools_used"), "error": res["error"],
                   "answer": res["text"][:400]}
            rows.append(row)
            print(f"[{i}/{len(items)}] {row['category']:<12} hit5={hit5} recall={recall} refusal_ok={refusal_ok} "
                  f"faith={row['faith']} rel={row['rel']} {row['latency_ms']}ms {row['path']} {row['model']}")
            time.sleep(args.sleep)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.mean(xs), 3) if xs else None

    hits = [r["hit5"] for r in rows if r["hit5"] is not None]
    lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None and r["path"] not in ("cache", "guard")]
    summary = {
        "questions": len(rows),
        "retrieval_hit@5": mean([1.0 if h else 0.0 for h in hits]),
        "keyword_recall": mean([r["recall"] for r in rows]),
        "refusal_accuracy": mean([1.0 if r["refusal_ok"] else 0.0 for r in rows]),
        "judge_faithfulness_1to5": mean([r["faith"] for r in rows]),
        "judge_relevance_1to5": mean([r["rel"] for r in rows]),
        "latency_p50_ms": round(pct(lat, 0.5)) if lat else None,
        "latency_p95_ms": round(pct(lat, 0.95)) if lat else None,
        "errors": sum(1 for r in rows if r["error"]),
    }
    models = sorted({r["model"] for r in rows if r["model"]})
    paths = {p: sum(1 for r in rows if r["path"] == p) for p in sorted({r["path"] for r in rows if r["path"]})}

    md = ["# Evaluation report\n",
          f"- Run: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} against `{base}`",
          f"- Dataset: {health.get('listings')} listings, {health.get('projects')} projects, "
          f"{health.get('chunks')} chunks (scraped {health.get('scrape_date')})",
          f"- Answer models used: {', '.join(models) or 'n/a'} · answer paths: {paths}",
          f"- Judge: `{args.judge_model}` (sees the full text of every source the answer cites, up to 12 sources, 3,000 chars each)" if use_judge else "- Judge: not run",
          "\n## Summary\n", "| Metric | Value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in summary.items()]
    md += ["\n## Per question\n", "| # | Category | Question | hit@5 | Recall | Refusal OK | Faith | Rel | ms | Path |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        q = r["question"].replace("|", "/")[:70]
        md.append(f"| {r['id']} | {r['category']} | {q} | {r['hit5']} | {r['recall'] if r['recall'] is None else round(r['recall'], 2)} "
                  f"| {r['refusal_ok']} | {r['faith']} | {r['rel']} | {r['latency_ms']} | {r['path']} |")
    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2,
                                                              ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
