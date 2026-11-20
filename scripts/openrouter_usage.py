"""Record OpenRouter key usage for budget tracking. Never prints the key or its label.

    python scripts/openrouter_usage.py <checkpoint-name>

Appends {checkpoint, usage, ...} to logs/openrouter_spend.jsonl and prints spend since the first
("baseline") checkpoint. Exits with code 3 if project spend reaches the stop threshold.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "openrouter_spend.jsonl"
STOP_AT_USD = 4.0
BUDGET_USD = 5.0


def key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^OPENROUTER_API_KEY=(.*)$", line.strip())
        if m and m.group(1).strip():
            return m.group(1).strip()
    sys.exit("OPENROUTER_API_KEY missing in .env")


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "check"
    r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key()}"}, timeout=30)
    r.raise_for_status()
    d = r.json().get("data", {})
    rec = {"checkpoint": name, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "usage": d.get("usage"),
           "usage_daily": d.get("usage_daily"), "limit": d.get("limit"), "limit_remaining": d.get("limit_remaining"),
           "is_free_tier": d.get("is_free_tier")}
    LOG.parent.mkdir(exist_ok=True)
    rows = [json.loads(x) for x in LOG.read_text(encoding="utf-8").splitlines()] if LOG.exists() else []
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    base = next((x for x in rows if x["checkpoint"] == "baseline"), rec)
    spent = float(rec["usage"] or 0) - float(base["usage"] or 0)
    print(f"[{name}] usage=${float(rec['usage'] or 0):.6f} | project spend since baseline=${spent:.6f} "
          f"(stop at ${STOP_AT_USD:.2f}, budget ${BUDGET_USD:.2f}) | limit_remaining={rec['limit_remaining']}")
    return 3 if spent >= STOP_AT_USD else 0


if __name__ == "__main__":
    sys.exit(main())
