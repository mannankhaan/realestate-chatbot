"""Shared scraping utilities: polite rate limiting, robots checks, resumable JSONL output, logging."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.robotparser
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# A normal desktop Chrome UA. No personal details in headers.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36"
)

# Extra deny patterns: stdlib robotparser handles wildcards poorly, so we enforce these ourselves.
GLOBAL_DENY = [re.compile(p) for p in (r"\?", r"utm_", r"/search(/|$)", r"/cdn-cgi/")]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


class RateLimiter:
    """Enforces a minimum interval between requests (single domain, single concurrency)."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def _delay(self) -> float:
        return max(0.0, self._last + self.min_interval_s - time.monotonic())

    def wait(self) -> None:
        d = self._delay()
        if d:
            time.sleep(d)
        self._last = time.monotonic()

    async def await_wait(self) -> None:
        d = self._delay()
        if d:
            await asyncio.sleep(d)
        self._last = time.monotonic()


class Robots:
    """robots.txt check (loaded once per domain) plus our own deny-patterns."""

    def __init__(self, robots_txt: str, extra_deny: Iterable[str] = ()):
        self.rp = urllib.robotparser.RobotFileParser()
        self.rp.parse(robots_txt.splitlines())
        self.extra = [re.compile(p) for p in extra_deny]

    def allowed(self, url: str) -> bool:
        if any(p.search(url) for p in GLOBAL_DENY + self.extra):
            return False
        return self.rp.can_fetch("*", url)


class JsonlWriter:
    """Append-mode JSONL writer; flushes each line so Ctrl-C never loses finished work."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = path.open("a", encoding="utf-8")

    def write(self, obj: dict[str, Any]) -> None:
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line truncated by Ctrl-C
    return out


def load_done_ids(path: Path, key: str = "url") -> set[str]:
    """IDs already scraped -> enables resume after interruption."""
    return {str(r[key]) for r in read_jsonl(path) if key in r}


def path_of(url: str) -> str:
    return urlsplit(url).path


def clean_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class Counters:
    def __init__(self) -> None:
        self.fetched = self.skipped = self.failed = 0

    def __str__(self) -> str:
        return f"fetched={self.fetched} skipped={self.skipped} failed={self.failed}"
