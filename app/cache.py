"""LRU + TTL answer cache. Saves free-model quota and makes repeated/suggested questions instant."""
from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from typing import Any


def normalize_question(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\s\u0600-\u06ff]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def cache_key(message: str, history: list[dict[str, str]] | None) -> str:
    # The last turn matters for follow-ups ("what about villas?"); older turns rarely change the answer.
    ctx = ""
    if history:
        last_user = [h["content"] for h in history if h.get("role") == "user"][-1:]
        ctx = normalize_question(" ".join(last_user))
    raw = normalize_question(message) + "||" + ctx
    return hashlib.sha256(raw.encode()).hexdigest()


class AnswerCache:
    def __init__(self, max_size: int = 512, ttl_s: int = 24 * 3600):
        self.max_size, self.ttl_s = max_size, ttl_s
        self._d: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._d.get(key)
            if not item or item[0] < time.time():
                if item:
                    del self._d[key]
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return item[1]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._d[key] = (time.time() + self.ttl_s, value)
            self._d.move_to_end(key)
            while len(self._d) > self.max_size:
                self._d.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)
