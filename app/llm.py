"""OpenRouter client: streaming chat completions with tool calls, timeouts and a model fallback chain."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger("llm")


class LLMError(Exception):
    """Model failed (5xx, timeout, bad response) -> try the next model."""


class RateLimited(LLMError):
    """429 / quota exhausted."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def parsed_args(self) -> dict[str, Any] | None:
        try:
            v = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return None
        return v if isinstance(v, dict) else None


@dataclass
class StreamResult:
    model: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    first_token_ms: int | None = None


class TagFilter:
    """Streaming filter: removes <think>...</think> and captures a trailing <followups>...</followups>
    block, even when tags are split across chunks."""

    HIDE = ("<think>", "</think>")
    CAPTURE = "<followups>"

    def __init__(self) -> None:
        self.buf = ""
        self.in_think = False
        self.captured: str | None = None  # text after <followups>

    def feed(self, chunk: str) -> str:
        if self.captured is not None:
            self.captured += chunk
            return ""
        self.buf += chunk
        out = []
        while self.buf:
            if self.in_think:
                i = self.buf.find("</think>")
                if i < 0:
                    self.buf = self.buf[-8:]
                    return "".join(out)
                self.buf = self.buf[i + 8:]
                self.in_think = False
                continue
            i_think, i_fu = self.buf.find("<think>"), self.buf.find(self.CAPTURE)
            idx = min([i for i in (i_think, i_fu) if i >= 0], default=-1)
            if idx >= 0:
                out.append(self.buf[:idx])
                if idx == i_fu:
                    self.captured = self.buf[idx + len(self.CAPTURE):]
                    self.buf = ""
                    break
                self.buf = self.buf[idx + 7:]
                self.in_think = True
                continue
            # keep a possible partial tag at the end
            lt = self.buf.rfind("<")
            if lt >= 0 and len(self.buf) - lt < len(self.CAPTURE) and ">" not in self.buf[lt:]:
                out.append(self.buf[:lt])
                self.buf = self.buf[lt:]
            else:
                out.append(self.buf)
                self.buf = ""
            break
        return "".join(out)

    def flush(self) -> str:
        rest, self.buf = ("" if self.in_think else self.buf), ""
        return rest

    def followups(self) -> list[str]:
        if not self.captured:
            return []
        raw = self.captured.split("</followups>")[0]
        try:
            items = json.loads(raw.strip())
        except json.JSONDecodeError:
            items = re.findall(r'"([^"]{5,140})"', raw)
        return [str(x).strip() for x in items if isinstance(x, str) and x.strip()][:3]


def is_free_model(model_id: str) -> bool:
    return model_id.endswith(":free")


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()


class OpenRouter:
    def __init__(self, api_key: str, models: list[str], base_url: str = "https://openrouter.ai/api/v1",
                 referer: str = "http://localhost:7860", title: str = "Real Estate Chatbot",
                 connect_timeout: float = 10.0, first_token_timeout: float = 25.0,
                 client: httpx.AsyncClient | None = None, allowed_paid: list[str] | tuple[str, ...] = ()):
        self.api_key = api_key
        # Cost guard: a model is only ever called if it is free (":free") or explicitly allow-listed
        # (ALLOWED_PAID_MODELS). Anything else is dropped from the chain and refused at call time.
        self.allowed_paid = set(allowed_paid)
        self.models = [m for m in models if self.is_allowed(m)]
        dropped = [m for m in models if m not in self.models]
        if dropped:
            log.warning("ignoring models that are neither free nor in ALLOWED_PAID_MODELS: %s", dropped)
        if not self.models:
            raise ValueError("no allowed OpenRouter models configured")
        self.spent_usd = 0.0  # OpenRouter-reported cost of all calls made by this process
        self.base_url = base_url.rstrip("/")
        self.first_token_timeout = first_token_timeout
        self.headers = {"HTTP-Referer": referer, "X-Title": title, "Content-Type": "application/json"}
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=connect_timeout))
        self.cooldown: dict[str, float] = {}  # model -> unix time until which we skip it

    def is_allowed(self, model: str) -> bool:
        return is_free_model(model) or model in self.allowed_paid

    def _check_cost(self, model: str) -> None:
        if not self.is_allowed(model):
            raise LLMError(f"refusing model {model!r}: not free and not in ALLOWED_PAID_MODELS")

    def _record_cost(self, usage: dict[str, Any] | None) -> None:
        try:
            self.spent_usd += float((usage or {}).get("cost") or 0)
        except (TypeError, ValueError):
            pass

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def available_models(self) -> list[str]:
        now = time.time()
        live = [m for m in self.models if self.cooldown.get(m, 0) < now]
        return live or list(self.models)  # if all cooling down, try anyway

    def _auth(self) -> dict[str, str]:
        return {**self.headers, "Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _raise_for(status: int, body: str, model: str) -> None:
        msg = body[:300]
        try:
            msg = json.loads(body).get("error", {}).get("message", msg)
        except (ValueError, AttributeError):
            pass
        if status == 429:
            raise RateLimited(f"{model}: 429 {msg}")
        if status == 402:
            raise RateLimited(f"{model}: 402 {msg}")
        raise LLMError(f"{model}: HTTP {status} {msg}")

    async def stream(self, messages: list[dict[str, Any]], model: str, tools: list[dict] | None = None,
                     tool_choice: str = "auto",
                     temperature: float = 0.2, max_tokens: int = 900) -> AsyncIterator[tuple[str, Any]]:
        """Yields ("token", str) as text arrives, then ("result", StreamResult) once at the end."""
        self._check_cost(model)
        # reasoning disabled: thinking tokens add latency and are billed as output on paid models
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True,
                                   "temperature": temperature, "max_tokens": max_tokens,
                                   "reasoning": {"enabled": False}, "usage": {"include": True}}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        res = StreamResult(model=model)
        calls: dict[int, dict[str, str]] = {}
        t0 = time.perf_counter()
        try:
            async with self.client.stream("POST", f"{self.base_url}/chat/completions", json=payload,
                                          headers=self._auth()) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    self._raise_for(resp.status_code, body, model)
                lines = resp.aiter_lines()
                got_data = False
                deadline = time.monotonic() + self.first_token_timeout
                while True:
                    timeout = max(0.1, deadline - time.monotonic()) if not got_data else 60.0
                    try:
                        line = await asyncio.wait_for(lines.__anext__(), timeout=timeout)
                    except StopAsyncIteration:
                        break
                    except TimeoutError as e:
                        raise LLMError(f"{model}: {'first-token' if not got_data else 'stream'} timeout") from e
                    if not line or line.startswith(":"):
                        continue  # keep-alive comments (": OPENROUTER PROCESSING")
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:
                        err = chunk["error"] or {}
                        code = err.get("code")
                        if code == 429:
                            raise RateLimited(f"{model}: {err.get('message')}")
                        raise LLMError(f"{model}: stream error {err.get('message')}")
                    if chunk.get("usage"):
                        res.usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if choice.get("finish_reason"):
                            res.finish_reason = choice["finish_reason"]
                        text = delta.get("content")
                        if text:
                            if not got_data:
                                res.first_token_ms = int((time.perf_counter() - t0) * 1000)
                            got_data = True
                            res.text += text
                            yield "token", text
                        for tc in delta.get("tool_calls") or []:
                            got_data = True
                            slot = calls.setdefault(tc.get("index", len(calls)), {"id": "", "name": "", "arguments": ""})
                            slot["id"] = tc.get("id") or slot["id"]
                            fn = tc.get("function") or {}
                            slot["name"] += fn.get("name") or ""
                            slot["arguments"] += fn.get("arguments") or ""
                        if delta.get("reasoning") and not got_data:
                            # reasoning tokens prove the model is alive; extend the deadline a little
                            deadline = max(deadline, time.monotonic() + 10)
        except httpx.TimeoutException as e:
            raise LLMError(f"{model}: timeout {type(e).__name__}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"{model}: {type(e).__name__} {e}") from e
        res.tool_calls = [ToolCall(id=c["id"] or f"call_{i}", name=c["name"], arguments=c["arguments"])
                          for i, c in sorted(calls.items()) if c["name"]]
        self._record_cost(res.usage)
        yield "result", res

    async def complete(self, messages: list[dict[str, Any]], model: str | None = None, temperature: float = 0.0,
                       max_tokens: int = 400, json_mode: bool = False) -> tuple[str, str]:
        """Non-streaming call with fallback. Returns (text, model)."""
        last: Exception | None = None
        for m in ([model] if model else self.available_models()):
            self._check_cost(m)
            payload: dict[str, Any] = {"model": m, "messages": messages, "temperature": temperature,
                                       "max_tokens": max_tokens, "reasoning": {"enabled": False},
                                       "usage": {"include": True}}
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            for attempt in range(2):
                try:
                    r = await self.client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._auth())
                    if r.status_code != 200:
                        self._raise_for(r.status_code, r.text, m)
                    body = r.json()
                    if "error" in body:
                        raise LLMError(f"{m}: {body['error']}")
                    self._record_cost(body.get("usage"))
                    text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                    return strip_think(text), m
                except RateLimited as e:
                    last = e
                    if attempt == 0:
                        await asyncio.sleep(2)
                        continue
                    self.cooldown[m] = time.time() + 60
                except (LLMError, httpx.HTTPError, ValueError) as e:
                    last = e
                break
        raise LLMError(f"all models failed: {last}")

    async def aclose(self) -> None:
        await self.client.aclose()
