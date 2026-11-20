"""Runtime configuration (env vars / .env)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

# OpenRouter models, tried in order. All free models are blocked by the owning account's guardrail, so these
# are the cheapest allowed paid models that passed a live tool-calling test on 2026-11-20 (see
# docs/RESEARCH_NOTES.md). Override with OPENROUTER_MODELS="a,b,c" (+ ALLOWED_PAID_MODELS for paid ids).
DEFAULT_MODELS = [
    "deepseek/deepseek-v4-flash-0731",  # primary: cheapest model allowed on the account, clean tool calls
    "openai/gpt-5.6-luna",              # fallback: only used if the primary errors / times out
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = Field(default="", repr=False)
    openrouter_models: str = ""
    # Cost guard: paid models are only ever called if listed here (comma-separated). ":free" ids always allowed.
    allowed_paid_models: str = "deepseek/deepseek-v4-flash-0731,openai/gpt-5.6-luna"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    app_public_url: str = "http://localhost:7860"
    app_title: str = "Property Guide"
    repo_url: str = ""  # public source-code link shown in the UI footer (hidden when empty)

    data_dir: Path = ROOT / "data" / "processed"
    index_dir: Path = ROOT / "index"
    log_dir: Path = ROOT / "logs"

    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True  # set false on tiny CPUs (Render free = 0.1 CPU): RRF + title boost only
    onnx_threads: int = 2

    connect_timeout_s: float = 10.0
    first_token_timeout_s: float = 25.0
    max_tool_rounds: int = 3
    temperature: float = 0.2
    max_context_chars: int = 22_000  # ~6k tokens

    rate_limit: str = "10/minute"
    cache_size: int = 512
    cache_ttl_s: int = 24 * 3600
    prewarm_cache: bool = False  # costs ~2-3 free-model calls per suggestion; enable with credits

    @field_validator("openrouter_api_key", "openrouter_models", "allowed_paid_models", "app_public_url", mode="before")
    @classmethod
    def _strip(cls, v):
        # `docker --env-file` and dashboard copy/paste keep stray spaces/quotes; a key with a leading space
        # produced "401 Missing Authentication header" in the container.
        return v.strip().strip('"').strip("'").strip() if isinstance(v, str) else v

    @property
    def models(self) -> list[str]:
        if self.openrouter_models.strip():
            return [m.strip() for m in self.openrouter_models.split(",") if m.strip()]
        return list(DEFAULT_MODELS)


@lru_cache
def get_settings() -> Settings:
    return Settings()
