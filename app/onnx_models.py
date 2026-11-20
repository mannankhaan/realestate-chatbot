"""Lightweight ONNX runtime wrappers for query-time inference.

fastembed is used at build time (model download + batch embedding). At query time we run the *same*
ONNX files directly with onnxruntime + tokenizers, using basic graph optimisation and no CPU memory
arena. Measured in the container: the embedder costs ~75 MB instead of ~250 MB and memory no longer
grows per request, which keeps the app inside a 512 MB free-tier instance.
"""
from __future__ import annotations

import glob
import os
import tempfile
from pathlib import Path

import numpy as np


def _cache_roots() -> list[Path]:
    roots = []
    if os.getenv("FASTEMBED_CACHE_PATH"):
        roots.append(Path(os.environ["FASTEMBED_CACHE_PATH"]))
    roots.append(Path(tempfile.gettempdir()) / "fastembed_cache")
    return roots


def find_model_file(hf_repo: str, model_file: str) -> Path | None:
    """Locate a fastembed-downloaded ONNX file (HF hub cache layout) without network access."""
    org, name = hf_repo.split("/", 1)
    for root in _cache_roots():
        hits = glob.glob(str(root / f"models--{org}--{name}" / "snapshots" / "*" / model_file))
        if hits:
            return Path(sorted(hits)[-1])
    return None


def _model_source(model_name: str, kind: str) -> tuple[str, str]:
    if kind == "embed":
        from fastembed import TextEmbedding

        models = TextEmbedding.list_supported_models()
    else:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        models = TextCrossEncoder.list_supported_models()
    for m in models:
        if m["model"] == model_name:
            return m["sources"]["hf"], m["model_file"]
    raise ValueError(f"unknown {kind} model {model_name}")


def _session(path: Path, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.intra_op_num_threads = max(1, threads)
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _tokenizer(model_dir: Path, max_length: int):
    from tokenizers import Tokenizer

    tok_file = next((p for p in (model_dir / "tokenizer.json", model_dir.parent / "tokenizer.json") if p.exists()), None)
    if tok_file is None:
        raise FileNotFoundError(f"tokenizer.json not found near {model_dir}")
    tok = Tokenizer.from_file(str(tok_file))
    tok.enable_truncation(max_length=max_length)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")  # noqa: S106 - BERT padding token, not a secret
    return tok


def _feed(session, enc) -> dict[str, np.ndarray]:
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
    names = {i.name for i in session.get_inputs()}
    if "token_type_ids" in names:
        feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
    return {k: v for k, v in feed.items() if k in names}


class OnnxEmbedder:
    """BGE-style embedder: CLS pooling + L2 normalisation (same as fastembed for bge models)."""

    def __init__(self, model_name: str, threads: int = 1, max_length: int = 512):
        repo, file = _model_source(model_name, "embed")
        path = find_model_file(repo, file)
        if path is None:
            raise FileNotFoundError(f"{model_name} not downloaded; run `python -m app.build_index` first")
        self.session = _session(path, threads)
        self.tokenizer = _tokenizer(path.parent, max_length)

    def embed(self, texts, batch_size: int = 16):
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer.encode_batch(texts[i:i + batch_size])
            out = self.session.run(None, _feed(self.session, enc))[0]
            cls = out[:, 0, :] if out.ndim == 3 else out
            cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-12)
            yield from cls.astype(np.float32)


class OnnxReranker:
    """Cross-encoder re-ranker: score(query, doc) = logit of the relevance head."""

    def __init__(self, model_name: str, threads: int = 1, max_length: int = 512):
        repo, file = _model_source(model_name, "rerank")
        path = find_model_file(repo, file)
        if path is None:
            raise FileNotFoundError(f"{model_name} not downloaded; run `python -m app.build_index` first")
        self.session = _session(path, threads)
        self.tokenizer = _tokenizer(path.parent, max_length)

    def rerank(self, query: str, documents, batch_size: int = 12):
        docs = list(documents)
        for i in range(0, len(docs), batch_size):
            enc = self.tokenizer.encode_batch([(query, d) for d in docs[i:i + batch_size]])
            logits = self.session.run(None, _feed(self.session, enc))[0]
            yield from (logits[:, 0] if logits.ndim == 2 else logits).astype(float).tolist()
