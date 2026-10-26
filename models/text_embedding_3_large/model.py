from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI
import yaml


class TextEmbedding3LargeClient:
    """Thin config-driven wrapper around OpenAI text-embedding-3-large."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config_path = Path(config_path or Path(__file__).with_name("config.yaml")).resolve()
        self.config = self._load_config(config or {})
        api_key = str(self.config.get("api_key", "") or "").strip()
        if not api_key:
            raise ValueError("Missing OpenAI API key for text-embedding-3-large.")

        base_url = str(self.config.get("base_url", "") or "").strip() or None
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=float(self.config.get("timeout_seconds", 60) or 60))
        self.model_name = str(self.config.get("model", "text-embedding-3-large") or "text-embedding-3-large").strip()
        self.dimensions = int(self.config.get("dimensions", 3072) or 3072)
        self.batch_size = int(self.config.get("batch_size", 64) or 64)

    def _load_config(self, overrides: dict[str, Any]) -> dict[str, Any]:
        file_config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(file_config, dict):
            raise ValueError(f"Embedding config must be a mapping: {self.config_path}")
        merged = {**file_config, **overrides}

        api_key = str(merged.get("api_key", "") or "").strip()
        if not api_key:
            source_path_raw = str(merged.get("api_key_source_path", "") or "").strip()
            if source_path_raw:
                source_path = Path(source_path_raw)
                if not source_path.is_absolute():
                    source_path = (self.config_path.parent / source_path).resolve()
                source_config = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
                if not isinstance(source_config, dict):
                    raise ValueError(f"Expected mapping config for shared API key: {source_path}")
                api_key = str(source_config.get("api_key", "") or "").strip()
                if api_key:
                    merged["api_key"] = api_key
        if not api_key:
            env_name = str(merged.get("api_key_env", "OPENAI_API_KEY") or "").strip()
            api_key = str(os.environ.get(env_name, "") or "").strip()
            if api_key:
                merged["api_key"] = api_key
        return merged

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "input": batch,
            }
            if self.dimensions > 0:
                kwargs["dimensions"] = self.dimensions
            response = self.client.embeddings.create(**kwargs)
            batch_embeddings = [list(item.embedding) for item in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        items = self.embed_texts([text])
        return items[0] if items else []

    def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()
