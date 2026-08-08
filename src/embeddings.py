"""Create embeddings for movie plot texts."""

from __future__ import annotations

from typing import Any

import numpy as np


class Embedder:
    """Encode movie plots into dense vectors using sentence-transformers."""

    def __init__(self, config: dict) -> None:
        self.config = config
        emb = config["embeddings"]
        self.model_name = emb["model"]
        self.batch_size = emb.get("batch_size", 64)
        self.device = emb.get("device", "cpu")
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into a (n, dim) float32 array."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
