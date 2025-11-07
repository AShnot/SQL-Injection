from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config
from .logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingManager:
    """Utility class around SentenceTransformer for encoding queries and windows."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or config.MODEL_NAME
        try:
            self.model = SentenceTransformer(self.model_name, device=device or "cpu")
        except Exception as exc:  # pragma: no cover - fallback rarely used
            logger.warning(
                "Failed to load model %s due to %s. Falling back to %s.",
                self.model_name,
                exc,
                config.FALLBACK_MODEL_NAME,
            )
            self.model_name = config.FALLBACK_MODEL_NAME
            self.model = SentenceTransformer(self.model_name, device=device or "cpu")
        self.normalize = config.NORMALIZE_EMBEDDINGS
        logger.info("Loaded sentence transformer model %s", self.model_name)

    def encode(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not isinstance(texts, (list, tuple)):
            texts = list(texts)
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        batch_size = batch_size or config.BATCH_SIZE
        show_bar = len(texts) >= batch_size
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_bar,
            normalize_embeddings=self.normalize,
        )
        if isinstance(embeddings, list):
            embeddings = np.asarray(embeddings)
        embeddings = embeddings.astype(np.float32)
        return embeddings

    def save_metadata(self) -> None:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_name": self.model_name,
            "normalize_embeddings": self.normalize,
            "window_tokens": config.WINDOW_TOKENS,
            "window_stride": config.WINDOW_STRIDE,
            "batch_size": config.BATCH_SIZE,
        }
        with config.VECTOR_META_FILE.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved embedding metadata to %s", config.VECTOR_META_FILE)

    @staticmethod
    def load_metadata(path: Path = config.VECTOR_META_FILE) -> Dict[str, object]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


