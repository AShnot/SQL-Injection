from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import faiss  # type: ignore
import numpy as np

from sql_injection import config
from sql_injection.embedding_manager import EmbeddingManager
from sql_injection.logging_utils import get_logger
from sql_injection.pipeline import register_new_attack
from sql_injection.preprocessing import iter_windows, lex_query, normalise_text, window_metadata
from sql_injection.thresholds import ThresholdResult

logger = get_logger(__name__)


@dataclass
class Prediction:
    full_query: str
    label: int
    predicted_type: Optional[str]
    confidence: float
    centroid_similarity: float
    matched_window: Optional[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "full_query": self.full_query,
            "label": self.label,
            "predicted_type": self.predicted_type,
            "confidence": self.confidence,
            "centroid_similarity": self.centroid_similarity,
            "matched_window": self.matched_window,
        }


class SQLInjectionDetector:
    span_threshold: float = 0.85

    def __init__(self, device: Optional[str] = None):
        metadata = EmbeddingManager.load_metadata()
        self.manager = EmbeddingManager(model_name=metadata["model_name"], device=device)
        self.window_tokens = metadata.get("window_tokens", config.WINDOW_TOKENS)
        self.window_stride = metadata.get("window_stride", config.WINDOW_STRIDE)
        self.thresholds = self._load_thresholds()
        self.store = self._load_store()
        self.technique_order = self.store.get("__order__", [])
        self.position_lookup = self._prepare_position_lookup(self.store)
        self.class_index = self._load_faiss_index()

    def _load_thresholds(self) -> ThresholdResult:
        with config.THRESHOLDS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return ThresholdResult(**data)

    def _load_store(self) -> Dict[str, Dict[str, object]]:
        with config.CENTROIDS_JSON_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _prepare_position_lookup(
        self, store: Dict[str, Dict[str, object]]
    ) -> Dict[str, Dict[str, object]]:
        lookup: Dict[str, Dict[str, object]] = {}
        for name, data in store.items():
            if name.startswith("__"):
                continue
            entries: List[Dict[str, object]] = data.get("position_centroids", [])  # type: ignore[arg-type]
            if not entries:
                continue
            texts = [item.get("text", "") for item in entries]
            vectors = np.array([item.get("embedding", []) for item in entries], dtype=np.float32)
            if vectors.size == 0:
                continue
            lookup[name] = {"texts": texts, "embeddings": vectors}
        return lookup

    def _load_faiss_index(self) -> faiss.Index:
        try:
            index = faiss.read_index(str(config.CLASS_INDEX_FILE))
            if not self.technique_order:
                self.technique_order = [name for name in self.store.keys() if not name.startswith("__")]
            return index
        except (faiss.FaissException, OSError, ValueError):
            logger.warning("FAISS index missing; rebuilding from centroid store")
            if not self.technique_order:
                self.technique_order = [name for name in self.store.keys() if not name.startswith("__")]
            vectors = [
                np.array(self.store[name]["class_centroid"], dtype=np.float32)
                for name in self.technique_order
            ]
            if not vectors:
                raise RuntimeError("No centroids available to rebuild the FAISS index")
            dim = vectors[0].shape[0]
            index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
            vectors_array = np.stack(vectors).astype(np.float32)
            index.train(vectors_array)
            index.add(vectors_array)
            faiss.write_index(index, str(config.CLASS_INDEX_FILE))
            return index

    def predict(self, full_query: str) -> Prediction:
        normalized = normalise_text(full_query)
        embedding = self.manager.encode([normalized])[0]
        if not self.technique_order:
            return Prediction(full_query, 0, None, 0.0, 0.0, None)

        sims, indices = self.class_index.search(np.expand_dims(embedding, axis=0), 1)
        score = float(sims[0][0]) if sims.size else 0.0
        idx = int(indices[0][0]) if indices.size else -1
        predicted = self.technique_order[idx] if 0 <= idx < len(self.technique_order) else None
        label = int(score >= self.thresholds.detection_threshold)

        matched = None
        if label == 1 and predicted:
            matched = self._localise_span(full_query, normalized, predicted)

        return Prediction(
            full_query=full_query,
            label=label,
            predicted_type=predicted if label else None,
            confidence=score,
            centroid_similarity=score,
            matched_window=matched,
        )

    def _localise_span(
        self, original_query: str, normalized_query: str, technique: str
    ) -> Optional[Dict[str, object]]:
        reference = self.position_lookup.get(technique)
        if not reference:
            return None

        tokens = lex_query(normalized_query)
        windows: List[Dict[str, object]] = []
        texts: List[str] = []
        for _, _, token_window in iter_windows(tokens, self.window_tokens, self.window_stride):
            meta = window_metadata(token_window, normalized_query)
            if not meta["text"]:
                continue
            meta = dict(meta)
            meta["text"] = original_query[meta["start_char"] : meta["end_char"]]
            windows.append(meta)
            texts.append(meta.get("text", ""))
        if not windows:
            return None

        embeddings = self.manager.encode([normalise_text(text) for text in texts])
        position_vectors = reference["embeddings"]
        sims = embeddings @ position_vectors.T
        best_scores = sims.max(axis=1)

        candidates = []
        for meta, score in zip(windows, best_scores):
            entry = dict(meta)
            entry["window_score"] = float(score)
            if score >= self.span_threshold:
                candidates.append(entry)

        if not candidates:
            best_idx = int(np.argmax(best_scores))
            fallback = dict(windows[best_idx])
            fallback["window_score"] = float(best_scores[best_idx])
            return fallback

        candidates.sort(key=lambda item: item["start_char"])
        merged: List[Dict[str, object]] = []
        current = candidates[0]
        for candidate in candidates[1:]:
            if candidate["start_char"] <= current["end_char"]:
                current["end_char"] = max(current["end_char"], candidate["end_char"])
                current["window_score"] = max(current["window_score"], candidate["window_score"])
                current["text"] = original_query[current["start_char"] : current["end_char"]]
            else:
                current["text"] = original_query[current["start_char"] : current["end_char"]]
                merged.append(current)
                current = candidate
        current["text"] = original_query[current["start_char"] : current["end_char"]]
        merged.append(current)

        best = max(merged, key=lambda item: item["window_score"])
        return best

    def add_new_technique(self, technique: str, examples: List[tuple[str, str]]) -> None:
        register_new_attack(technique, examples, self.manager)
        self.store = self._load_store()
        self.technique_order = self.store.get("__order__", [])
        self.position_lookup = self._prepare_position_lookup(self.store)
        self.class_index = self._load_faiss_index()


def cli() -> None:
    parser = argparse.ArgumentParser(description="SQL injection detector")
    parser.add_argument("--query", required=True, help="SQL query to analyse")
    parser.add_argument("--device", default=None, help="Torch device for embeddings")
    args = parser.parse_args()

    detector = SQLInjectionDetector(device=args.device)
    prediction = detector.predict(args.query)
    print(json.dumps(prediction.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
