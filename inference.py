from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

from sql_injection import config
from sql_injection.embedding_manager import EmbeddingManager
from sql_injection.logging_utils import get_logger
from sql_injection.preprocessing import iter_windows, lex_query, normalise_text, window_metadata

logger = get_logger(__name__)


@dataclass
class Prediction:
    full_query: str
    label: int
    predicted_type: Optional[str]
    confidence: float
    centroid_distance: float
    matched_window: Optional[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "full_query": self.full_query,
            "label": self.label,
            "predicted_type": self.predicted_type,
            "confidence": self.confidence,
            "centroid_distance": self.centroid_distance,
            "matched_window": self.matched_window,
        }


class SQLInjectionDetector:
    def __init__(self, device: Optional[str] = None):
        metadata = EmbeddingManager.load_metadata()
        self.manager = EmbeddingManager(model_name=metadata["model_name"], device=device)
        self.thresholds = self._load_thresholds()
        self.centroids = self._load_centroids()
        self.window_tokens = metadata.get("window_tokens", config.WINDOW_TOKENS)
        self.window_stride = metadata.get("window_stride", config.WINDOW_STRIDE)
        logger.info("Detector initialised with threshold %.4f", self.thresholds["detection_threshold"])

    def _load_thresholds(self) -> Dict[str, float]:
        with config.THRESHOLDS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_centroids(self) -> Dict[str, Dict[str, object]]:
        return joblib.load(config.CENTROIDS_FILE)

    def predict(self, full_query: str) -> Prediction:
        normalized = normalise_text(full_query)
        embedding = self.manager.encode([normalized])[0]
        techniques = [k for k in self.centroids.keys() if k != "__negative__"]
        if not techniques:
            logger.warning("No centroids available. Returning negative prediction by default.")
            return Prediction(full_query=full_query, label=0, predicted_type=None, confidence=0.0, centroid_distance=0.0, matched_window=None)

        centroid_matrix = np.stack([self.centroids[t]["centroid"] for t in techniques])
        similarities = centroid_matrix @ embedding
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        best_type = techniques[best_idx]
        label = int(best_sim >= self.thresholds["detection_threshold"])
        matched_window = None
        if label == 1:
            matched_window = self._localise_span(normalized, best_type)
        return Prediction(
            full_query=full_query,
            label=label,
            predicted_type=best_type if label else None,
            confidence=best_sim,
            centroid_distance=best_sim,
            matched_window=matched_window,
        )

    def _localise_span(self, normalized: str, technique: str) -> Optional[Dict[str, object]]:
        tokens = lex_query(normalized)
        windows = []
        for _, _, token_window in iter_windows(tokens, self.window_tokens, self.window_stride):
            meta = window_metadata(token_window, normalized)
            if meta["text"]:
                windows.append(meta)
        if not windows:
            return None
        embeddings = self.manager.encode([w["text"] for w in windows])
        centroid = self.centroids[technique]["centroid"]
        sims = embeddings @ centroid
        idx = int(np.argmax(sims))
        best = windows[idx]
        best["window_score"] = float(sims[idx])
        return best

    def add_new_technique(self, technique: str, samples: list[str]) -> None:
        """Add or update a centroid using a small number of labelled examples."""
        if not samples:
            raise ValueError("No samples provided for the new technique")
        normalized_samples = [normalise_text(sample) for sample in samples]
        embeddings = self.manager.encode(normalized_samples)
        centroid = embeddings.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0:
            raise ValueError("Cannot create centroid from zero vectors")
        centroid = centroid / norm
        entry = self.centroids.get(technique)
        if entry:
            n_old = entry.get("n_examples", 0)
            combined = entry["centroid"] * n_old + centroid * len(samples)
            combined_norm = np.linalg.norm(combined)
            combined = combined / combined_norm if combined_norm else centroid
            entry["centroid"] = combined.astype(np.float32)
            entry["n_examples"] = int(n_old + len(samples))
        else:
            self.centroids[technique] = {
                "centroid": centroid.astype(np.float32),
                "n_examples": int(len(samples)),
            }
        final_centroid = self.centroids[technique]["centroid"]
        overlaps = []
        for name, data in self.centroids.items():
            if name == technique or name == "__negative__":
                continue
            overlaps.append(float(np.dot(final_centroid, data["centroid"])))
        if overlaps and max(overlaps) > 0.95:
            logger.warning(
                "New centroid %s is highly similar to an existing technique (similarity=%.4f)",
                technique,
                max(overlaps),
            )
        joblib.dump(self.centroids, config.CENTROIDS_FILE)
        logger.info("Registered/updated centroid for %s with %d samples", technique, len(samples))



def cli() -> None:
    parser = argparse.ArgumentParser(description="SQL Injection detector inference")
    parser.add_argument("--query", required=True, help="Full SQL query to analyse")
    parser.add_argument("--device", default=None, help="Device for sentence transformer (cpu/cuda)")
    args = parser.parse_args()

    detector = SQLInjectionDetector(device=args.device)
    prediction = detector.predict(args.query)
    print(json.dumps(prediction.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()

