"""Utilities for storing and retrieving dynamic centroids and FAISS indexes."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

import faiss  # type: ignore
import numpy as np

from . import config
from .logging_utils import get_logger

logger = get_logger(__name__)


def _unit_vector(vectors: np.ndarray) -> np.ndarray:
    centroid = vectors.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0:
        raise ValueError("Cannot create centroid from zero vectors")
    return (centroid / norm).astype(np.float32)


def prepare_centroid_payload(
    class_vectors: Dict[str, np.ndarray],
    position_vectors: Dict[str, List[Dict[str, object]]],
) -> Dict[str, Dict[str, object]]:
    payload: Dict[str, Dict[str, object]] = {}
    for technique, centroid in class_vectors.items():
        payload[technique] = {
            "class_centroid": centroid.astype(np.float32).tolist(),
            "position_centroids": [
                {"text": item["text"], "embedding": item["embedding"].astype(np.float32).tolist()}
                for item in position_vectors.get(technique, [])
            ],
        }
    return payload


def save_centroids_json(store: Dict[str, Dict[str, object]]) -> None:
    config.CENTROIDS_DIR.mkdir(parents=True, exist_ok=True)
    with config.CENTROIDS_JSON_FILE.open("w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
    logger.info("Saved centroid store to %s", config.CENTROIDS_JSON_FILE)


def load_centroids_json() -> Dict[str, Dict[str, object]]:
    with config.CENTROIDS_JSON_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_class_index(vectors: Iterable[np.ndarray]) -> faiss.Index:
    vector_list = list(vectors)
    if not vector_list:
        raise ValueError("No vectors provided to build the class index")
    vectors_array = np.stack(vector_list).astype(np.float32)
    dim = vectors_array.shape[1]
    index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors_array)
    index.add(vectors_array)
    return index


def save_faiss_index(index: faiss.Index) -> None:
    config.CENTROIDS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.CLASS_INDEX_FILE))
    logger.info("Persisted FAISS class index to %s", config.CLASS_INDEX_FILE)


def load_faiss_index() -> faiss.Index:
    return faiss.read_index(str(config.CLASS_INDEX_FILE))


def add_or_update_centroid(
    store: Dict[str, Dict[str, object]],
    technique: str,
    class_embedding: np.ndarray,
    position_embeddings: List[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    entry = store.get(technique, {"position_centroids": []})
    entry["class_centroid"] = class_embedding.astype(np.float32).tolist()
    existing_positions = entry.get("position_centroids", [])
    existing_positions.extend(
        {"text": item["text"], "embedding": item["embedding"].astype(np.float32).tolist()}
        for item in position_embeddings
    )
    entry["position_centroids"] = existing_positions
    store[technique] = entry
    return store


__all__ = [
    "_unit_vector",
    "prepare_centroid_payload",
    "save_centroids_json",
    "load_centroids_json",
    "build_class_index",
    "save_faiss_index",
    "load_faiss_index",
    "add_or_update_centroid",
]
