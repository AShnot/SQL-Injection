"""Training pipeline for dynamic centroid + local signature detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .centroid_utils import (
    _unit_vector,
    add_or_update_centroid,
    build_class_index,
    prepare_centroid_payload,
    save_centroids_json,
    save_faiss_index,
    load_centroids_json,
)
from .data_loader import load_dataset
from .embedding_manager import EmbeddingManager
from .logging_utils import get_logger
from .preprocessing import iter_windows, lex_query, normalise_text, window_metadata
from .thresholds import ThresholdResult, calibrate_detection_threshold, save_thresholds

logger = get_logger(__name__)


@dataclass
class TrainingArtifacts:
    embeddings: np.ndarray
    ids: np.ndarray
    centroids: Dict[str, Dict[str, object]]
    faiss_index_built: bool
    thresholds: ThresholdResult


def ensure_directories() -> None:
    for path in [
        config.ARTIFACTS_DIR,
        config.EMBEDDINGS_DIR,
        config.CENTROIDS_DIR,
        config.MODELS_DIR,
        config.METRICS_DIR,
        config.LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify_key = df["label"].astype(str) + "_" + df["attack_technique"].astype(str)
    train_val, test = train_test_split(
        df,
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=stratify_key,
    )
    val_size = config.VALIDATION_SPLIT / (1 - config.TEST_SPLIT)
    stratify_key_train = train_val["label"].astype(str) + "_" + train_val["attack_technique"].astype(str)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=config.RANDOM_STATE,
        stratify=stratify_key_train,
    )
    return train.sort_index(), val.sort_index(), test.sort_index()


def build_embeddings(df: pd.DataFrame, manager: EmbeddingManager) -> np.ndarray:
    texts = df["normalized_query"].tolist()
    embeddings = manager.encode(texts)
    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_FILE, embeddings.astype(np.float32))
    np.save(config.IDS_FILE, df["row_id"].to_numpy(dtype=np.int32))
    return embeddings


def _positive_grouped(train_df: pd.DataFrame, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    positives = train_df[train_df["label"] == 1]
    grouped: Dict[str, np.ndarray] = {}
    for technique, group in positives.groupby("attack_technique"):
        rows = group["row_id"].to_numpy()
        vectors = embeddings[rows]
        try:
            grouped[technique] = _unit_vector(vectors)
        except ValueError:
            logger.warning("Skipping technique %s due to zero-norm centroid", technique)
    return grouped


def _collect_position_centroids(
    df: pd.DataFrame, manager: EmbeddingManager
) -> Dict[str, List[Dict[str, object]]]:
    per_technique: Dict[str, List[Dict[str, object]]] = {}
    for _, row in df[df["label"] == 1].iterrows():
        technique = row["attack_technique"]
        user_span = row.get("normalized_user_input", "")
        if not user_span:
            continue
        normalized_query = row["normalized_query"]
        start = normalized_query.find(user_span)
        if start == -1:
            continue
        end = start + len(user_span)
        tokens = lex_query(normalized_query)
        candidate_windows: List[Dict[str, object]] = []
        for _, _, token_window in iter_windows(tokens, config.WINDOW_TOKENS, config.WINDOW_STRIDE):
            meta = window_metadata(token_window, normalized_query)
            if not meta["text"]:
                continue
            if meta["end_char"] <= start or meta["start_char"] >= end:
                continue
            candidate_windows.append(meta)
        if not candidate_windows:
            continue
        embeddings = manager.encode([item["text"] for item in candidate_windows])
        per_technique.setdefault(technique, [])
        for meta, emb in zip(candidate_windows, embeddings):
            per_technique[technique].append({"text": meta["text"], "embedding": emb})
    return per_technique


def compute_thresholds(
    df_val: pd.DataFrame,
    centroids: Dict[str, np.ndarray],
    embeddings: np.ndarray,
) -> ThresholdResult:
    techniques = list(centroids.keys())
    if not techniques:
        raise RuntimeError("No centroids available for threshold calibration")
    matrix = np.stack([centroids[t] for t in techniques])
    val_vectors = embeddings[df_val["row_id"].to_numpy()]
    sims = val_vectors @ matrix.T
    best = sims.max(axis=1)
    return calibrate_detection_threshold(best, df_val["label"].to_numpy())


def build_and_save_indexes(
    centroids: Dict[str, np.ndarray],
    positions: Dict[str, List[Dict[str, object]]],
) -> Dict[str, Dict[str, object]]:
    store = prepare_centroid_payload(centroids, positions)
    if store:
        order = sorted(store.keys())
        store["__order__"] = order
        save_centroids_json(store)
        vectors = (np.array(store[name]["class_centroid"], dtype=np.float32) for name in order)
        index = build_class_index(vectors)
        save_faiss_index(index)
    return store


def rebuild_faiss_from_store(store: Dict[str, Dict[str, object]]) -> None:
    if not store:
        return
    order = store.get("__order__")
    if not order:
        order = sorted(name for name in store.keys() if not name.startswith("__"))
    vectors = [np.array(store[name]["class_centroid"], dtype=np.float32) for name in order]
    store["__order__"] = order
    save_centroids_json(store)
    index = build_class_index(vectors)
    save_faiss_index(index)


def train_pipeline(device: str | None = None) -> TrainingArtifacts:
    ensure_directories()
    manager = EmbeddingManager(device=device)
    manager.save_metadata()

    df = load_dataset()
    train_df, val_df, _ = split_dataset(df)

    embeddings = build_embeddings(df, manager)

    class_centroids = _positive_grouped(train_df, embeddings)
    position_centroids = _collect_position_centroids(train_df, manager)
    store = build_and_save_indexes(class_centroids, position_centroids)

    if not class_centroids:
        raise RuntimeError("No attack centroids computed; ensure positive samples are available")

    thresholds = compute_thresholds(val_df, class_centroids, embeddings)
    save_thresholds(thresholds)

    return TrainingArtifacts(
        embeddings=embeddings,
        ids=df["row_id"].to_numpy(),
        centroids=store,
        faiss_index_built=bool(store),
        thresholds=thresholds,
    )


def register_new_attack(
    technique: str,
    examples: List[Tuple[str, str]],
    manager: EmbeddingManager,
) -> Dict[str, Dict[str, object]]:
    """Register a new attack type from few-shot examples.

    Args:
        technique: Name of the new technique.
        examples: List of tuples ``(full_query, user_input)``.
        manager: Embedding manager for vectorisation.
    Returns:
        Updated centroid store dictionary.
    """

    if not examples:
        raise ValueError("At least one example is required to register a new attack")

    normalized_queries = [normalise_text(query) for query, _ in examples]
    query_embeddings = manager.encode(normalized_queries)
    try:
        class_centroid = _unit_vector(query_embeddings)
    except ValueError as exc:
        raise ValueError("Cannot compute centroid for the provided examples") from exc

    position_embeddings: List[Dict[str, object]] = []
    for (query, user_input) in examples:
        normalized_query = normalise_text(query)
        normalized_input = normalise_text(user_input)
        if not normalized_input:
            continue
        start = normalized_query.find(normalized_input)
        if start == -1:
            continue
        end = start + len(normalized_input)
        tokens = lex_query(normalized_query)
        windows: List[Dict[str, object]] = []
        for _, _, token_window in iter_windows(tokens, config.WINDOW_TOKENS, config.WINDOW_STRIDE):
            meta = window_metadata(token_window, normalized_query)
            if not meta["text"]:
                continue
            if meta["end_char"] <= start or meta["start_char"] >= end:
                continue
            windows.append(meta)
        if not windows:
            continue
        embeddings = manager.encode([meta["text"] for meta in windows])
        for meta, embedding in zip(windows, embeddings):
            position_embeddings.append({"text": meta["text"], "embedding": embedding})

    try:
        store = load_centroids_json()
    except FileNotFoundError:
        store = {}

    store = add_or_update_centroid(store, technique, class_centroid, position_embeddings)
    save_centroids_json(store)
    rebuild_faiss_from_store(store)
    return store

