from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from . import config
from .logging_utils import get_logger
from .preprocessing import lex_query, normalise_text, save_tokens

logger = get_logger(__name__)


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


REQUIRED_COLUMNS = {"full_query", "label", "user_inputs", "attack_technique"}


def load_dataset(path: str = None) -> pd.DataFrame:
    path = path or str(config.DATA_FILE)
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=["full_query", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df["attack_technique"] = df["attack_technique"].fillna("unknown")
    df["full_query"] = df["full_query"].astype(str)
    df["user_inputs"] = df["user_inputs"].fillna("").astype(str)

    bad_rows: List[Dict[str, str]] = []
    for idx, row in df.iterrows():
        if row["user_inputs"] and row["user_inputs"] not in row["full_query"]:
            logger.warning(
                "user_inputs not found in full_query for index %s", idx
            )
            bad_rows.append(
                {
                    "index": int(idx),
                    "full_query": row["full_query"],
                    "user_inputs": row["user_inputs"],
                    "label": int(row["label"]),
                    "attack_technique": row["attack_technique"],
                }
            )
    if bad_rows:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(bad_rows).to_csv(config.BAD_ROWS_FILE, index=False)
        logger.info("Saved %d problematic rows to %s", len(bad_rows), config.BAD_ROWS_FILE)

    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)
    df["normalized_query"] = df["full_query"].map(normalise_text)

    config.TOKEN_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        tokens = lex_query(row["normalized_query"])
        save_tokens(int(row["row_id"]), tokens)
    logger.info("Tokenised %d queries", len(df))

    return df


