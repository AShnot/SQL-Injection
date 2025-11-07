from pathlib import Path

# Configuration constants used across the training and inference pipeline.

DATA_FILE = Path("data.csv")

ARTIFACTS_DIR = Path("artifacts")
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
CENTROIDS_DIR = ARTIFACTS_DIR / "centroids"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
LOGS_DIR = ARTIFACTS_DIR / "logs"
ANALYSIS_DIR = ARTIFACTS_DIR / "analysis"

WINDOW_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "windows"
TOKEN_METADATA_DIR = EMBEDDINGS_DIR / "tokens"

EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings_full.npy"
IDS_FILE = EMBEDDINGS_DIR / "ids.npy"
WINDOW_META_TEMPLATE = "{row_id}_windows_meta.json"
WINDOW_EMBED_TEMPLATE = "{row_id}_windows.npy"
TOKEN_META_TEMPLATE = "{row_id}_tokens.json"

CENTROIDS_JSON_FILE = CENTROIDS_DIR / "centroids.json"
CLASS_INDEX_FILE = CENTROIDS_DIR / "class_index.faiss"

THRESHOLDS_FILE = METRICS_DIR / "thresholds.json"
VECTOR_META_FILE = MODELS_DIR / "vectorizer_meta.json"
BAD_ROWS_FILE = MODELS_DIR / "bad_rows.csv"
REPORT_FILE = METRICS_DIR / "report.json"
RUN_LOG_FILE = LOGS_DIR / "run.log"

MODEL_NAME = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WINDOW_TOKENS = 5
WINDOW_STRIDE = 1
BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
MIN_EXAMPLES_FOR_CENTROID = 5
SIMILARITY_METRIC = "cosine"

RANDOM_STATE = 42

