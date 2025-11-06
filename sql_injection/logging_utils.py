import logging
from pathlib import Path

from . import config


def setup_logging() -> logging.Logger:
    """Configure logging to both stdout and a file in the artifacts directory."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sql_injection")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(config.RUN_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging initialised. Artifacts will be stored in %s", config.ARTIFACTS_DIR)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve a child logger with the common configuration."""
    parent = setup_logging()
    return parent.getChild(name)

