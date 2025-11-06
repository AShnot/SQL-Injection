from __future__ import annotations

import argparse
from sql_injection.logging_utils import setup_logging
from sql_injection.pipeline import train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SQL injection detection pipeline")
    parser.add_argument("--device", default=None, help="Device for sentence transformer (cpu/cuda)")
    args = parser.parse_args()

    logger = setup_logging()
    if args.device:
        logger.info("Device override requested: %s", args.device)
    train_pipeline(device=args.device)


if __name__ == "__main__":
    main()

