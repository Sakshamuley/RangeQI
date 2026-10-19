"""CLI: generate synthetic training dataset."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.data_gen import OUTPUT_PATH, save_training_csv


def main() -> None:
    path = save_training_csv(n_samples=8000)
    print(f"Saved training data to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
