"""Lightweight CSV logger for experiment metrics."""

import csv
import os
from pathlib import Path
from typing import Any


class CSVLogger:
    """Appends one row per epoch to a CSV file.

    Creates the output directory and writes a header on first use.

    Usage::

        logger = CSVLogger("results/R2a/k8.csv")
        logger.write_row({"epoch": 1, "train_loss": 0.5, "test_error": 0.3})
    """

    def __init__(self, path: str | os.PathLike, overwrite: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None
        self._header_written = False

        if overwrite and self.path.exists():
            self.path.unlink()

    def write_row(self, row: dict[str, Any]) -> None:
        if self._file is None:
            self._open(list(row.keys()))
        self._writer.writerow(row)
        self._file.flush()

    def _open(self, fieldnames: list[str]) -> None:
        self._file = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if not self._header_written and self.path.stat().st_size == 0:
            self._writer.writeheader()
        self._header_written = True

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
