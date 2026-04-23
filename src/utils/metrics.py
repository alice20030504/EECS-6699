"""Post-hoc metrics computed from saved CSV results."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


Phase = Literal["benign", "tempered", "catastrophic"]

# Test Error Gap thresholds (see project plan Sec 4)
_BENIGN_MAX = 0.02
_TEMPERED_MAX = 0.10


def load_run(path: str | Path) -> pd.DataFrame:
    """Load a single-run CSV produced by CSVLogger."""
    return pd.read_csv(path)


def final_test_error(df: pd.DataFrame) -> float:
    """Return the test_error of the last epoch in a run CSV."""
    return float(df["test_error"].iloc[-1])


def classify_phase(gap: float) -> Phase:
    """Map a Test Error Gap to a phase label.

    Test Error Gap = TestErr(η, k) - TestErr(0, k)

    Thresholds (Mallinar et al. 2022 inspired, see configs/N1.yaml):
        Benign:       gap < 2%
        Tempered:     2% ≤ gap < 10%
        Catastrophic: gap ≥ 10%
    """
    if gap < _BENIGN_MAX:
        return "benign"
    elif gap < _TEMPERED_MAX:
        return "tempered"
    return "catastrophic"


def build_phase_table(results_dir: str | Path) -> pd.DataFrame:
    """Aggregate all N1 run CSVs into a (k, eta) phase table.

    Expects files named: results/N1/k{k}_eta{eta_pct}.csv
    where eta_pct is the integer percent (0, 10, 20, 40).

    Returns:
        DataFrame with columns [k, eta, test_error, gap, phase].

    TODO:
        1. Glob for all CSVs matching the naming pattern.
        2. For each file, parse k and eta from the filename.
        3. Compute final_test_error for each (k, eta) pair.
        4. For each k, compute gap = test_error(eta, k) - test_error(0, k).
        5. Apply classify_phase to each gap.
    """
    raise NotImplementedError("Aggregate N1 results into phase table.")
