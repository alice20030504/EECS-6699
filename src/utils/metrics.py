"""Post-hoc metrics computed from saved CSV results (experiments/) or JSON results (run_r1/r2)."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


Phase = Literal["benign", "tempered", "catastrophic"]

# Test Error Gap thresholds (see project plan Sec 4 / configs/N1.yaml)
_BENIGN_MAX   = 0.03   # Δ < 3%   → benign      (plan uses 3%)
_TEMPERED_MAX = 0.10   # Δ < 10%  → tempered


def load_run(path: str | Path) -> pd.DataFrame:
    """Load a single-run CSV produced by CSVLogger."""
    return pd.read_csv(path)


def final_test_error(df: pd.DataFrame) -> float:
    """Return the test_error of the last epoch in a run CSV."""
    return float(df["test_error"].iloc[-1])


def classify_phase(gap: float) -> Phase:
    """Map a Test Error Gap to a phase label.

    Test Error Gap = TestErr(η, k) − TestErr(0, k*)

    Thresholds (plan Sec 4 / Mallinar 2022 inspired):
        Benign:       gap < 3%
        Tempered:     3% ≤ gap < 10%
        Catastrophic: gap ≥ 10%
    """
    if gap < _BENIGN_MAX:
        return "benign"
    elif gap < _TEMPERED_MAX:
        return "tempered"
    return "catastrophic"


def build_phase_table(results_dir: str | Path) -> pd.DataFrame:
    """Aggregate all N1 run CSVs into a (k, eta, test_error, gap, phase) table.

    Expects CSV files named: k{k}_eta{eta_pct}.csv  (e.g. k8_eta20.csv)
    where eta_pct is the integer percentage (0, 10, 20, 40).

    Returns:
        DataFrame with columns [k, eta, test_error, gap, phase].
    """
    results_dir = Path(results_dir)
    rows = []

    for csv_path in sorted(results_dir.glob("k*_eta*.csv")):
        stem   = csv_path.stem          # e.g. "k8_eta20"
        parts  = stem.split("_")
        k      = int(parts[0][1:])      # "k8" → 8
        eta    = int(parts[1][3:]) / 100.0   # "eta20" → 0.20

        df = load_run(csv_path)
        te = final_test_error(df)
        rows.append({"k": k, "eta": eta, "test_error": te})

    if not rows:
        raise FileNotFoundError(f"No k*_eta*.csv files found in {results_dir}")

    table = pd.DataFrame(rows).sort_values(["k", "eta"]).reset_index(drop=True)

    # Baseline: test error at eta=0 for each k
    baseline = (
        table[table["eta"] == 0.0]
        .set_index("k")["test_error"]
        .to_dict()
    )
    # Use minimum over all baselines if eta=0 is missing for some k
    global_baseline = min(baseline.values()) if baseline else 0.0

    table["gap"]   = table.apply(
        lambda r: r["test_error"] - baseline.get(r["k"], global_baseline), axis=1
    )
    table["phase"] = table["gap"].apply(classify_phase)

    return table


def build_phase_table_from_json(results_dir: str | Path) -> pd.DataFrame:
    """Same as build_phase_table but reads run_r1/run_r2 style JSON files.

    Expects JSON files produced by run_r1.py / run_n1.py (future).
    JSON must have keys: width_multiplier, noise_rate, test_error.
    """
    import json

    results_dir = Path(results_dir)
    rows = []

    for json_path in sorted(results_dir.glob("*.json")):
        with open(json_path) as f:
            r = json.load(f)
        if "width_multiplier" in r and "noise_rate" in r:
            rows.append({
                "k":          r["width_multiplier"],
                "eta":        r["noise_rate"],
                "test_error": r["test_error"],
            })

    if not rows:
        raise FileNotFoundError(f"No compatible JSON files in {results_dir}")

    table = pd.DataFrame(rows).sort_values(["k", "eta"]).reset_index(drop=True)
    baseline = (
        table[table["eta"] == 0.0]
        .set_index("k")["test_error"]
        .to_dict()
    )
    global_baseline = min(baseline.values()) if baseline else 0.0
    table["gap"]   = table.apply(
        lambda r: r["test_error"] - baseline.get(r["k"], global_baseline), axis=1
    )
    table["phase"] = table["gap"].apply(classify_phase)
    return table
