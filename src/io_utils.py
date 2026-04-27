"""Result persistence and Google Drive helpers.

All experiments use save_result() / load_results() for a consistent JSON schema.
get_result_dir() routes output to Drive when running in Colab (--drive flag).
"""
import json
from pathlib import Path


# ── Saving / loading ──────────────────────────────────────────────────────────

def save_result(result: dict, path: str) -> None:
    """Write *result* dict to a JSON file, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  [save] {p}")


def load_results(result_dir: str, pattern: str = '*.json') -> list[dict]:
    """Load all JSON results from *result_dir* matching *pattern*."""
    results = []
    for p in sorted(Path(result_dir).glob(pattern)):
        with open(p) as f:
            results.append(json.load(f))
    return results


# ── Google Drive / Colab helpers ──────────────────────────────────────────────

def mount_drive_if_colab(mount_point: str = '/content/drive') -> bool:
    """Mount Google Drive if running inside Google Colab.

    Returns True if mount succeeded, False otherwise.
    """
    try:
        import google.colab  # noqa: F401 — presence check only
        from google.colab import drive
        drive.mount(mount_point)
        print(f"[drive] mounted at {mount_point}")
        return True
    except ImportError:
        return False


def get_result_dir(
    base_dir: str,
    experiment: str,
    use_drive: bool = False,
    drive_root: str = '/content/drive/MyDrive/benign_overfitting',
) -> str:
    """Return (and create) the result directory for *experiment*.

    If *use_drive* is True and we're in Colab, results go to Google Drive so
    they survive session restarts.
    """
    if use_drive:
        path = f'{drive_root}/{experiment}'
    else:
        path = str(Path(base_dir) / 'results' / experiment)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
