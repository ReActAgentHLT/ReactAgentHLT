from pathlib import Path

import pandas as pd

# One row of a run is identified by these columns.
KEY = ["model", "dataset", "mode", "k", "budget", "question"]
CONFIG = ["model", "dataset", "k", "budget"]


def load(results_dir: str = "results") -> pd.DataFrame:
    """Concatenate all CSV files of `results_dir`; a re-run replaces the old rows."""
    files = sorted(Path(results_dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in '{results_dir}'.")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    for column in ("correct", "failed_action"):
        df[column] = df[column].astype(str).str.lower() == "true"
    return df.drop_duplicates(KEY, keep="last")
