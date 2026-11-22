"""Load the result CSVs written by reactagenthlt.evaluation.save_results."""

from pathlib import Path

import pandas as pd

# One row of a run is identified by these columns.
KEY = ["model", "dataset", "mode", "k", "budget", "question"]
CONFIG = ["model", "dataset", "k", "budget"]


def load(results_dir: str = "results") -> pd.DataFrame:
    """Concatenate the run CSVs of `results_dir` (summary.csv and other files are skipped).

    Keep only ONE run (same n and seed) per folder: rows of different runs would be mixed.
    """
    frames = []
    for file in sorted(Path(results_dir).glob("*.csv")):
        df = pd.read_csv(file)
        if set(KEY + ["correct"]).issubset(df.columns):  # skips summary.csv
            frames.append(df)
    if not frames:
        raise SystemExit(f"No run CSV files found in '{results_dir}'.")
    df = pd.concat(frames, ignore_index=True)
    for column in ("correct", "failed_action"):
        df[column] = df[column].astype(str).str.lower() == "true"
    print(f"Loaded {len(frames)} run files, {len(df)} rows.")
    return df.drop_duplicates(KEY, keep="last")
