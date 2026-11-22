"""Failure analysis: termination counts, a sheet to label wrong runs, label summary.

Usage (from the repo root):
    python analysis/failures.py counts            # termination reasons of wrong runs
    python analysis/failures.py sheet --per-group 25 [--models gemma qwen --modes react focused_react --datasets hotpotqa]
    python analysis/failures.py summary           # after filling the 'category' column
"""

import argparse
from pathlib import Path

import pandas as pd
from common import CONFIG, load

OUT = Path(__file__).parent / "out"
SHEET = OUT / "failure_sheet.csv"
# Categories of the slides (fill the 'category' column by hand with one of these).
CATEGORIES = ["wrong_format", "invalid_action", "goal_drift", "action_repetition",
              "bad_search_query", "label_mismatch", "wrong_reasoning", "other"]
# First guess taken from the termination reason; the rest must be labelled by hand.
AUTO = {
    "missing_action": "wrong_format", "malformed_action": "wrong_format",
    "multiple_actions": "wrong_format", "unsupported_action": "invalid_action",
    "invalid_tool_input": "invalid_action", "invalid_final_answer": "invalid_action",
}


def counts(df: pd.DataFrame) -> None:
    wrong = df[~df["correct"]]
    table = pd.crosstab([wrong[c] for c in CONFIG + ["mode"]], wrong["termination"], normalize="index") * 100
    table.insert(0, "wrong_runs", wrong.groupby(CONFIG + ["mode"]).size())
    table.insert(0, "runs", df.groupby(CONFIG + ["mode"]).size())
    OUT.mkdir(exist_ok=True)
    table.round(1).to_csv(OUT / "termination_counts.csv")
    print("Termination reasons, % of the wrong runs:")
    print(table.round(1).to_string())


def sheet(df: pd.DataFrame, per_group: int, seed: int, models=None, modes=None, datasets=None) -> None:
    wrong = df[~df["correct"]]
    if models:  # substring match, e.g. gemma qwen
        wrong = wrong[wrong["model"].str.contains("|".join(models), case=False)]
    if modes:
        wrong = wrong[wrong["mode"].isin(modes)]
    if datasets:
        wrong = wrong[wrong["dataset"].isin(datasets)]
    if wrong.empty:
        raise SystemExit("No wrong runs match these filters.")
    shuffled = wrong.sample(frac=1, random_state=seed)  # random order, then first N per group
    sample = shuffled.groupby(CONFIG + ["mode"]).head(per_group)
    columns = CONFIG + ["mode", "question", "gold", "prediction", "termination", "actions"]
    if "trace" in sample:  # only if the traces were saved
        columns.append("trace")
    sample = sample[columns].reset_index(drop=True)
    sample.insert(0, "id", sample.index)
    sample["auto_category"] = sample["termination"].map(AUTO).fillna("")
    sample["category"], sample["notes"] = "", ""
    OUT.mkdir(exist_ok=True)
    sample.to_csv(SHEET, index=False)
    print(f"{len(sample)} wrong runs written to {SHEET}\nCategories: {CATEGORIES}")


def summary() -> None:
    labelled = pd.read_csv(SHEET)
    labelled["label"] = labelled["category"].fillna(labelled["auto_category"]).replace("", pd.NA)
    unlabelled = labelled["label"].isna().sum()
    table = pd.crosstab([labelled["dataset"], labelled["mode"]], labelled["label"])
    percent = (pd.crosstab([labelled["dataset"], labelled["mode"]], labelled["label"], normalize="index") * 100).round(1)
    table.to_csv(OUT / "failure_labels.csv")
    print(f"Unlabelled rows: {unlabelled}\nCounts:\n{table}\n\nPercent of the labelled wrong runs:\n{percent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["counts", "sheet", "summary"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--per-group", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="*", help="only these models (substring), e.g. gemma qwen")
    parser.add_argument("--modes", nargs="*", help="only these modes, e.g. react focused_react")
    parser.add_argument("--datasets", nargs="*", help="only these datasets, e.g. hotpotqa")
    args = parser.parse_args()
    if args.command == "summary":
        summary()
    else:
        data = load(args.results_dir)
        counts(data) if args.command == "counts" else sheet(data, args.per_group, args.seed, args.models, args.modes, args.datasets)
