
"""Sanity checks on the raw result CSVs (run this before trusting any table).

Usage (from the repo root):  python analysis/check_results.py [results_dir]
"""

import sys
from pathlib import Path

import pandas as pd

folder = sys.argv[1] if len(sys.argv) > 1 else "results"
problems = []
for file in sorted(Path(folder).glob("*.csv")):
    df = pd.read_csv(file)
    if "correct" not in df.columns:
        continue  # summary.csv
    df["correct"] = df["correct"].astype(str).str.lower() == "true"
    line = f"{file.name[:58]:58} rows={len(df):4} unique_q={df['question'].nunique():4} correct={df['correct'].sum():3}"
    if "f1" in df.columns:  # HotpotQA files
        em1 = df["em"] == 1
        bad = {
            "f1 outside [0,1]": int(((df["f1"] < 0) | (df["f1"] > 1)).sum()),
            "em not in {0,1}": int((~df["em"].isin([0, 1])).sum()),
            "em=1 but f1<1": int((em1 & (df["f1"] < 1)).sum()),
            "em=1 but correct=False": int((em1 & ~df["correct"]).sum()),
        }
        line += f" em1={int(em1.sum()):3} f1_max={df['f1'].max():.2f} em_max={df['em'].max():.2f}"
        for name, count in bad.items():
            if count:
                problems.append(f"{file.name}: {name}: {count} rows")
        stuck = df[em1 & ~df["correct"]]
        if len(stuck):
            problems.append(f"   terminations of em=1 but correct=False: {stuck['termination'].value_counts().to_dict()}")
    if len(df) != df["question"].nunique():
        problems.append(f"{file.name}: duplicated questions ({len(df)} rows, {df['question'].nunique()} unique)")
    print(line)
print("\nPROBLEMS FOUND:" if problems else "\nNo problems found.")
print("\n".join(problems))
