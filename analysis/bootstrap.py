"""Paired bootstrap between agent modes (CoT vs ReAct, ReAct vs Focused ReAct...).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common import CONFIG, load

PAIRS = [("direct", "cot"), ("cot", "react"), ("react", "focused_react")]
COST = ["seconds", "input_tokens", "output_tokens"]
METRICS = {"hotpotqa": ["f1", "em", "correct"] + COST, "gsm8k": ["correct"] + COST}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int, rng) -> tuple:
    """Mean difference b - a with a 95% confidence interval and a p-value."""
    diff = b - a
    picks = rng.integers(0, len(diff), (n_boot, len(diff)))  # resample questions
    means = diff[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    return diff.mean(), low, high, p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df, rng, rows = load(args.results_dir), np.random.default_rng(args.seed), []
    for config, group in df.groupby(CONFIG):
        for metric in METRICS[config[1]]:
            wide = group.pivot(index="question", columns="mode", values=metric).astype(float)
            for a, b in PAIRS:
                if a not in wide or b not in wide:
                    continue
                pair = wide[[a, b]].dropna()  # questions answered by both modes
                if len(pair) < 2:
                    continue
                diff, low, high, p = paired_bootstrap(pair[a].to_numpy(), pair[b].to_numpy(), args.n_boot, rng)
                rows.append(dict(zip(CONFIG, config)) | {
                    "metric": metric, "A": a, "B": b, "n": len(pair),
                    "mean_A": pair[a].mean(), "mean_B": pair[b].mean(),
                    "diff_B_minus_A": diff, "ci_low": low, "ci_high": high,
                    "p_value": p, "ci_excludes_0": low > 0 or high < 0,
                })
    if not rows:
        raise SystemExit(f"No pairs found. Modes per config:\n{df.groupby(CONFIG)['mode'].unique()}")
    table = pd.DataFrame(rows).round(4)
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "bootstrap.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
