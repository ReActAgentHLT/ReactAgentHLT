"""Run an agent over a dataset and collect per-question metrics in a DataFrame."""

import re
import string
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from reactagenthlt.agents import LanguageModel, ReActAgent
from reactagenthlt.agents.react import RunResult
from reactagenthlt.preprocessing import hotpotqa_supporting_documents, load_gsm8k, load_hotpotqa
from reactagenthlt.tools import SearchEngine


# Replies that end the run because the action is missing or invalid
# (bad syntax, unknown action, invalid tool input or placeholder answer).
FAILED_ACTIONS = {
    "missing_action", "malformed_action", "multiple_actions",
    "unsupported_action", "invalid_tool_input", "invalid_final_answer",
}
CONFIG = ["model", "dataset", "mode", "k", "budget"]
# A HotpotQA answer counts as correct (for `actions_correct`) from this F1 on.
F1_THRESHOLD = 0.5
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


# Answer scoring


def _normalize_hotpotqa(answer: str) -> str:
    """HotpotQA's official normalization: lowercase, no punctuation or articles."""
    cleaned = answer.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", cleaned).split())


def hotpotqa_scores(prediction: str | None, gold: str) -> dict[str, float]:
    """EM, precision, recall and F1 as in the official HotpotQA evaluation."""
    pred, ref = _normalize_hotpotqa(prediction or ""), _normalize_hotpotqa(gold)
    scores = {"em": float(pred == ref), "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if pred != ref and {pred, ref} & {"yes", "no", "noanswer"}:
        return scores
    common = (Counter(pred.split()) & Counter(ref.split())).total()
    if common:
        precision, recall = common / len(pred.split()), common / len(ref.split())
        scores.update(
            precision=precision, recall=recall,
            f1=2 * precision * recall / (precision + recall),
        )
    return scores


def gsm8k_number(text: str | None) -> Decimal | None:
    """Keep only the last number of the text, after removing thousands commas."""
    numbers = NUMBER.findall(THOUSANDS_COMMA.sub("", text or ""))
    return Decimal(numbers[-1]) if numbers else None


def gsm8k_correct(prediction: str | None, gold: str) -> bool:
    """Numerical match between the model answer and the gold answer."""
    number = gsm8k_number(prediction)
    return number is not None and number == gsm8k_number(gold)


# Per-question record


def score_run(result: RunResult, row: pd.Series, dataset: str) -> dict:
    """Metrics of one run; `row` is the dataset row with the gold answer."""
    tokens = result.token_usage
    record = {
        "question": row["question"],
        "gold": row["answer"],
        "prediction": result.answer,
        "termination": result.termination,
        "actions": len(result.steps),
        "failed_action": result.termination in FAILED_ACTIONS,
        "seconds": result.elapsed_seconds,
        # Only the input computed at each call: the KV-cached prefix is not counted.
        "input_tokens": tokens.input_tokens - tokens.cached_tokens,
        "output_tokens": tokens.output_tokens,
    }
    if dataset == "hotpotqa":
        record.update(hotpotqa_scores(result.answer, row["answer"]))
        record["correct"] = record["f1"] >= F1_THRESHOLD
        support = hotpotqa_supporting_documents(row)
        searches = [
            set(step.documents) for step in result.steps
            if step.action == "Search" and step.tool_called
        ]
        record["searches"] = len(searches)
        record["gold_searches"] = sum(bool(docs & support) for docs in searches)
    else:
        record["correct"] = gsm8k_correct(result.answer, row["answer"])
    return record


# Evaluation loop


def evaluate(
    model: LanguageModel,
    dataset: str,
    k: int,
    budget: int,
    *,
    mode: str = "react",
    n: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Run the agent on `dataset` and return one row per question.

    Parameters:
        model: A loaded language model (shared across evaluations).
        dataset: "hotpotqa" (distractor validation) or "gsm8k" (test).
        k: Number of paragraphs returned by each Search action.
        budget: Maximum number of ordinary model calls per question.
        mode: "react", "focused_react", "direct" or "cot".
        n: Evaluate a random sample of `n` questions (all if None).
        seed: Random seed of the sample.
    """
    if dataset == "hotpotqa":
        data = load_hotpotqa(include_context=True)
    elif dataset == "gsm8k":
        data = load_gsm8k()
    else:
        raise ValueError(f"Unknown dataset {dataset!r}: use 'hotpotqa' or 'gsm8k'.")
    if n is not None:
        data = data.sample(n, random_state=seed)

    search_engine = SearchEngine(kind="sparse", search_by="document", top_k=k)
    agent = ReActAgent(model, search_engine, max_steps=budget, task=dataset)
    config = dict(zip(CONFIG, (model.config["model"], dataset, mode, k, budget)))

    records = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc=f"{dataset} {mode}", ncols=80):
        result = agent.run(row["question"], mode=mode)
        records.append(config | score_run(result, row, dataset))
    return pd.DataFrame(records)


def save_results(df: pd.DataFrame, directory: str | Path = RESULTS_DIR) -> Path:
    """
    Save one `evaluate` run as CSV named by date, time and configuration, e.g.
    `2026-11-21_15-30_Qwen2.5-3B-Instruct-4bit_gsm8k_focused_react_k2_b9.csv`.
    """
    run = df.iloc[0]
    name = "_".join([
        datetime.now().strftime("%Y-%m-%d_%H-%M"), run["model"].split("/")[-1],
        run["dataset"], run["mode"], f"k{run['k']}", f"b{run['budget']}",
    ])
    path = Path(directory) / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the rows of `evaluate` (or several concatenated) per configuration."""
    summaries = []
    for values, runs in df.groupby(CONFIG, sort=False):
        summary = dict(zip(CONFIG, values))
        summary.update(
            questions=len(runs),
            failed_action_pct=100 * runs["failed_action"].sum() / runs["actions"].sum(),
            seconds=runs["seconds"].mean(),
            input_tokens=runs["input_tokens"].mean(),
            output_tokens=runs["output_tokens"].mean(),
            actions_all=runs["actions"].mean(),
            actions_correct=runs.loc[runs["correct"].astype(bool), "actions"].mean(),
        )
        if summary["dataset"] == "hotpotqa":
            summary.update({m: 100 * runs[m].mean() for m in ("em", "precision", "recall", "f1")})
            searches = runs["searches"].sum()
            summary["gold_search_pct"] = (
                100 * runs["gold_searches"].sum() / searches if searches else float("nan")
            )
        else:
            summary["accuracy"] = 100 * runs["correct"].mean()
        summaries.append(summary)
    return pd.DataFrame(summaries)