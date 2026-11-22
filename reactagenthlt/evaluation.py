"""Run an agent over a dataset and collect per-question metrics in a DataFrame."""

import argparse
import re
import string
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from itertools import count
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from reactagenthlt.agents import ReActAgent
from reactagenthlt.agents.react import AGENT_MODES, BASELINES, RunResult
from reactagenthlt.language_model import LanguageModel
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
DEFAULT_BUDGETS = {"hotpotqa": 7, "gsm8k": 9}
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
    verbose: bool = False,
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
        verbose: Print one status line per question with running metrics.
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
    total = len(data)
    correct_sum = 0
    f1_sum = 0.0
    em_sum = 0.0
    failed_sum = 0
    actions_sum = 0

    bar = tqdm(data.iterrows(), total=total, desc=f"{dataset} {mode}")
    for done, (_, row) in enumerate(bar, start=1):
        result = agent.run(row["question"], mode=mode)
        scored = score_run(result, row, dataset)
        records.append(config | scored)

        correct_sum += int(scored["correct"])
        failed_sum += int(scored["failed_action"])
        actions_sum += int(scored["actions"])
        status = "OK" if scored["correct"] else "WRONG"
        fail_pct = 100 * failed_sum / actions_sum if actions_sum else 0.0

        if dataset == "hotpotqa":
            f1_sum += float(scored["f1"])
            em_sum += float(scored["em"])
            avg_f1 = 100 * f1_sum / done
            avg_em = 100 * em_sum / done
            bar.set_postfix_str(
                f"last={status} | F1={avg_f1:.1f}% EM={avg_em:.1f}% | fail={fail_pct:.1f}%"
            )
            if verbose:
                tqdm.write(
                    f"[{done}/{total}] {status:<5} | f1={scored['f1']:.2f} em={int(scored['em'])} "
                    f"| term={scored['termination']} act={scored['actions']} "
                    f"| pred={scored['prediction']!r} gold={scored['gold']!r} "
                    f"| running F1={avg_f1:.1f}% EM={avg_em:.1f}%"
                )
        else:
            acc = 100 * correct_sum / done
            bar.set_postfix_str(
                f"last={status} | acc={acc:.1f}% ({correct_sum}/{done}) | fail={fail_pct:.1f}%"
            )
            if verbose:
                tqdm.write(
                    f"[{done}/{total}] {status:<5} | term={scored['termination']} act={scored['actions']} "
                    f"| pred={scored['prediction']!r} gold={scored['gold']!r} "
                    f"| running acc={acc:.1f}% ({correct_sum}/{done})"
                )
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
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index in count():
        suffix = f"_{index}" if index else ""
        path = directory / f"{name}{suffix}.csv"
        try:
            output = path.open("x", encoding="utf-8", newline="")
        except FileExistsError:
            continue
        try:
            with output:
                df.to_csv(output, index=False)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
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


PERCENT_COLUMNS = {
    "failed_action_pct",
    "accuracy",
    "em",
    "precision",
    "recall",
    "f1",
    "gold_search_pct",
}
HEADER_LABELS = {
    "failed_action_pct": "failed_action (%)",
    "accuracy": "accuracy (%)",
    "em": "em (%)",
    "precision": "precision (%)",
    "recall": "recall (%)",
    "f1": "f1 (%)",
    "gold_search_pct": "gold_search (%)",
}


def _format_cell(column: str, value) -> str:
    """Format one summary table cell cleanly for PNG export."""
    if pd.isna(value):
        return "-"
    if column in {"k", "budget", "questions"}:
        return str(int(value))
    if isinstance(value, (int, float)):
        formatted = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{formatted}%" if column in PERCENT_COLUMNS else formatted
    return str(value)


def _load_table_fonts(size: int = 16):
    """Load clean TrueType fonts when available, falling back to Pillow's default."""
    from PIL import ImageFont

    candidates = [
        ("consolab.ttf", "consola.ttf"),
        ("segoeuib.ttf", "segoeui.ttf"),
        ("DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf"),
        ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"),
    ]
    for bold_name, regular_name in candidates:
        try:
            title_font = ImageFont.truetype(bold_name, size + 4)
            bold_font = ImageFont.truetype(bold_name, size)
            regular_font = ImageFont.truetype(regular_name, size)
            return title_font, bold_font, regular_font
        except OSError:
            continue
    default = ImageFont.load_default()
    return default, default, default


def save_summary_png(
    summary: pd.DataFrame,
    path: str | Path = RESULTS_DIR / "summary.png",
) -> Path:
    """Render a `summarize()` DataFrame into a crisp PNG table for sharing."""
    from PIL import Image, ImageDraw

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode_rank = {"direct": 0, "cot": 1, "react": 2, "focused_react": 3}
    title_font, bold_font, regular_font = _load_table_fonts(size=15)

    # Split by dataset so HotpotQA and GSM8K each display only their relevant metrics.
    sections: list[tuple[str, list[str], list[list[str]], list[str]]] = []
    for dataset, group in summary.groupby("dataset", sort=True):
        ordered = group.assign(
            _mode_rank=group["mode"].map(lambda m: mode_rank.get(m, 99))
        ).sort_values(["model", "k", "budget", "_mode_rank", "mode"])
        ordered = ordered.drop(columns=["_mode_rank"])

        columns = [
            col for col in ordered.columns
            if col != "dataset" and ordered[col].notna().any()
        ]
        rows = [
            [_format_cell(col, row[col]) for col in columns]
            for _, row in ordered.iterrows()
        ]
        models = ordered["model"].tolist()
        sections.append((str(dataset).upper(), columns, rows, models))

    pad_x, pad_y = 14, 9
    outer_pad = 24
    section_gap = 28
    title_height = 34
    header_height = 36
    row_height = 32

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def text_width(text: str, font) -> int:
        box = measure.textbbox((0, 0), text, font=font)
        return int(box[2] - box[0])

    # Compute per-section column widths and overall canvas size.
    section_widths: list[list[int]] = []
    total_width = 0
    total_height = outer_pad
    for _, columns, rows, _ in sections:
        widths = []
        for idx, col in enumerate(columns):
            w = text_width(HEADER_LABELS.get(col, col), bold_font)
            for row in rows:
                w = max(w, text_width(row[idx], regular_font))
            widths.append(w + 2 * pad_x)
        section_widths.append(widths)
        total_width = max(total_width, sum(widths) + 2 * outer_pad)
        total_height += title_height + header_height + len(rows) * row_height + section_gap
    total_height = total_height - section_gap + outer_pad

    img = Image.new("RGB", (total_width, total_height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    text_cols = {"model", "mode"}
    y = outer_pad
    for (title, columns, rows, models), widths in zip(sections, section_widths):
        table_w = sum(widths)
        x0 = outer_pad

        # Section title
        draw.text((x0, y + 4), f"{title} Evaluation Summary", fill="#111827", font=title_font)
        y += title_height

        # Header row
        draw.rectangle([x0, y, x0 + table_w, y + header_height], fill="#1F2937")
        x = x0
        for col, w in zip(columns, widths):
            label = HEADER_LABELS.get(col, col)
            tw = text_width(label, bold_font)
            tx = x + pad_x if col in text_cols else x + w - pad_x - tw
            draw.text((tx, y + pad_y), label, fill="#FFFFFF", font=bold_font)
            x += w
        y += header_height

        # Data rows
        prev_model = None
        for row_idx, (row, model_name) in enumerate(zip(rows, models)):
            bg = "#F9FAFB" if row_idx % 2 else "#FFFFFF"
            draw.rectangle([x0, y, x0 + table_w, y + row_height], fill=bg)
            x = x0
            for col, val, w in zip(columns, row, widths):
                tw = text_width(val, regular_font)
                tx = x + pad_x if col in text_cols else x + w - pad_x - tw
                draw.text((tx, y + pad_y - 1), val, fill="#111827", font=regular_font)
                draw.line([(x, y), (x, y + row_height)], fill="#E5E7EB", width=1)
                x += w
            draw.line([(x0 + table_w, y), (x0 + table_w, y + row_height)], fill="#E5E7EB", width=1)

            # Stronger horizontal divider when switching models
            border_color = "#9CA3AF" if (prev_model is not None and model_name != prev_model) else "#E5E7EB"
            border_width = 2 if (prev_model is not None and model_name != prev_model) else 1
            draw.line([(x0, y), (x0 + table_w, y)], fill=border_color, width=border_width)
            prev_model = model_name
            y += row_height

        # Outer border around table
        top_y = y - len(rows) * row_height - header_height
        draw.rectangle([x0, top_y, x0 + table_w, y], outline="#9CA3AF", width=1)
        y += section_gap

    tmp = path.with_suffix(".tmp.png")
    img.save(tmp, format="PNG")
    tmp.replace(path)
    return path


def save_summary(
    summary: pd.DataFrame,
    directory: str | Path = RESULTS_DIR,
    stem: str = "summary",
) -> tuple[Path, Path]:
    """Save an aggregated summary DataFrame as both CSV and PNG."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{stem}.csv"
    tmp_csv = csv_path.with_suffix(".tmp")
    summary.to_csv(tmp_csv, index=False)
    tmp_csv.replace(csv_path)
    png_path = save_summary_png(summary, directory / f"{stem}.png")
    return csv_path, png_path


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main(
    argv: list[str] | None = None,
    *,
    load_model: Callable[[str], LanguageModel] = LanguageModel,
    model_choices: tuple[str, ...] | None = None,
    output_dir: Path = RESULTS_DIR,
) -> None:
    """Run one evaluation from the terminal and save its per-question results."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "model",
        choices=model_choices,
        help="local model to run" if model_choices else "path to a model config JSON",
    )
    parser.add_argument("--dataset", required=True, choices=DEFAULT_BUDGETS)
    parser.add_argument("--mode", choices=sorted(AGENT_MODES | BASELINES.keys()), default="react")
    parser.add_argument("-k", "--k", type=positive_int, default=3, help="paragraphs per search (default: 3)")
    parser.add_argument("--budget", type=positive_int, help="action budget (default: 7 for HotpotQA, 9 for GSM8K)")
    parser.add_argument("--n", type=positive_int, help="sample size (default: evaluate the full split)")
    parser.add_argument("--seed", type=int, default=0, help="question sampling seed (default: 0)")
    parser.add_argument("--output-dir", type=Path, default=output_dir, help="directory for per-question CSVs")
    parser.add_argument("--verbose", action="store_true", help="print per-question outcome and running metrics")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    if model_choices is None and not Path(args.model).is_file():
        parser.error(f"model config file does not exist: {args.model}")

    budget = args.budget if args.budget is not None else DEFAULT_BUDGETS[args.dataset]
    model = load_model(args.model)
    frame = evaluate(
        model, args.dataset, args.k, budget,
        mode=args.mode, n=args.n, seed=args.seed, verbose=args.verbose,
    )
    path = save_results(frame, args.output_dir)
    summary = summarize(frame)
    csv_path, png_path = save_summary(summary, args.output_dir, stem=f"{path.stem}_summary")
    print(summary.round(3).to_string(index=False))
    print(f"Saved {len(frame)} results to {path}")
    print(f"Saved summary to {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
