"""Score completed runs and inspect invalid replies after the experiment."""

import re
import string
from collections import Counter
from collections.abc import Iterable
from decimal import Decimal

from reactagenthlt.agents.react import BASELINES, RunResult
from reactagenthlt.preprocessing import GSM8K_NUMBER


FORMAT_FAILURES = {"missing_action", "malformed_action", "multiple_actions"}
INVALID_REPLIES = FORMAT_FAILURES | {
    "unsupported_action", "invalid_tool_input", "invalid_final_answer"
}
INFRASTRUCTURE_FAILURES = {"model_error", "tool_error"}
LOOSE_COMMAND = re.compile(
    r"(?:(action)\s*(:?)\s*)?(search|calculate|finish)\b\s*([\[(]?)(.*)", re.I
)


def _format_deviations(text: str, mode: str) -> set[str]:
    """Recognize deviations in command-like lines for reporting only."""
    categories = set()
    commands = 0
    needs_prefix = mode not in BASELINES
    for line in text.splitlines():
        line = line.strip()
        command = line.strip("`*").strip()
        command = re.sub(r"^[-*]\s+", "", command)
        match = LOOSE_COMMAND.fullmatch(command)
        if not match or not (match[1] or match[4]):
            continue

        commands += 1
        prefix, colon, name, opening, remainder = match.groups()
        if command != line or text.strip().startswith("```"):
            categories.add("markdown_wrapper")
        if needs_prefix and not prefix:
            categories.add("missing_action_prefix")
        elif not needs_prefix and prefix:
            categories.add("unexpected_action_prefix")
        elif prefix and not colon:
            categories.add("missing_colon")
        if (prefix and prefix != "Action") or name != name.capitalize():
            categories.add("capitalization")
        if opening == "[" and bool(prefix) == needs_prefix and (not prefix or colon):
            canonical = f"{'Action: ' if needs_prefix else ''}{name}["
            if command[:match.end(4)].lower() != canonical.lower():
                categories.add("spacing")

        if not opening:
            categories.add("missing_opening_bracket")
        elif opening == "(":
            categories.add("parentheses_instead_of_brackets")
        else:
            argument, closing, trailing = remainder.rpartition("]")
            if not closing:
                categories.add("missing_closing_bracket")
            elif not argument.strip():
                categories.add("empty_argument")
            if closing and trailing.strip():
                categories.add("trailing_text")

    if commands > 1:
        categories.add("multiple_commands")
    if commands and not categories:
        categories.add("unexpected_text")
    return categories


def diagnose_formats(results: Iterable[RunResult]) -> dict:
    """Count deviations by type and count each affected model call once."""
    counts = Counter()
    affected_calls = 0
    for result in results:
        if result.termination not in INVALID_REPLIES or not result.steps:
            continue
        text = result.steps[-1].raw_output or ""
        categories = _format_deviations(text, result.mode)
        if result.termination not in FORMAT_FAILURES:
            categories &= {"empty_argument", "capitalization"}
        if categories:
            counts.update(categories)
            affected_calls += 1
    return {"affected_calls": affected_calls, "categories": dict(sorted(counts.items()))}


def summarize_run(result: RunResult, supporting_documents: set[str] | None = None) -> dict:
    """Collect the counts and times for one result, including failed runs."""
    tokens = result.token_usage
    summary = {
        "termination": result.termination,
        "model_calls": len(result.steps),
        "ordinary_calls": sum(not step.forced for step in result.steps),
        "tool_calls": sum(step.tool_called for step in result.steps),
        "prompt_tokens": tokens.input_tokens,
        "completion_tokens": tokens.output_tokens,
        "generation_seconds": result.generation_time,
        "elapsed_seconds": result.elapsed_seconds,
        "invalid_reply_calls": int(result.termination in INVALID_REPLIES),
        "repeated_actions": sum(step.repeated for step in result.steps),
        "early_stop_triggered": any(step.forced for step in result.steps),
        "budget_exhausted": result.termination == "budget_exhausted",
        "calculator_calls": sum(
            step.action == "Calculate" and step.tool_called for step in result.steps
        ),
        "invalid_calculations": int(
            result.termination == "invalid_tool_input"
            and result.steps[-1].action == "Calculate"
        ),
    }
    if supporting_documents is not None:
        retrieved = {document for step in result.steps for document in step.documents}
        summary["supporting_paragraphs_retrieved"] = len(retrieved & supporting_documents)
        summary["all_supporting_paragraphs_retrieved"] = bool(supporting_documents) and (
            supporting_documents <= retrieved
        )
    return summary


def _normalize_hotpotqa(answer: str) -> str:
    """Use HotpotQA's official answer normalization rules."""
    cleaned = answer.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", cleaned).split())


def score_answer(result: RunResult, reference: str) -> dict[str, float] | None:
    """Score a prediction, with None reserved for infrastructure failures."""
    if result.termination in INFRASTRUCTURE_FAILURES:
        return None
    if result.task == "gsm8k":
        answer = (result.answer or "").strip()
        correct = bool(re.fullmatch(GSM8K_NUMBER, answer)) and (
            Decimal(answer.replace(",", "")) == Decimal(reference.replace(",", ""))
        )
        return {"accuracy": float(correct)}
    if result.answer is None:
        return {"em": 0.0, "f1": 0.0}
    prediction = _normalize_hotpotqa(result.answer)
    gold = _normalize_hotpotqa(reference)
    em = float(prediction == gold)
    if not em and {prediction, gold} & {"yes", "no", "noanswer"}:
        return {"em": em, "f1": 0.0}
    pred_tokens, gold_tokens = prediction.split(), gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    f1 = 2 * common.total() / (len(pred_tokens) + len(gold_tokens)) if common else 0.0
    return {"em": em, "f1": f1}
