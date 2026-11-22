"""Strict action parsing and argument normalization for ReAct agents."""

import re

from reactagenthlt.tools.search_engine import RetrievalTokenizer

BASELINE_MODES = {"direct", "cot"}
ACTION_LINE = re.compile(r"Action: ([A-Za-z][A-Za-z0-9_]*)\[([^\[\]]*)\]")
# The end of the first complete action, spacing tolerated: generation stops
# here and the reply is cut here, so the model cannot write its own
# observation or a second action. The strict parser then checks the format.
ACTION_END = r"Action\s*:\s*[A-Za-z]\w*\s*\[[^\[\]]*\]"
# The end of the first complete standalone Finish line in baseline modes
# (direct / cot), so the model stops immediately after its final answer
# without rambling past it.
FINISH_END = r"(?:^|\n)\s*Finish\s*\[[^\[\]]*\]"
FINISH_LINE = re.compile(r"(Finish)\[([^\[\]]*)\]")
CONTROL_LINE = re.compile(
    r"(?:Action\s*:|Observation\s*:|(?:Search|Calculate|Finish)\s*[\[(])", re.I
)
GROUPED_NUMBER = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?!\d)")
SEARCH_TOKENIZER = RetrievalTokenizer()
PLACEHOLDER_ANSWERS = {"", "answer", "final answer", "your answer", "number"}


class InvalidOutput(ValueError):
    """A model response that ends the run; the message is the termination reason."""


def truncate_after_action(text: str) -> str:
    """Keep everything up to the end of the first complete Action line, if present."""
    end = re.search(ACTION_END, text)
    return text[: end.end()] if end else text


def truncate_after_finish(text: str) -> str:
    """Keep everything up to the end of the first standalone Finish line, if present."""
    end = re.search(FINISH_END, text)
    return text[: end.end()] if end else text


def parse_action(
    model_output: str, mode: str = "react", forced: bool = False
) -> tuple[str, str]:
    """
    Extract an action and its exact argument from the language model's output.

    The command must be the last non-blank line: `Action: Name[argument]` in the
    ReAct modes, `Finish[answer]` in the direct and CoT baselines. Earlier lines
    are allowed only as reasoning (CoT, or a ReAct reply that starts with
    `Thought:`) and may not contain another command or an invented observation.

    Raises:
        InvalidOutput: The reply ends the run; the message is the reason.
    """
    lines = [line.strip() for line in model_output.splitlines() if line.strip()]
    baseline = mode in BASELINE_MODES

    # Expect exactly one command per model response.
    prefix = "Finish[" if baseline else "Action:"
    count = sum(line.startswith(prefix) for line in lines)
    if not count:
        raise InvalidOutput("missing_action")
    if count > 1:
        raise InvalidOutput("multiple_actions")

    # Check the command format and extract its name and argument.
    match = (FINISH_LINE if baseline else ACTION_LINE).fullmatch(lines[-1])
    if not match:
        raise InvalidOutput("malformed_action")

    reasoning = lines[:-1]
    if reasoning:
        allow_reasoning = mode == "cot" or (
            not baseline and not forced and reasoning[0].startswith("Thought:")
        )
        if not allow_reasoning or any(CONTROL_LINE.match(line) for line in reasoning):
            raise InvalidOutput("malformed_action")
    return match[1], match[2]


def validate_final_answer(argument: str) -> str:
    """Reject empty or placeholder final answers."""
    cleaned = argument.strip("<> ").lower()
    if cleaned in PLACEHOLDER_ANSWERS:
        raise InvalidOutput("invalid_final_answer")
    return argument


def normalize_tool_argument(action: str, raw_argument: str) -> str:
    """Normalize a tool argument for deduplication and tool execution."""
    argument = raw_argument.strip()
    if action == "Search":
        argument = " ".join(SEARCH_TOKENIZER.tokenize(argument))
    else:
        argument = GROUPED_NUMBER.sub(
            lambda m: m[0].replace(",", ""), argument
        )
    if not argument:
        raise InvalidOutput("invalid_tool_input")
    return argument
