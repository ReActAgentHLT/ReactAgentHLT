"""Run the strict Direct, CoT, ReAct and Focused ReAct experiments."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter

from reactagenthlt.agents.language_model import TokenUsage
from reactagenthlt.agents.prompts import COT, DIRECT, FORCE_FINISH, REACT, REITERATE
from reactagenthlt.controller import (
    InvalidCalculation,
    execute_calculation,
    format_search_observation,
    retrieve_documents,
)
from reactagenthlt.preprocessing import GSM8K_NUMBER
from reactagenthlt.tools import SearchEngine
from reactagenthlt.tools.search_engine import RetrievalTokenizer


TASK_TOOLS = {"hotpotqa": "Search", "gsm8k": "Calculate"}
STEP_LIMITS = {"hotpotqa": 7, "gsm8k": 9}
BASELINES = {"direct": DIRECT, "cot": COT}
INTERVENTIONS = {
    "react": (False, False),
    "reiteration": (True, False),
    "early_stop": (False, True),
    "focused_react": (True, True),
}

ACTION_LINE = re.compile(r"Action: ([A-Za-z][A-Za-z0-9_]*)\[([^\[\]]*)\]")
FINISH_LINE = re.compile(r"Finish\[([^\[\]]*)\]")
CONTROL_LINE = re.compile(
    r"(?:Action\s*:|Observation\s*:|(?:Search|Calculate|Finish)\s*[\[(])", re.I
)
SEARCH_TOKENIZER = RetrievalTokenizer()


class InvalidOutput(ValueError):
    """A model response that ends an experimental run."""


@dataclass
class StepTrace:
    forced: bool = False
    raw_output: str | None = None
    token_usage: TokenUsage | None = None
    generation_time: float = 0.0
    action: str | None = None
    raw_argument: str | None = None
    argument: str | None = None
    observation: str | None = None
    documents: list[str] = field(default_factory=list)
    repeated: bool = False
    tool_called: bool = False


@dataclass
class RunResult:
    task: str
    mode: str
    answer: str | None = None
    termination: str = "budget_exhausted"
    steps: list[StepTrace] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None


def _parse_experiment_output(text: str, mode: str, forced: bool) -> tuple[str, str]:
    """Read the final command and preserve the exact argument for the trace."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    reasoning = lines[:-1]
    extra_control = any(CONTROL_LINE.match(line) for line in reasoning)
    if mode in BASELINES:
        commands = [line for line in lines if line.startswith("Finish[")]
        match = FINISH_LINE.fullmatch(lines[-1]) if lines else None
        valid_prefix = not reasoning or (mode == "cot" and not extra_control)
    else:
        commands = [line for line in lines if line.startswith("Action:")]
        match = ACTION_LINE.fullmatch(lines[-1]) if lines else None
        valid_prefix = not reasoning or (
            not forced and reasoning[0].startswith("Thought:") and not extra_control
        )

    if len(commands) > 1:
        raise InvalidOutput("multiple_actions")
    if not commands:
        raise InvalidOutput("missing_action")
    if match is None or not valid_prefix:
        raise InvalidOutput("malformed_action")
    return ("Finish", match[1]) if mode in BASELINES else (match[1], match[2])


def _final_answer(argument: str, task: str) -> str:
    if task == "gsm8k":
        if not re.fullmatch(GSM8K_NUMBER, argument):
            raise InvalidOutput("invalid_final_answer")
        return argument.replace(",", "")
    cleaned = argument.strip("<> ").lower()
    if cleaned in {"", "answer", "final answer", "your answer", "number"}:
        raise InvalidOutput("invalid_final_answer")
    return argument


def _read_step(text: str, step: StepTrace, result: RunResult) -> None:
    step.action, step.raw_argument = _parse_experiment_output(
        text, result.mode, step.forced
    )
    step.argument = step.raw_argument.strip()
    if step.action == "Finish":
        result.answer = _final_answer(step.argument, result.task)
    elif step.forced or step.action != TASK_TOOLS[result.task]:
        raise InvalidOutput("unsupported_action")
    elif step.action == "Search":
        step.argument = " ".join(SEARCH_TOKENIZER.tokenize(step.argument))
    if not step.argument:
        raise InvalidOutput("invalid_tool_input")


def run_experiment(
    question: str,
    generate: Callable[[list[dict[str, str]]], tuple[str, TokenUsage]],
    *,
    task: str,
    mode: str = "react",
    search_engine: SearchEngine | None = None,
) -> RunResult:
    """Run one question with a generator that returns the full reply and token usage."""
    if task not in TASK_TOOLS or (mode not in BASELINES and mode not in INTERVENTIONS):
        raise ValueError("Unknown experiment task or mode.")
    question = question.strip()
    if not question:
        raise ValueError("Provide a question.")

    start = perf_counter()
    result = RunResult(task, mode)
    if task == "hotpotqa" and mode in INTERVENTIONS and search_engine is None:
        result.termination = "tool_error"
        result.error = "A shared HotpotQA search engine is required."
        result.elapsed_seconds = perf_counter() - start
        return result

    instructions = BASELINES.get(mode, REACT)[task]
    reiterate, early_stop = INTERVENTIONS.get(mode, (False, False))
    budget = 1 if mode in BASELINES else STEP_LIMITS[task]
    prompt = f"{instructions}\n\nQuestion: {question}\n"
    messages = []
    seen_actions = set()
    force_finish = False
    while len(result.steps) < budget or force_finish:
        if reiterate and not force_finish:
            prompt += "\n" + REITERATE.format(question=question)
        messages.append({"role": "user", "content": prompt})

        step = StepTrace(forced=force_finish)
        result.steps.append(step)
        call_start = perf_counter()
        try:
            text, step.token_usage = generate(messages.copy())
            if not isinstance(text, str):
                raise TypeError("Expected model output as text.")
            step.raw_output = text
        except Exception as error:
            result.termination = "model_error"
            result.error = f"{type(error).__name__}: {error}"
            break
        finally:
            step.generation_time = perf_counter() - call_start

        try:
            _read_step(text, step, result)
        except InvalidOutput as error:
            result.termination = str(error)
            break
        if step.action == "Finish":
            result.termination = "forced_finish" if force_finish else "finished"
            break

        messages.append({"role": "assistant", "content": text})
        action_key = (step.action, step.argument)
        step.repeated = action_key in seen_actions
        if early_stop and step.repeated:
            force_finish = True
            placeholder = "number" if task == "gsm8k" else "answer"
            prompt = FORCE_FINISH.format(question=question, answer=placeholder)
            continue

        step.tool_called = True
        try:
            if step.action == "Search":
                step.documents = retrieve_documents(step.argument, search_engine)
                step.observation = format_search_observation(step.documents)
            else:
                step.observation = execute_calculation(step.argument)
        except InvalidCalculation as error:
            result.termination = "invalid_tool_input"
            result.error = str(error)
            break
        except Exception as error:
            result.termination = "tool_error"
            result.error = f"{type(error).__name__}: {error}"
            break

        seen_actions.add(action_key)
        prompt = f"Observation: {step.observation}\n"

    result.elapsed_seconds = perf_counter() - start
    return result
