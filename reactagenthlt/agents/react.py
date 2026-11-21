"""ReAct agent built on Omar's ReAct controller.

The loop (`run_react`) and the action parser (`parse_action`) come from
Omar's `controller.py`. They were moved here and changed only where the
agreed experiment rules require it: a strict parser, task-specific tools,
recorded terminations instead of exceptions, and the Focused ReAct
interventions. Every ReAct agent can use every tool and picks the action
that suits its goal; the dataset and the step budget are experiment
settings chosen by the caller. Tools are Federico's `Calculator` and `SearchEngine`, called
directly: their own input checks and errors decide what is invalid.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from time import perf_counter

from reactagenthlt.agents.base import BaseAgent, Response
from reactagenthlt.language_model import LanguageModel, TokenUsage
from reactagenthlt.agents.parsing import (
    ACTION_END,
    InvalidOutput,
    normalize_tool_argument,
    parse_action,
    truncate_after_action,
    validate_final_answer,
)
from reactagenthlt.agents.prompts import COT, DIRECT, FORCE_FINISH, REACT, REITERATE
from reactagenthlt.tools import Calculator, SearchEngine

TOOLS = {"Search", "Calculate"}
TASKS = {"hotpotqa", "gsm8k"}
DEFAULT_MAX_STEPS = 9
BASELINES = {"direct": DIRECT, "cot": COT}
AGENT_MODES = {"react", "focused_react"}

Generate = Callable[[list[dict[str, str]]], tuple[str, TokenUsage]]


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
    task: str | None
    mode: str
    answer: str | None = None
    termination: str = "budget_exhausted"
    steps: list[StepTrace] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None

    @property
    def token_usage(self) -> TokenUsage:
        counts = [
            step.token_usage for step in self.steps if step.token_usage is not None
        ]
        return TokenUsage(
            input_tokens=sum(count.input_tokens for count in counts),
            output_tokens=sum(count.output_tokens for count in counts),
            cached_tokens=sum(count.cached_tokens for count in counts),
        )

    @property
    def generation_time(self) -> float:
        return sum(step.generation_time for step in self.steps)


def _read_step(text: str, step: StepTrace, result: RunResult) -> None:
    step.action, step.raw_argument = parse_action(text, result.mode, step.forced)
    if step.action == "Finish":
        step.argument = step.raw_argument.strip()
        result.answer = validate_final_answer(step.argument)
    elif step.forced or step.action not in TOOLS:
        raise InvalidOutput("unsupported_action")
    else:
        step.argument = normalize_tool_argument(step.action, step.raw_argument)


# The ReAct loop


def _call_model(
    generate: Generate,
    messages: list[dict[str, str]],
    prompt: str,
    result: RunResult,
    forced: bool = False,
) -> StepTrace | None:
    """Send one user turn, record the call and parse the reply; None ends the run."""
    messages.append({"role": "user", "content": prompt})
    step = StepTrace(forced=forced)
    result.steps.append(step)
    call_start = perf_counter()
    try:
        text, step.token_usage = generate(messages.copy())
        if not isinstance(text, str):
            raise TypeError("Expected model output as text.")
        text = step.raw_output = (
            truncate_after_action(text) if result.mode in AGENT_MODES else text
        )
    except Exception as error:
        result.termination = "model_error"
        result.error = f"{type(error).__name__}: {error}"
        return None
    finally:
        step.generation_time = perf_counter() - call_start

    try:
        _read_step(text, step, result)
    except InvalidOutput as error:
        result.termination = str(error)
        return None
    if step.action == "Finish":
        result.termination = "forced_finish" if forced else "finished"
    return step


def run_react(
    question: str,
    generate: Generate,
    *,
    mode: str = "react",
    task: str | None = None,
    search_engine: SearchEngine | None = None,
    calculator: Calculator | None = None,
    max_steps: int | None = None,
    system_prompts: Iterable[dict[str, str]] = (),
) -> RunResult:
    """
    Run the main ReAct reasoning and action loop.

    The controller sends the conversation to a language model, reads the
    generated action, and executes the appropriate tool. The tool result is
    added to the conversation as an observation, and the updated
    conversation is passed back to the model. The process continues until
    the model produces a Finish action or the step budget is spent.

    `direct` and `cot` make a single closed-book call. `focused_react`
    restates the question before every step and, when the model repeats a
    tool action, asks once for the final answer instead of running it again.

    Parameters:
        question: The original question to be answered.
        generate: A function that receives the chat messages and returns
                  the generated text and its token usage.
        mode: "direct", "cot", "react" or "focused_react".
        task: "hotpotqa" or "gsm8k". Required by the direct and CoT
              prompts; for the ReAct modes it is only a label stored in the
              result for the evaluation, and the agent never reads it.
        search_engine: The shared SearchEngine. A Search action without one
                       ends the run as a tool error.
        calculator: An existing Calculator instance (a new one by default).
        max_steps: Maximum number of ordinary model calls (default: 9).
        system_prompts: System messages sent before the question.

    Returns:
        The answer, the termination reason and a trace of every model call.

    Raises:
        ValueError: For an unknown task or mode, a baseline without a task,
                    an empty question or a step limit below 1.
    """

    # Reject invalid settings and empty questions before starting the agent.
    if (task is not None and task not in TASKS) or (
        mode not in BASELINES and mode not in AGENT_MODES
    ):
        raise ValueError("Unknown task or mode.")
    if mode in BASELINES and task is None:
        raise ValueError("The direct and CoT baselines need a task.")
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    # The agent must have at least one step to produce an answer.
    if max_steps is not None and max_steps < 1:
        raise ValueError("Maximum steps must be at least 1.")

    result = RunResult(task, mode)
    start = perf_counter()
    focused = mode == "focused_react"
    budget = 1 if mode in BASELINES else (max_steps or DEFAULT_MAX_STEPS)

    calculator = calculator or Calculator()

    # Start the conversation with the task instructions and the original question.
    messages = list(system_prompts)
    prompt = f"{BASELINES[mode][task] if mode in BASELINES else REACT}\n\nQuestion: {question}\n"
    seen_actions = set()

    # Run the agent for a limited number of steps.
    # The limit prevents the agent from running indefinitely.
    for _ in range(budget):
        if focused:
            prompt += "\n" + REITERATE.format(question=question)

        # Ask the language model to generate its next action.
        step = _call_model(generate, messages, prompt, result)

        # Stop on an invalid reply or model failure, or when the model
        # provides its final answer.
        if step is None or step.action == "Finish":
            break

        # Save the model's response in the conversation history.
        messages.append({"role": "assistant", "content": step.raw_output})
        action_key = (step.action, step.argument)
        step.repeated = action_key in seen_actions

        # Focused ReAct early stop: ask for the final answer instead of
        # repeating the tool. This one extra call is allowed past the budget.
        if focused and step.repeated:
            finish = FORCE_FINISH.format(question=question, answer="answer")
            _call_model(generate, messages, finish, result, forced=True)
            break

        # Execute the requested tool with Federico's tools. A calculator error
        # means the model's expression was invalid, so the run ends as a wrong
        # answer; a search error (including a missing search engine) is an
        # infrastructure failure, reported apart.
        step.tool_called = True
        try:
            if step.action == "Search":
                if search_engine is None:
                    raise RuntimeError("A shared search engine is required for Search.")
                results = search_engine.search(step.raw_argument.strip())
                step.documents = results["document"].tolist()
                step.observation = "\n\n".join(step.documents)
            else:
                step.observation = str(calculator.calculate(step.argument))
        except Exception as error:
            invalid = step.action == "Calculate"
            result.termination = "invalid_tool_input" if invalid else "tool_error"
            result.error = f"{type(error).__name__}: {error}"
            break

        # Add the tool result to the conversation history.
        # The model will receive this observation in the next step.
        seen_actions.add(action_key)
        prompt = f"Observation: {step.observation}\n"

    result.elapsed_seconds = perf_counter() - start
    return result


class ReActAgent(BaseAgent):
    """Answer independent questions with the ReAct loop above.

    `task` is an optional evaluation label (needed only for the direct and CoT
    baselines); it does not change the ReAct prompt, tools or step budget.
    """

    def __init__(
        self,
        language_model: LanguageModel,
        search_engine: SearchEngine | None = None,
        calculator: Calculator | None = None,
        max_steps: int | None = None,
        *,
        task: str | None = None,
    ) -> None:
        if task is not None and task not in TASKS:
            raise ValueError("Unknown experiment task.")
        if max_steps is not None and max_steps < 1:
            raise ValueError("Maximum steps must be at least 1.")
        super().__init__(language_model)
        self.search_engine = search_engine
        self.calculator = calculator if calculator is not None else Calculator()
        self.task = task
        self.max_steps = max_steps or DEFAULT_MAX_STEPS

    def run(
        self,
        question: str,
        *,
        mode: str = "react",
        do_sample: bool = False,
        stream: bool = False,
    ) -> RunResult:
        """Return the full trace for one question and experiment condition."""
        generate = partial(
            self.language_model.generate,
            do_sample=do_sample,
            stream=stream,
            strip_output=False,
            stop_criteria=(ACTION_END,) if mode in AGENT_MODES else (),
        )
        return run_react(
            question,
            generate,
            task=self.task,
            mode=mode,
            search_engine=self.search_engine,
            calculator=self.calculator,
            max_steps=self.max_steps,
            system_prompts=self.system_prompts,
        )

    def ask(
        self,
        prompt: str,
        do_sample: bool = False,
        stream: bool = False,
    ) -> Response:
        """Answer a question with vanilla ReAct and return model-call totals."""
        result = self.run(prompt, do_sample=do_sample, stream=stream)
        return Response(result.answer or "", result.token_usage, result.generation_time)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="filepath to model config")
    parser.add_argument("--sample", action="store_true", help="generate with sampling")
    parser.add_argument("--stream", action="store_true", help="enable chat streaming")
    args = parser.parse_args()

    language_model = LanguageModel(args.config)
    search_engine = SearchEngine(kind="sparse", search_by="document", top_k=2)
    agent = ReActAgent(language_model, search_engine=search_engine)
    while True:
        try:
            prompt = input("Question: ")
            response = agent.ask(prompt, do_sample=args.sample, stream=args.stream)
            if not args.stream:
                print(response.output)

            print()
            print(f"input tokens: {response.token_usage.input_tokens}")
            print(f"output tokens: {response.token_usage.output_tokens}")
            print(f"generation time: {response.generation_time:.2f} seconds")
            print("=" * 50)
        except EOFError:
            break
