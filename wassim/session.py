"""ReAct-style controller: stream, pause for a tool, append the observation, resume."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .engine import ModelEngine, RunMetrics, StepResult, fallback_answer, parse_action, parse_finish
from .prompts import (
    cot_prompt,
    direct_prompt,
    focused_step_prompt,
    forced_finish_prompt,
    react_prompt,
)

MAX_NEW_TOKENS = {
    "direct": 64,
    "cot": 512,
    "react": 160,
    "focused_react": 160,
    "forced": 128,
}


@dataclass
class RunResult:
    answer: str
    metrics: RunMetrics
    trajectory: str = ""


@dataclass
class RunTracker:
    metrics: RunMetrics = field(default_factory=RunMetrics)

    def add_step(self, step: StepResult) -> None:
        if self.metrics.num_steps == 0:
            self.metrics.ttft_seconds = step.metrics.ttft_seconds
        self.metrics.prompt_tokens += step.metrics.prompt_tokens
        self.metrics.completion_tokens += step.metrics.completion_tokens
        self.metrics.total_tokens += step.metrics.total_tokens
        self.metrics.inference_seconds += step.metrics.inference_seconds
        self.metrics.num_steps += 1

    def add_tool_call(self, tool: str, argument: str, observation: str) -> None:
        self.metrics.tool_calls.append(
            {"tool": tool, "argument": argument, "observation": observation}
        )

    def finish(self, reason: str) -> None:
        self.metrics.stop_reason = reason
        if self.metrics.inference_seconds > 0:
            self.metrics.tokens_per_second = (
                self.metrics.completion_tokens / self.metrics.inference_seconds
            )


class AgentSession:
    """One question through one mode, with manual or automatic tool observations."""

    def __init__(
        self,
        engine: ModelEngine,
        task: str,
        mode: str,
        *,
        max_steps: int = 7,
        max_new_tokens: int | None = None,
        on_token: Callable[[str], None] | None = None,
        observation_provider: Callable[[str, str], str] | None = None,
    ):
        if task not in ("hotpotqa", "gsm8k"):
            raise ValueError("task must be 'hotpotqa' or 'gsm8k'")
        if mode not in ("direct", "cot", "react", "focused_react"):
            raise ValueError("mode must be direct, cot, react, or focused_react")
        self.engine = engine
        self.task = task
        self.mode = mode
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens or MAX_NEW_TOKENS[mode]
        self.on_token = on_token
        self.observation_provider = observation_provider

    @property
    def uses_tools(self) -> bool:
        """Direct and CoT are closed-book; ReAct and Focused ReAct use a tool."""
        return self.mode in ("react", "focused_react")

    def allowed_tools(self) -> set[str]:
        """The only action names this task allows."""
        return {"Search", "Finish"} if self.task == "hotpotqa" else {"Calculate", "Finish"}

    def run(self, question: str) -> RunResult:
        tracker = RunTracker()
        history = ""
        seen_actions: set[tuple[str, str]] = set()
        invalid_actions = 0

        for _ in range(self.max_steps):
            prompt = self._prompt(question, history)
            stop_policy = "finish" if self.mode in ("direct", "cot") else "action"
            step = self.engine.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                stop_policy=stop_policy,
                on_token=self.on_token,
            )
            tracker.add_step(step)
            text = step.text.strip()

            if self.mode in ("direct", "cot"):
                answer = parse_finish(text)
                if answer is not None:
                    reason = "finished"
                    final_answer = answer
                elif step.metrics.stop_reason == "max_tokens":
                    reason = "max_tokens"
                    final_answer = ""
                else:
                    reason = "parse_error"
                    final_answer = fallback_answer(text, self.task)
                tracker.finish(reason)
                return RunResult(
                    answer=final_answer,
                    metrics=tracker.metrics,
                    trajectory=(history + text).strip(),
                )

            action = parse_action(text)
            if action is None:
                invalid_actions += 1
                observation = self._invalid_message()
                history += f"{text}\nObservation: {observation}\n"
                if self.mode == "focused_react" and invalid_actions >= 2:
                    return self._forced_finish(question, history, tracker)
                continue

            tool, argument = action
            if tool == "Finish":
                answer = parse_finish(text)
                if answer is None:
                    invalid_actions += 1
                    observation = "Invalid answer. Put a real value inside Finish[...]."
                    history += f"{text}\nObservation: {observation}\n"
                    if self.mode == "focused_react" and invalid_actions >= 2:
                        return self._forced_finish(question, history, tracker)
                    continue
                tracker.finish("finished")
                return RunResult(
                    answer=answer,
                    metrics=tracker.metrics,
                    trajectory=(history + text).strip(),
                )

            if tool not in self.allowed_tools():
                invalid_actions += 1
                observation = self._invalid_message()
                history += f"{text}\nObservation: {observation}\n"
                if self.mode == "focused_react" and invalid_actions >= 2:
                    return self._forced_finish(question, history, tracker)
                continue

            invalid_actions = 0
            key = (tool, " ".join(argument.lower().split()))
            if key in seen_actions:
                if self.mode == "focused_react":
                    return self._forced_finish(question, history + text, tracker)
                tracker.finish("repeated_action")
                return RunResult(
                    answer="",
                    metrics=tracker.metrics,
                    trajectory=(history + text).strip(),
                )
            seen_actions.add(key)

            if self.observation_provider is None:
                tracker.finish("error")
                return RunResult(
                    answer="",
                    metrics=tracker.metrics,
                    trajectory=(history + text).strip(),
                )

            observation = self.observation_provider(tool, argument)
            if observation is None:
                observation = "(no observation)"
            tracker.add_tool_call(tool, argument, observation)
            history += f"{text}\nObservation: {observation}\n"

        if self.mode == "focused_react":
            return self._forced_finish(question, history, tracker)
        tracker.finish("max_steps")
        return RunResult(answer="", metrics=tracker.metrics, trajectory=history.strip())

    def _prompt(self, question: str, history: str) -> str:
        if self.mode == "direct":
            return direct_prompt(self.task, question)
        if self.mode == "cot":
            return cot_prompt(self.task, question)
        if self.mode == "focused_react":
            return focused_step_prompt(self.task, question, history)
        return react_prompt(self.task, question, history)

    def _forced_finish(
        self, question: str, history: str, tracker: RunTracker
    ) -> RunResult:
        prompt = forced_finish_prompt(self.task, question, history)
        step = self.engine.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS["forced"],
            stop_policy="finish",
            on_token=self.on_token,
        )
        tracker.add_step(step)
        answer = parse_finish(step.text)
        tracker.finish("forced_finish" if answer is not None else "parse_error")
        return RunResult(
            answer=answer or fallback_answer(step.text, self.task),
            metrics=tracker.metrics,
            trajectory=(history + step.text).strip(),
        )

    def _invalid_message(self) -> str:
        syntax = "Search[...]" if self.task == "hotpotqa" else "Calculate[...]"
        return f"Invalid action. Use {syntax} or Finish[...]."
