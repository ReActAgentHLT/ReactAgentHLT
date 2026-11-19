"""Unified streaming inference engine with per-step and per-run metrics."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .backends import HFBackend, LlamaCppBackend, detect_backend

ACTION_RE = re.compile(r"Action:\s*(Search|Calculate|Finish)\[(.*?)\]", re.S)
FINISH_RE = re.compile(r"Finish\[(.*?)\]", re.S)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
PLACEHOLDERS = {"answer", "final answer", "your answer", "number", "<answer>", "<final answer>"}

STOP_STRINGS = ("\nObservation:", "Observation:")


@dataclass
class StepMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_seconds: float = 0.0
    inference_seconds: float = 0.0
    tokens_per_second: float = 0.0
    stop_reason: str = "finished"


@dataclass
class StepResult:
    text: str
    metrics: StepMetrics


@dataclass
class RunMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_seconds: float = 0.0
    inference_seconds: float = 0.0
    tokens_per_second: float = 0.0
    num_steps: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: str = ""


def parse_action(text: str) -> tuple[str, str] | None:
    """Return (tool, argument) from the first complete Action[...] in text."""
    match = ACTION_RE.search(text)
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def parse_finish(text: str) -> str | None:
    """Return the contents of the first Finish[...] in text."""
    match = FINISH_RE.search(text)
    if not match:
        return None
    answer = match.group(1).strip()
    if answer.startswith("<") and answer.endswith(">"):
        answer = answer[1:-1].strip()
    return None if answer.lower() in PLACEHOLDERS else answer or None


def fallback_answer(text: str, task: str) -> str:
    """Last number for GSM8K, last non-empty line for HotPotQA."""
    if task == "gsm8k":
        numbers = NUMBER_RE.findall(text.replace(",", "").replace("$", ""))
        return numbers[-1] if numbers else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def has_action(text: str) -> bool:
    return ACTION_RE.search(text) is not None


def has_finish(text: str) -> bool:
    return FINISH_RE.search(text) is not None


class ModelEngine:
    """Load one model, stream text, and return metrics for every call."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 8192,
        device_map: str = "auto",
        n_gpu_layers: int = -1,
    ):
        kind = detect_backend(model_path)
        if kind == "gguf":
            self.backend = LlamaCppBackend(
                model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers
            )
        else:
            self.backend = HFBackend(model_path, device_map=device_map)
        self.model_path = Path(model_path)

    def info(self) -> dict:
        return self.backend.info()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        stop_policy: str = "finish",
        on_token: Callable[[str], None] | None = None,
    ) -> StepResult:
        """Stream one completion.

        stop_policy:
          'finish' -> stop at the first complete Finish[...] (Direct / CoT)
          'action' -> stop at the first complete Action[...] (ReAct / Focused)
          'none'   -> stop only at EOS or max_new_tokens
        """
        if stop_policy not in ("finish", "action", "none"):
            raise ValueError("stop_policy must be 'finish', 'action', or 'none'")

        rendered = self.backend.render(prompt)
        prompt_tokens = (
            self.backend.count_prompt_tokens(rendered)
            if hasattr(self.backend, "count_prompt_tokens")
            else self.backend.count_tokens(rendered)
        )

        def should_stop(text: str) -> bool:
            if "Observation:" in text:
                return True
            if stop_policy == "finish":
                return has_finish(text)
            if stop_policy == "action":
                return has_action(text)
            return False

        start = time.perf_counter()
        first_token_at: float | None = None
        text = ""
        for piece in self.backend.stream(
            rendered, max_new_tokens, STOP_STRINGS, should_stop
        ):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            text += piece
            if on_token:
                on_token(piece)
            if should_stop(text):
                break
        end = time.perf_counter()

        fake_observation = "Observation:" in text
        if fake_observation:
            text = text.split("Observation:", 1)[0].rstrip()

        completion_tokens = self.backend.last_completion_tokens or self.backend.count_tokens(text)
        total_tokens = prompt_tokens + completion_tokens
        inference_seconds = end - start
        ttft_seconds = (first_token_at - start) if first_token_at else 0.0
        tokens_per_second = (
            completion_tokens / inference_seconds if inference_seconds > 0 else 0.0
        )

        if stop_policy == "finish" and has_finish(text):
            stop_reason = "finished"
        elif stop_policy == "action" and has_action(text):
            stop_reason = "action"
        elif fake_observation:
            stop_reason = "stop_sequence"
        elif completion_tokens >= max_new_tokens:
            stop_reason = "max_tokens"
        else:
            stop_reason = "eos"

        return StepResult(
            text=text.strip(),
            metrics=StepMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                ttft_seconds=ttft_seconds,
                inference_seconds=inference_seconds,
                tokens_per_second=tokens_per_second,
                stop_reason=stop_reason,
            ),
        )

    def unload(self) -> None:
        self.backend.unload()
