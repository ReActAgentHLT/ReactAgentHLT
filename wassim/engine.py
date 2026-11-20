"""General streaming text generation.

The engine knows nothing about ReAct, actions, tools, tasks, or datasets.
It returns only the generated text and the number of completion tokens.
The caller owns prompts, stop rules, history, tools, timing, and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .backends import ChatInput, HFBackend, LlamaCppBackend, detect_backend


@dataclass
class Generation:
    text: str
    completion_tokens: int


class ModelEngine:
    """Load one model and stream a completion for a prompt or chat messages."""

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
                model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
            )
        else:
            self.backend = HFBackend(model_path, device_map=device_map)

    def info(self) -> dict:
        return self.backend.info()

    def generate(
        self,
        prompt: ChatInput,
        *,
        max_new_tokens: int = 256,
        stop_strings: tuple[str, ...] = (),
        should_stop: Callable[[str], bool] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> Generation:
        """Stream one completion and return its text and completion tokens.

        The caller decides when to stop by passing stop strings and/or a
        should_stop(text) callback. The engine does not interpret the text.
        """
        rendered = self.backend.render(prompt)
        stop = should_stop or (lambda _text: False)

        text = ""
        for piece in self.backend.stream(
            rendered,
            max_new_tokens,
            stop_strings,
            stop,
        ):
            text += piece
            if on_token:
                on_token(piece)
            if stop(text):
                break

        completion_tokens = (
            self.backend.last_completion_tokens or self.backend.count_tokens(text)
        )
        return Generation(text=text.strip(), completion_tokens=completion_tokens)

    def unload(self) -> None:
        self.backend.unload()
