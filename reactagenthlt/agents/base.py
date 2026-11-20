from abc import ABC, abstractmethod
from dataclasses import dataclass

from reactagenthlt.agents.language_model import LanguageModel, TokenUsage


@dataclass
class Response:
    output: str
    token_usage: TokenUsage
    generation_time: float


class BaseAgent(ABC):
    """Shared conversation state and interface for agents using a language model."""

    def __init__(self, language_model: LanguageModel) -> None:
        """Use an existing model without loading or taking ownership of it."""
        self.language_model = language_model

        self.system_prompts: list[dict[str, str]] = []
        self.history: list[dict[str, str]] = []

    def add_system_prompt(self, prompt: str) -> None:
        self.system_prompts.append(
            {
                "role": "system",
                "content": prompt,
            }
        )

    @abstractmethod
    def ask(
        self,
        prompt: str,
        do_sample: bool = False,
        stream: bool = False,
    ) -> Response:
        """Process a prompt and return its answer and generation metrics."""
        ...

    def clear_ctx(self) -> None:
        """Clear conversation history while preserving system prompts."""
        self.history.clear()
