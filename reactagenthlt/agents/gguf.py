"""Optional llama.cpp backend for the experiment's chat generation interface."""

from pathlib import Path

from reactagenthlt.agents.language_model import TokenUsage


class GGUFModel:
    """Load one local GGUF model and use its chat template for generation."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        max_new_tokens: int = 256,
        verbose: bool = False,
    ) -> None:
        from llama_cpp import Llama

        if max_new_tokens < 1:
            raise ValueError("The generation limit must be positive.")
        self.max_new_tokens = max_new_tokens
        self.model = Llama(
            model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
            seed=0, verbose=verbose,
        )

    def generate(self, messages: list[dict[str, str]]) -> tuple[str, TokenUsage]:
        reply = self.model.create_chat_completion(
            messages=messages, max_tokens=self.max_new_tokens, temperature=0.0
        )
        text = reply["choices"][0]["message"]["content"]
        usage = reply["usage"]
        return text, TokenUsage(usage["prompt_tokens"], usage["completion_tokens"])
