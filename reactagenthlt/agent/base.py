from abc import ABC


def get_thinking(model: str, thinking: bool) -> str:
    """
    return the string that explicitly enables or disables the thinking for a
    specific model.
    """
    return ""


class Agent(ABC):
    def __init__(self, config: str, thinking: bool = False) -> None:
        self.system_prompts = [
            {
                "role": "system",
                "content": sys_prompt,
            }
        ]
        self.system_prompts.append(
            {
                "role": "system",
                "content": get_thinking(config.model, thinking),
            }
        )

        self.history = []

    def ask(self, prompt: str):
        pass

    def clear_ctx(self) -> None:
        pass
