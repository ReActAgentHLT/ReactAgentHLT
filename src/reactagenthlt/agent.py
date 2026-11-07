import json
import typing

from .language_model import LanguageModel


class Agent:
    def __init__(self, config_path: str) -> None:

        with open(config_path, "r") as conf:
            config = dict(json.load(conf))

        self.lm = LanguageModel(config["model"])
        self.generation_specs = config["generation_specs"]
        self.sys_prompts = []
        self.history = []

    def add_sys_prompt(self, content: str) -> None:
        self.sys_prompts.append(
            {
                "role": "system",
                "content": content,
            }
        )

    def prompt(self, content: str) -> str:
        self.history.append(
            {
                "role": "user",
                "content": content,
            }
        )
        messages = self.sys_prompts + self.history

        response = self.lm.generate(messages, self.generation_specs)
        self.history.append({"role": "assistant", "content": response})

        return response

    def clear_context(self) -> None:
        self.history.clear()
