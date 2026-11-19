import json
import time
from dataclasses import dataclass

from reactagenthlt.agents.language_model import LanguageModel, TokenUsage


@dataclass
class Response:
    output: str
    token_usage: TokenUsage
    generation_time: float


class Agent:
    def __init__(self, configpath: str) -> None:
        self.language_model = LanguageModel(configpath)

        self.system_prompts = []
        self.history = []

    def add_system_prompt(self, prompt: str) -> None:
        self.system_prompts.append(
            {
                "role": "system",
                "content": prompt,
            }
        )

    def ask(
        self,
        prompt: str,
        do_sample: bool = False,
        stream: bool = False,
    ) -> Response:
        # append the user input to the history
        self.history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        start = time.perf_counter()
        output, token_usage = self.language_model.generate(
            messages=self.system_prompts + self.history,
            do_sample=do_sample,
            stream=stream,
        )
        end = time.perf_counter()

        # append the agent response to the history
        self.history.append(
            {
                "role": "assistant",
                "content": output,
            }
        )

        return Response(
            output=output,
            token_usage=token_usage,
            generation_time=end - start,
        )

    def clear_ctx(self) -> None:
        self.history.clear()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="filepath to model config")
    parser.add_argument("--sample", action="store_true", help="generate with sampling")
    parser.add_argument("--stream", action="store_true", help="enable chat streaming")
    args = parser.parse_args()

    agent = Agent(configpath=args.config)
    agent.add_system_prompt("your name is Mario and you are an helpful assistant")
    while True:
        try:
            prompt = input("user: ")
            print("agent: ", end="")
            response = agent.ask(prompt, do_sample=args.sample, stream=args.stream)

            print()
            print(f"input tokens: {response.token_usage.input_tokens}")
            print(f"output tokens: {response.token_usage.output_tokens}")
            print(f"generation time: {response.generation_time}")
            print("=" * 50)
        except EOFError:
            break
