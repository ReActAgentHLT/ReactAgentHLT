import time

from reactagenthlt.agents.base import BaseAgent, Response
from reactagenthlt.agents.language_model import LanguageModel
from reactagenthlt.tools import Calculator, SearchEngine


class ReActAgent(BaseAgent):
    """ReAct scaffold; ask currently performs only a single generation."""

    def __init__(
        self,
        language_model: LanguageModel,
        search_engine: SearchEngine | None = None,
        calculator: Calculator | None = None,
        max_steps: int = 7,
    ) -> None:
        """Accept explicit tools and a step limit for the future ReAct loop."""
        if max_steps < 1:
            raise ValueError("Maximum steps must be at least 1.")
        super().__init__(language_model)
        self.search_engine = search_engine
        self.calculator = calculator if calculator is not None else Calculator()
        self.max_steps = max_steps

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
    agent.add_system_prompt("")
    while True:
        try:
            prompt = input("user: ")
            print("agent: ", end="")
            response = agent.ask(prompt, do_sample=args.sample, stream=args.stream)

            print()
            print(f"input tokens: {response.token_usage.input_tokens}")
            print(f"output tokens: {response.token_usage.output_tokens}")
            print(f"generation time: {response.generation_time:.2f} seconds")
            print("=" * 50)
        except EOFError:
            break
