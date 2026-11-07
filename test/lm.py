import re
import typing

import pandas as pd

from reactagenthlt import LanguageModel


def remove_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


if __name__ == "__main__":
    system_prompt = """ provide just the answer to the question without
    explainations or details """
    question = "Who is the USA president?"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    generation_specs = {
        "max_new_tokens": 32768,
        "do_sample": True,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
    }

    agent = LanguageModel(model_id="Qwen/Qwen3-1.7B")
    answer = str(agent.generate(messages, generation_specs))
    answer_without_thinking = remove_thinking(answer)

    print(f"question: {question}")
    print(f"answer: {answer_without_thinking}")
