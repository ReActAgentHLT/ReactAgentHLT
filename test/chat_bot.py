import argparse
import json
import re
import time

from colorama import Fore

from reactagenthlt import Agent


def remove_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to agent config")
    args = parser.parse_args()

    system_prompt = """ provide just the answer to the question without
    explainations or details """

    agent = Agent(args.config)
    agent.add_sys_prompt(system_prompt)

    while True:
        try:
            prompt = input("> ")
            start = time.perf_counter()
            answer = remove_thinking(agent.prompt(prompt))
            end = time.perf_counter()
            print(
                f"< {answer}\t\t{Fore.GREEN}reponse in {end - start:.2f} seconds{Fore.RESET}"
            )
        except EOFError:
            break
