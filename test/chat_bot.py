import argparse
import json
import re
import time

from colorama import Fore

import reactagenthlt as hlt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to agent config")
    args = parser.parse_args()

    system_prompt = """ provide just the answer to the question without
    explainations or details """

    agent = hlt.Agent(args.config)
    agent.add_sys_prompt(system_prompt)

    while True:
        try:
            prompt = input(f"{Fore.GREEN}User: {Fore.RESET}")
            start = time.perf_counter()
            answer = hlt.remove_thinking(agent.prompt(prompt))

            end = time.perf_counter()
            print(f"{Fore.LIGHTRED_EX}Agent:{Fore.RESET} {answer}")
            print(
                f"""{Fore.LIGHTBLACK_EX}--- response in {end - start:.2f}s ---{Fore.RESET}"""
            )
        except EOFError:
            print(f"\n{Fore.LIGHTRED_EX}Agent:{Fore.RESET} Bye")
            break
