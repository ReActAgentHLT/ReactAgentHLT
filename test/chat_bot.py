import argparse
import time

from colorama import Fore

import reactagenthlt as hlt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to agent config")
    args = parser.parse_args()

    system_prompt = (
        "you are an helpful assistant, try to answer as concisely"
        + " as you can to every submitted question"
    )

    agent = hlt.agents.Standard(args.config)
    agent.add_sys_prompt(system_prompt)

    while True:
        try:
            prompt = input(f"{Fore.CYAN}User: {Fore.RESET}")
            start = time.perf_counter()
            response = agent.ask(prompt)
            end = time.perf_counter()

            usage = (
                response.usage.prompt_tokens
                + response.usage.thinking_tokens
                + response.usage.answer_tokens
            )

            if response.thinking != "":
                print(f"Thinking: {Fore.LIGHTBLACK_EX}{response.thinking}{Fore.RESET}")

            print(f"{Fore.LIGHTRED_EX}Response:{Fore.RESET} {response.answer}")
            print(
                f"{Fore.GREEN}---",
                f"response in {end - start:.2f}s,",
                f"generated tokens: {usage},",
                f"total generated tokens: {agent.total_usage()}",
                f"---{Fore.RESET}",
            )
        except EOFError:
            print(f"\n{Fore.LIGHTRED_EX}Agent:{Fore.RESET} Bye")
            break
