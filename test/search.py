import time

import numpy as np
from colorama import Fore

from reactagenthlt.tools import SearchEngine

if __name__ == "__main__":
    se = SearchEngine(search_by="document", top_k=1)

    while True:
        try:
            query = input(f"{Fore.YELLOW}search: {Fore.RESET}")
            start = time.perf_counter()
            results = se.search(query)
            end = time.perf_counter()

            for i, res in enumerate(results):
                print(f"{Fore.GREEN}result {i + 1}{Fore.RESET}\n{res}")

            print(
                f"{Fore.LIGHTBLACK_EX}"
                + f"--- search time: {end - start:.2f} seconds ---"
                + f"{Fore.RESET}"
            )
        except EOFError:
            break
