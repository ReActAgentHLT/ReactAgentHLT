import argparse
import time

import numpy as np
from colorama import Fore

from reactagenthlt.tools import SearchEngine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="number of retrieved documents for each query",
    )
    args = parser.parse_args()

    se = SearchEngine(search_by="title", top_k=args.k)

    while True:
        try:
            query = input(f"{Fore.YELLOW}search{Fore.RESET}: ")
            start = time.perf_counter()
            results = se.search(query)
            end = time.perf_counter()

            titles = results["title"].to_list()
            contents = results["content"].to_list()

            for i, (title, content) in enumerate(zip(titles, contents)):
                print(f"{Fore.GREEN}{i + 1}. {title}{Fore.RESET}")
                print(content)

            print(
                f"{Fore.LIGHTBLACK_EX}"
                + f"--- search time: {end - start:.2f} seconds ---"
                + f"{Fore.RESET}"
            )
        except EOFError:
            break
