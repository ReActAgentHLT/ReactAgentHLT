import re
import string
import time
import typing
from argparse import ArgumentParser

import pandas as pd
from colorama import Fore
from datasets import load_dataset

from reactagenthlt.agents import ChatBot
from reactagenthlt.utility import (
    accuracy_score,
    exact_match,
    f1_score,
    precision,
    recall,
)


def parse_thinking(content: str) -> list[str]:
    match = re.search(r"<think>(.*?)</think>(.*)", content, flags=re.DOTALL)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]
    return ["", content.strip()]


def normalize(content: str) -> str:
    content = content.lower()
    content = content.translate(str.maketrans("", "", string.punctuation))
    return " ".join(content.split())


if __name__ == "__main__":
    # CLI arguments parsing
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="path to model config file")
    parser.add_argument(
        "--num_samples", type=int, default=50, help="how many sample to process"
    )
    args = parser.parse_args()

    # fetch the dataset and take only the first `num_samples` samples
    df = typing.cast(
        pd.DataFrame,
        load_dataset("hotpotqa/hotpot_qa", "distractor")["validation"].to_pandas(),
    )[: args.num_samples]

    agent = ChatBot(args.config)
    agent.add_sys_prompt(
        """answer concisely to submitted questions without explainations and
        without recovering what was asked in the question. Your answers
        are evaluated against a ground truth answer that is very concise like
        'yes', 'no' or just the name of something"""
    )

    questions = df["question"].to_list()
    ground_truths = df["answer"].to_list()

    results = {
        "exact_match": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        # "avg_new_tokens": [],
        "time": [],
    }

    for question, target in zip(questions, ground_truths):
        start = time.perf_counter()
        answer = parse_thinking(agent.ask(question))[1]
        end = time.perf_counter()

        target = normalize(target)
        answer = normalize(answer)

        results["exact_match"].append(exact_match(answer, target))
        results["accuracy"].append(accuracy_score(answer, target))
        results["precision"].append(precision(answer, target))
        results["recall"].append(recall(answer, target))
        results["f1"].append(f1_score(answer, target))
        results["time"].append(end - start)

        print(f"{Fore.RED}question:{Fore.RESET} {question}")
        print(f"{Fore.GREEN}ground truth:{Fore.RESET} {target}")
        print(f"{Fore.YELLOW}answer:{Fore.RESET} {answer}")
        print("-" * 50)

    results_df = pd.DataFrame(results)
    total_samples = len(results_df)
    em_total = int(results_df["exact_match"].sum()) if not results_df.empty else 0
    em_rate = (results_df["exact_match"].mean() * 100) if not results_df.empty else 0.0

    acc_total = int(results_df["accuracy"].sum()) if not results_df.empty else 0
    acc_rate = (results_df["accuracy"].mean() * 100) if not results_df.empty else 0.0

    p_total = int(results_df["precision"].sum()) if not results_df.empty else 0
    p_rate = (results_df["precision"].mean() * 100) if not results_df.empty else 0.0

    r_total = int(results_df["recall"].sum()) if not results_df.empty else 0
    r_rate = (results_df["recall"].mean() * 100) if not results_df.empty else 0.0

    f1_total = int(results_df["f1"].sum()) if not results_df.empty else 0
    f1_rate = (results_df["f1"].mean() * 100) if not results_df.empty else 0.0

    avg_time = results_df["time"].mean() if not results_df.empty else 0.0
    total_time = results_df["time"].sum() if not results_df.empty else 0.0

    print("\n" + "=" * 50)
    print(f"{Fore.CYAN}--- EVALUATION RESULTS ---{Fore.RESET}")
    print(f"Exact Match (EM):        {em_total}/{total_samples} ({em_rate:.2f}%)")
    print(f"Accuracy:                {acc_total}/{total_samples} ({acc_rate:.2f}%)")
    print(f"Precision:                {p_total}/{total_samples} ({p_rate:.2f}%)")
    print(f"Recall:                {r_total}/{total_samples} ({r_rate:.2f}%)")
    print(f"F1 Score:                {f1_total}/{total_samples} ({f1_rate:.2f}%)")
    print(f"Avg time per question:   {avg_time:.2f}s")
    print(f"Total evaluation time:   {total_time:.2f}s")
    print("=" * 50)
