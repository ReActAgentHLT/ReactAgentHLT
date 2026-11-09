import string
import time
import typing
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from colorama import Fore
from datasets import load_dataset

from reactagenthlt.agents import Standard
from reactagenthlt.utility import (
    accuracy_score,
    exact_match,
    f1_score,
    precision,
    recall,
)


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
    )[5 : 5 + args.num_samples]

    agent = Standard(args.config)
    agent.add_sys_prompt(
        """answer concisely to submitted questions without explainations and
        without recovering what was asked in the question. Your answers
        are evaluated against a ground truth answer that is very concise like
        'yes', 'no' or just the name of something"""
    )
    # agent.add_sys_prompt("/no_think")

    questions = df["question"].to_list()
    ground_truths = df["answer"].to_list()

    results = {
        "exact_match": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "thinking_tokens": [],
        "answer_tokens": [],
        "time": [],
    }

    for question, target in zip(questions, ground_truths):
        start = time.perf_counter()
        response = agent.ask(question)
        end = time.perf_counter()

        print(f"{Fore.LIGHTRED_EX}Question:{Fore.RESET} {question}")
        print(f"{Fore.GREEN}Ground truth:{Fore.RESET} {target}")
        print(f"{Fore.YELLOW}Answer:{Fore.RESET} {response.answer}")

        # always clear context to make the agent focus on single questions
        agent.clear_context()

        # lower case everything and remove punctuation
        target = normalize(target)
        answer = normalize(response.answer)
        thinking = normalize(response.thinking)

        results["exact_match"].append(exact_match(answer, target))
        results["accuracy"].append(accuracy_score(answer, target))
        results["precision"].append(precision(answer, target))
        results["recall"].append(recall(answer, target))
        results["f1"].append(f1_score(answer, target))

        # log metrics for the current question
        em = results["exact_match"][-1]
        acc = results["accuracy"][-1]
        prec = results["precision"][-1]
        rec = results["recall"][-1]
        f1 = results["f1"][-1]

        avg_em = np.mean(results["exact_match"])
        avg_accuracy = np.mean(results["accuracy"])
        avg_precision = np.mean(results["precision"])
        avg_recall = np.mean(results["recall"])
        avg_f1 = np.mean(results["f1"])

        print(f"exact_match: {em} - avg: {avg_em:.2f}")
        print(f"accuracy: {acc:.2f} - avg: {avg_accuracy:.2f}")
        print(f"precision: {prec:.2f} - avg: {avg_precision:.2f}")
        print(f"recall: {rec:.2f} - avg: {avg_recall:.2f}")
        print(f"f1: {f1:.2f} - {avg_f1:.2f}")

        # tokens usage
        thinking_tokens = response.usage.thinking_tokens
        answer_tokens = response.usage.answer_tokens

        results["thinking_tokens"].append(thinking_tokens)
        results["answer_tokens"].append(answer_tokens)

        avg_thinking_tokens = np.mean(results["thinking_tokens"])
        avg_answer_tokens = np.mean(results["answer_tokens"])
        print(f"thinking tokens: {thinking_tokens} - avg: {avg_thinking_tokens:.2f}")
        print(f"answer tokens: {answer_tokens} - avg: {avg_answer_tokens:.2f}")

        # time to answer a question
        results["time"].append(end - start)
        avg_time = np.mean(results["time"])
        print(f"time: {results['time'][-1]:.2f}s - avg: {avg_time:.2f}s")

        print("=" * 50)

        # log metrics progression over the dataset
        # print("Accumulated results")
        # print(f"EM: {results['exact_match'][-1]}")
        # print(f"Accuracy: {results['accuracy'][-1]}")
        # print(f"Precision: {results['precision'][-1]}")
        # print(f"Recall: {results['recall'][-1]}")
        # print(f"F1: {results['f1'][-1]}")
        # print("=" * 50)

    results_df = pd.DataFrame(results)
