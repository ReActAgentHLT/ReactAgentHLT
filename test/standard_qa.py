import argparse
import re
import string
import time
from collections import Counter

from colorama import Fore, Style
from datasets import load_dataset

from reactagenthlt import Agent


def remove_thinking(text: str) -> str:
    """Remove reasoning/thinking content enclosed in <think> tags."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def normalize_answer(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, articles, and extra spaces."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punctuation(s: str) -> str:
        return "".join(ch for ch in s if ch not in string.punctuation)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    return white_space_fix(remove_articles(remove_punctuation(text.lower())))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact Match (EM): 1.0 if normalized prediction matches ground truth exactly, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_accuracy(prediction: str, ground_truth: str) -> float:
    """Accuracy: 1.0 if normalized ground truth is contained in normalized prediction, else 0.0."""
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)
    if not norm_gold:
        return 0.0
    return float(norm_gold in norm_pred)


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def run_benchmark(
    config_path: str, num_samples: int = 50, split: str = "validation"
) -> None:
    print(f"Loading Agent with config: {config_path}")
    agent = Agent(config_path)

    system_prompt = (
        "provide just the answer to the question without explainations or details"
    )
    agent.add_sys_prompt(system_prompt)

    print(f"Loading HotpotQA dataset (split: {split}[:{num_samples}])...")
    try:
        dataset = load_dataset(
            "hotpotqa/hotpot_qa", "distractor", split=f"{split}[:{num_samples}]"
        )
    except Exception:
        dataset = load_dataset(
            "hotpot_qa", "distractor", split=f"{split}[:{num_samples}]"
        )

    print(
        f"Starting standard QA evaluation on {len(dataset)} questions...\n" + "=" * 60
    )

    em_scores: list[float] = []
    acc_scores: list[float] = []
    f1_scores: list[float] = []

    for idx, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["answer"]

        start_time = time.perf_counter()
        raw_output = agent.prompt(question)
        elapsed = time.perf_counter() - start_time

        prediction = remove_thinking(raw_output)

        # Clear context history so questions are answered independently
        agent.clear_context()

        em = compute_exact_match(prediction, ground_truth)
        acc = compute_accuracy(prediction, ground_truth)
        f1 = compute_f1(prediction, ground_truth)

        em_scores.append(em)
        acc_scores.append(acc)
        f1_scores.append(f1)

        print(f"[{idx}/{len(dataset)}] ({elapsed:.2f}s)")
        print(f"  {Fore.CYAN}Question:{Style.RESET_ALL} {question}")
        print(f"  {Fore.GREEN}Ground Truth:{Style.RESET_ALL} {ground_truth}")
        print(f"  {Fore.YELLOW}Prediction:{Style.RESET_ALL} {prediction}")
        print(f"  Scores -> EM: {em:.2f} | Acc: {acc:.2f} | F1: {f1:.2f}\n")

    mean_em = (sum(em_scores) / len(em_scores)) * 100 if em_scores else 0.0
    mean_acc = (sum(acc_scores) / len(acc_scores)) * 100 if acc_scores else 0.0
    mean_f1 = (sum(f1_scores) / len(f1_scores)) * 100 if f1_scores else 0.0

    print("=" * 60)
    print(
        f"{Fore.MAGENTA}FINAL EVALUATION RESULTS ({len(dataset)} questions):{Style.RESET_ALL}"
    )
    print(f"  Exact Match (EM): {mean_em:.2f}%")
    print(f"  Accuracy:         {mean_acc:.2f}%")
    print(f"  F1 Score:         {mean_f1:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard QA evaluation on HotpotQA")
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="configs/qwen3-0.6B.json",
        help="Path to agent configuration JSON file (default: configs/qwen3-0.6B.json)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of questions to evaluate (default: 50)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="HotpotQA split (default: validation)",
    )
    args = parser.parse_args()

    run_benchmark(args.config, args.num_samples, args.split)
