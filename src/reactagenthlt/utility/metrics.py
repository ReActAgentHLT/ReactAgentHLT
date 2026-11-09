from collections import Counter


def exact_match(predicted: str, target: str) -> float:
    return 1.0 if predicted == target else 0.0


def accuracy_score(predicted: str, target: str) -> float:
    if not target:
        return 0.0
    return 1.0 if target in predicted else 0.0


def precision_score(predicted: str, target: str) -> float:
    pred_tokens = predicted.split()
    gold_tokens = target.split()

    if not pred_tokens:
        return 1.0 if not gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    return num_same / len(pred_tokens)


def recall_score(predicted: str, target: str) -> float:
    pred_tokens = predicted.split()
    gold_tokens = target.split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    return num_same / len(gold_tokens)


def f1_score(predicted: str, target: str) -> float:
    p = precision_score(predicted, target)
    r = recall_score(predicted, target)
    if p + r == 0:
        return 0.0
    return (2 * p * r) / (p + r)


precision = precision_score
recall = recall_score
f1 = f1_score

