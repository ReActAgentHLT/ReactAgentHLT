from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
OUT = Path(__file__).parent / "fake_results"
OUT.mkdir(exist_ok=True)
ACC = {"direct": 0.15, "cot": 0.35, "react": 0.30, "focused_react": 0.38}
TERM = ["finished", "forced_finish", "budget_exhausted", "malformed_action", "missing_action", "invalid_tool_input"]

for dataset in ("hotpotqa", "gsm8k"):
    questions = [f"{dataset} question {i}" for i in range(100)]
    for mode, acc in ACC.items():
        ok = rng.random(100) < acc
        agent = mode in ("react", "focused_react")
        rows = pd.DataFrame({
            "model": "Qwen2.5-3B-Instruct", "dataset": dataset, "mode": mode, "k": 3, "budget": 9,
            "question": questions, "gold": "42", "prediction": np.where(ok, "42", "17"),
            "termination": np.where(ok, "finished", rng.choice(TERM, 100)),
            "actions": rng.integers(2, 9, 100) if agent else 1,
            "seconds": rng.gamma(4, 2 if agent else 0.8, 100),
            "input_tokens": rng.integers(300, 2500, 100) if agent else 400,
            "output_tokens": rng.integers(40, 300, 100),
            "correct": ok,
        })
        rows["failed_action"] = rows["termination"].isin(["malformed_action", "missing_action", "invalid_tool_input"])
        if dataset == "hotpotqa":
            f1 = np.where(ok, 1.0, rng.random(100) * 0.4)
            rows["em"], rows["precision"], rows["recall"], rows["f1"] = (f1 == 1).astype(float), f1, f1, f1
            rows["searches"], rows["gold_searches"] = 2, rng.integers(0, 3, 100)
        rows.to_csv(OUT / f"fake_{dataset}_{mode}.csv", index=False)
print(f"Fake results written to {OUT}")
