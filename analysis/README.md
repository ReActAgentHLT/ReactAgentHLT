# analysis (evaluation: bootstrap tests and failure analysis)

Run everything from the repo root. Needs `pandas` and `numpy` (already installed with the package).

1. `python analysis/bootstrap.py` : paired bootstrap between modes (direct vs cot, cot vs react,
   react vs focused_react) for F1/EM/accuracy and for time and tokens. Output: `analysis/out/bootstrap.csv`.
2. `python analysis/failures.py counts` : termination reasons of the wrong runs per configuration.
3. `python analysis/failures.py sheet --per-group 25` : sample of wrong runs to label by hand in
   `analysis/out/failure_sheet.csv` (fill the `category` column; `auto_category` is a first guess).
4. `python analysis/failures.py summary` : counts and percentages of the labelled categories.

Both scripts read `results/*.csv` (written by `reactagenthlt.evaluation.save_results`).
Use `--results-dir analysis/fake_results` after `python analysis/fake_results.py` to test with fake data.
