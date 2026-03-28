# Speculatores 14.5

`Speculatores 14.5` is the script-backed optimization pipeline for this repo.
It is intended to replace long-running notebook execution for heavy Optuna
searches, especially on Colab Pro where multiprocessing needs importable
top-level code and storage that tolerates concurrent workers.

## What changed

- Uses a real Python runner: `scripts/run_speculatores_145.py`
- Uses Optuna `JournalStorage` instead of notebook-local SQLite
- Supports parallel worker processes per side (`high` and `low`)
- Reuses fold-level detector precomputations from the library layer
- Exports one timestamped Markdown report per run with:
  - dataset metadata
  - default-params sanity run
  - best params for both sides
  - walk-forward fold breakdown
  - temporal holdout summary
  - cross-asset summary
  - Pine export block

## Example

```powershell
python scripts/run_speculatores_145.py `
  --dataset data/raw/SPX_1D_18710201_20260318.csv `
  --trials-per-side 500 `
  --workers-per-side 2 `
  --storage temp/speculatores_14_5.journal `
  --results-dir results
```

## Colab notes

- Prefer CPU runtimes. This workload is pandas/numpy + Python loops, not GPU-heavy.
- Keep `workers-per-side` modest. `2` is a reasonable starting point.
- Use a Drive-backed storage/report path if you need resume-safe runs.
