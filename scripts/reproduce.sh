#!/usr/bin/env bash
# Re-runs the whole study from the raw FRED-MD file, in dependency order.
#
#   bash scripts/reproduce.sh            evaluate with the committed TFT checkpoints (about 1 minute)
#   bash scripts/reproduce.sh --retrain  also repeat the hyperparameter search and retrain every TFT
#
# Set PYTHON=/path/to/python to choose the interpreter (default: .venv/bin/python if present, else python).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${PYTHON:-}" ]; then
  if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python; fi
fi
SEEDS="42 43 44 45 46"

run() { echo "== $*"; "$PYTHON" "$@"; }

run -m pytest tests -q                  # unit tests on synthetic data
run src/build_dataset.py                # raw csv -> transformed series
run src/data_pipeline.py                # -> estimation panel, ADF tests, figures/targets.png
run src/baselines.py                    # RW, Mean, ARIMA, VAR walk-forward forecasts
if [ "${1:-}" = "--retrain" ]; then
  run src/tft_train.py --tune           # grid search on UNRATE h=3 (validation loss)
  run src/tft_train.py --seeds $SEEDS   # six models per seed
  run src/tft_predict.py --seeds $SEEDS
else
  run src/tft_predict.py                # committed checkpoints (seed 42)
fi
run src/evaluate.py                     # RMSE, MAE, Diebold-Mariano tests
run src/plots.py                        # figures
echo "Done. Tables: results/summary.md"
