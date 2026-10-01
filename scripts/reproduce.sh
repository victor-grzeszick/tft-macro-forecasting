#!/usr/bin/env bash
# Re-runs the existing pipeline in dependency order. TFT models are only loaded (case_switch = 0):
# nothing is trained, nothing is downloaded, no source file is changed.
# Usage:  bash scripts/reproduce.sh [--toy]      (set PYTHON=/path/to/python to choose the interpreter)
set -eu
cd "$(dirname "$0")/.."     # repo root; every path in src/ is relative to it
export MPLBACKEND=Agg       # plt.show() does not open windows or block
PYTHON=${PYTHON:-python}

echo "WARNING: overwrites tracked files in data/processed/ and figures/ (they match the committed ones only up to floating-point noise across platforms); restore them with: git checkout -- data/processed figures"

# Stop if either TFT script has been switched to training.
for f in src/trainer.py src/pytorchDummy.py; do
  grep -Eq '^[[:space:]]*case_switch = 0' "$f" || { echo "ERROR: $f is not in load mode (case_switch = 0); stopping."; exit 1; }
done

run() { echo "== $1"; "$PYTHON" "src/$1"; }

run build_dataset.py        # data/raw/fred_md.csv -> fred_md_transformed.parquet
run validate.py             # ADF tests, prints only
run data_pipeline.py        # -> X_/Y_ train/val/test parquets
run baseline.py             # RW, ARIMA, VAR -> predictions_unalligned.pkl
for s in align_results.py plot_baselines.py; do    # -> Baseline_Results.csv, figures/{target}_h{h}.png
  run "$s" || echo "KNOWN FAILURE at HEAD — see REVIEW_FOR_VICTOR.md ($s, KeyError: 'TFT'); continuing with the committed outputs"
done
run trainer.py              # load mode; reads Baseline_Results.csv -> figures/tft_*.png + gitignored TFT results
run dm_test.py              # needs trainer.py's predictions_unalligned_tft.pkl -> dm_test_results.parquet
if [ "${1:-}" = "--toy" ]; then
  run pytorchDummy.py       # optional toy demo: loads models/tft_toy.ckpt -> figures/toy_idx*.png
fi
echo "Done."
