# Reproducing the results

**Short version.** On a clean machine, one script rebuilds the data splits and baseline forecasts, loads the six trained TFTs, and recomputes the DM statistics. No step trains a model. <!-- src: scripts/reproduce.sh:2-3,12-15 --> Regenerated files match the committed ones only up to floating-point noise across platforms, not byte for byte. <!-- src: scripts/reproduce.sh:10 --> Two scripts crash at the current commit, so `Baseline_Results.csv` and the six baseline figures must be rebuilt with the older code that made them (section 4). <!-- src: src/evaluation_utils.py:16; src/align_results.py:21; src/plot_baselines.py:19 --> Retraining the TFTs is not covered (section 5).

TODO(Victor): commit a log of one clean run (machine, OS, Python patch version, install time, per-script runtimes, largest differences from the committed files) and cite it here.

## 1. Environment

Python 3.11. Every package in `requirements.txt` is pinned to an exact version. <!-- src: README.md@80e837f:16; requirements.txt:1-51 -->

```bash
git clone https://github.com/victor-grzeszick/tft-macro-forecasting.git
cd tft-macro-forecasting
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
<!-- src: README.md@80e837f:18-24 -->

- **Linux:** `requirements.txt` lists no CUDA packages. TODO(Victor): note which CUDA wheels pip pulls for `torch==2.11.0` on Linux, and the install size. <!-- src: requirements.txt:46 (torch==2.11.0; no nvidia-* lines in the file) -->
- **Device:** no script requests a GPU or any other device; Lightning chooses one automatically. <!-- src: git grep -i -E "accelerator|device|mps|cuda" -- src scripts (no hits); src/trainer.py:19 -->
- **No network or API key needed.** `FRED_API_KEY` is used only by `archive/fred_test.py`, which is not part of the pipeline. <!-- src: archive/fred_test.py:7; scripts/reproduce.sh:19-30 -->

## 2. Check the data file

```bash
sha256sum data/raw/fred_md.csv      # macOS: shasum -a 256 data/raw/fred_md.csv
# expected: 5d1b4008d3b4d5cb14883c395e8d1d80750b8b0a4f301e24c91f3f6710621663
```
<!-- src: sha256sum data/raw/fred_md.csv at 80e837f -->

Do not download a newer FRED-MD file: values are revised every month, so the numbers would change. <!-- src: README.md@80e837f:28-29 --> Source and vintage: [DATA.md](DATA.md).

## 3. Run the pipeline

From the repo root:

```bash
bash scripts/reproduce.sh                          # main pipeline
bash scripts/reproduce.sh --toy                    # also the synthetic-sine TFT demo
PYTHON=.venv/bin/python bash scripts/reproduce.sh  # choose the interpreter
git checkout -- data/processed figures             # afterwards: restore the committed outputs
```
<!-- src: scripts/reproduce.sh:4,8,10,28 -->

The script sets `MPLBACKEND=Agg` (so plots never open a window), stops if either TFT script has been switched to training, and runs the steps below in dependency order. <!-- src: scripts/reproduce.sh:7,13-15,19-27 --> It **overwrites tracked files** in `data/processed/` and `figures/`; the last command above restores them. <!-- src: scripts/reproduce.sh:10 --> To run one script by hand: `MPLBACKEND=Agg python src/<script>.py` from the repo root; all paths in the code are relative to it. <!-- src: scripts/reproduce.sh:6 -->

| # | Script | Exit | What you should see |
|---|---|---|---|
| 1 | `build_dataset.py` | 0 | writes `fred_md_transformed.parquet` (807 months × 126 series) |
| 2 | `validate.py` | 0 | prints five ADF results; the expected values are in `notes/notes_data.md` lines 46–69 |
| 3 | `data_pipeline.py` | 0 | prints `(420, 125)`, `['ACOGNO']`, `(336, 125)`, `(24, 125)`, `(60, 125)`; writes the six `X_`/`Y_` parquets |
| 4 | `baseline.py` | 0 | writes `predictions_unalligned.pkl`: 18 forecast series (3 targets × 2 horizons × RW, ARIMA, VAR) |
| 5 | `align_results.py` | **1** | **crashes** with `KeyError: 'TFT'`; the script prints `KNOWN FAILURE` and continues |
| 6 | `plot_baselines.py` | **1** | **crashes** the same way |
| 7 | `trainer.py` (load mode) | 0 | loads six checkpoints; prints one `Target: … Horizon … MAE: … RMSE: …` line per model; writes `figures/tft_*.png` and three gitignored files |
| 8 | `dm_test.py` | 0 | prints a three-row table and writes `dm_test_results.parquet`. Committed values (5 decimals): DM −2.13928 (CPIAUCSL, ARIMA vs TFT), 0.59232 (UNRATE, VAR vs TFT), −2.15214 (INDPRO, ARIMA vs TFT) |
| toy | `pytorchDummy.py` | 0 | writes `figures/toy_idx{0,10,20}.png` |

<!-- src: 1 src/build_dataset.py:36-60, data/processed/fred_md_transformed.parquet shape; 2 src/validate.py:20-27, notes/notes_data.md:46-69; 3 src/data_pipeline.py:28-39,70-76; 4 src/baseline.py:86-104, data/processed/predictions_unalligned.pkl (18 rows); 5-6 src/evaluation_utils.py:16, scripts/reproduce.sh:23-25; 7 src/trainer.py:24,97-100,121-139; 8 src/dm_test.py:59-65, data/processed/dm_test_results.parquet; toy src/pytorchDummy.py:38-47 -->

Exit is each script's exit status. The script runs with `set -e`, so any non-zero exit other than steps 5–6 stops it; a complete run ends with `Done.` <!-- src: scripts/reproduce.sh:5,24,31 --> Step 7's TFT numbers have no committed copy to compare against ([FINDINGS §2](FINDINGS.md#2-tft-validation)).

## 4. Rebuild `Baseline_Results.csv` and the baseline figures

At the current commit, `align()` requires a TFT row that the baseline forecasts do not have (`src/evaluation_utils.py:16`), and both callers unpack four values where `align()` returns five (`src/align_results.py:21`, `src/plot_baselines.py:19`). <!-- src: src/evaluation_utils.py:16,26; src/align_results.py:21; src/plot_baselines.py:19 --> Commit 770bc4d last changed the CSV, the forecasts file and the six baseline figures; its `align()` needs no TFT row and returns the four values the callers expect. <!-- src: git log -- data/processed/Baseline_Results.csv data/processed/predictions_unalligned.pkl figures/UNRATE_h3.png; git show 770bc4d:src/evaluation_utils.py; git show 770bc4d:src/align_results.py (line 16) --> Run those versions from the repo root without changing any tracked source:

```bash
H=$(mktemp -d)
for f in evaluation_utils align_results plot_baselines; do git show 770bc4d:src/$f.py > "$H/$f.py"; done
MPLBACKEND=Agg python "$H/align_results.py"     # -> data/processed/Baseline_Results.csv
MPLBACKEND=Agg python "$H/plot_baselines.py"    # -> figures/{CPIAUCSL,UNRATE,INDPRO}_h{3,12}.png
sha256sum data/processed/Baseline_Results.csv
```

- On the committed `predictions_unalligned.pkl` (fresh clone, or after `git checkout -- data/processed`), the CSV should be byte-identical to the committed one: SHA-256 `4dc4daadad471f66e87b95195e5e35be238ab60b921e510526d0439e552e5ae2`. <!-- src: sha256sum data/processed/Baseline_Results.csv at 80e837f -->
- After step 4 has regenerated the forecasts, expect floating-point noise instead: compare values, not hashes, and the figures by eye. <!-- src: scripts/reproduce.sh:10 -->

TODO(Victor): decide whether to fix the current `align()` or keep this workaround.

## 5. Load mode and training mode

- `case_switch = 0` (load) is set in `src/trainer.py:97` and `src/pytorchDummy.py:14`. Load mode reads `models/tft_{target}_h{h}.ckpt`, predicts the validation windows and writes plots and metrics. It does not train. <!-- src: src/trainer.py:97,24,121-139; src/pytorchDummy.py:14,38 -->
- `case_switch = 1` trains and **overwrites the committed checkpoints**: early stopping on `val_loss` with patience 15, at most 500 epochs, seed 42, hyperparameters hard-coded at `src/trainer.py:117`. <!-- src: src/trainer.py:17-22,92,117 --> `tune_architecture()` is not called at the current commit (commit b16fcbe called it in `main()`). <!-- src: src/trainer.py:61-63; b16fcbe:src/trainer.py:90 -->
- Retraining has not been tested, so whether it gives the same checkpoints is unknown. TODO(Victor): record the training hardware, time per model and torch version (the notes mention "torch 2.6"; the pin is 2.11.0). <!-- src: notes/notes_models.md:147; requirements.txt:46 -->

## 6. Known blockers and what is not reproducible

1. **Hidden order dependency.** `dm_test.py` reads `data/processed/predictions_unalligned_tft.pkl`, which only `trainer.py` writes, and `trainer.py` reads `Baseline_Results.csv`. So the order is: baselines → results CSV → trainer → DM. <!-- src: src/dm_test.py:27; src/trainer.py:135-136 -->
2. **Side effects.** Each `trainer.py` run creates 24 `lightning_logs/version_*` folders (gitignored): four prediction calls per model, six models. <!-- src: src/trainer.py:30,38,44,56; .gitignore:225 -->
3. **ARIMA orders** are not written to any output file, and no committed code computes the VAR lag comparison; both appear only in `notes/notes_models.md`. <!-- src: src/baseline.py:27,57; notes/notes_models.md:35-38,44; git grep -i -E "select_order|bic|aic|hqic" over all revisions, *.py (no hits) -->
