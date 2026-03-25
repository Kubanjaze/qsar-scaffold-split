# qsar-scaffold-split

**Phase 21 — QSAR Model with Scaffold Split**
Track 1 — Cheminformatics Core

Trains a pIC50 RandomForest regressor using ECFP4 fingerprints and compares
predictive performance under scaffold split vs random split — quantifying why
scaffold split gives a more realistic estimate of prospective performance.

## Why scaffold split matters

In drug discovery, prospective QSAR models must predict activity on **new
chemotypes** not seen during training. Random split allows scaffold overlap
between train and test, inflating R² and underestimating generalization error.
Scaffold split forces the test set to contain scaffolds absent from training,
mimicking the real deployment scenario. The gap (ΔR² = random − scaffold) is
the "optimism bias" of random evaluation.

## Scaffold split algorithm

1. Extract Murcko scaffold for each compound.
2. Force the **single largest scaffold group** into TRAIN (prevents the dominant
   scaffold from filling the test set).
3. Sort remaining groups by: `(size desc, hetero_atom_count desc, smiles asc)`.
   The hetero-atom count tiebreaker preferentially sends N/O-containing scaffolds
   (e.g., indole, quinoline) to the test set before all-carbon scaffolds (naphthalene).
4. Greedily add groups to TEST until `N_test >= ceil(test_fraction × N)`.
5. Remaining groups go to TRAIN.

## ECFP4 settings

| Parameter | Value |
|---|---|
| Radius | 2 (ECFP4) |
| nBits | 2048 |
| useChirality | False |
| Function | `AllChem.GetMorganFingerprintAsBitVect` |

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

PYTHONUTF8=1 .venv\Scripts\python main.py --input data/compounds.csv
```

On Windows PowerShell:

```powershell
$env:PYTHONUTF8=1
.venv\Scripts\python main.py --input data/compounds.csv
```

> **RDKit install note:** `pip install rdkit` works on most systems. If it fails,
> use conda: `conda install -c conda-forge rdkit`

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Compounds CSV (compound_name, smiles, pic50) |
| `--test-fraction` | 0.25 | Target test fraction for scaffold split (integer groups may overshoot) |
| `--top-n` | 20 | Top-N ECFP4 bits in importance plot |
| `--radius` | 2 | Morgan fingerprint radius |
| `--nbits` | 2048 | Morgan fingerprint bit count |
| `--output-dir` | output | Output directory |

## Outputs

| File | Description |
|---|---|
| `output/qsar_results.csv` | Per-compound: scaffold, split assignment (train/test), pIC50, predictions (test only) |
| `output/split_comparison.png` | 1×2 parity plots with R²/MAE/RMSE; consistent axes across panels |
| `output/feature_importance.png` | Top-N ECFP4 bits by importance (scaffold-split model) |

## Notes

- **Synthetic dataset**: 45 compounds across 6 scaffold families with pIC50 correlated
  with EWG/EDG substituents and ring count. Metrics are illustrative, not predictive.
- **Feature importance**: impurity-based (Gini), from `model.feature_importances_`. Bit
  indices (e.g., `bit_0042`) are not directly chemically interpretable without a
  substructure lookup (e.g., via `AllChem.GetMorganFingerprintAsBitVect` with `bitInfo`).
- **Import ordering**: `matplotlib.use("Agg")` before pyplot; `RDLogger.DisableLog("rdApp.*")`
  before any RDKit mol operations.
- **Random split test size** matches the scaffold split test count for a fair comparison.
- Train predictions are left as NaN in `qsar_results.csv` to avoid confusion with test predictions.
