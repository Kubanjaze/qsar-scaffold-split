# Phase 21 — QSAR Model with Scaffold Split

**Version:** 1.1 (as-built)
**Author:** Kerwyn Medrano
**Date:** 2026-03-25
**Track:** Track 1 — Cheminformatics Core
**Tier:** Small (3–5 hrs)
**API Cost:** $0.00 — pure RDKit + scikit-learn + pandas + matplotlib

---

## 1. Project Overview

### Goal

Build a pIC50 regression model (RandomForest) using ECFP4 fingerprints on a
curated synthetic dataset. Evaluate with both scaffold-based and random
train/test splits, compare their R²/MAE/RMSE, and explain why scaffold split
gives a more realistic estimate of prospective performance.

```bash
python main.py --input data/compounds.csv
```

Outputs:
- `output/qsar_results.csv` — per-compound predictions for both splits
- `output/split_comparison.png` — 1×2 parity plot: scaffold split (left) vs random split (right)
- `output/feature_importance.png` — top-20 ECFP4 bit importances (scaffold split model)

### What This Phase Teaches

| Concept | Detail |
|---|---|
| ECFP4 fingerprint | `AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)` |
| Scaffold split | Train/test separated by Murcko scaffold; tests prospective generalization |
| Random split | `train_test_split(random_state=42)`; optimistically inflates metrics |
| RandomForest regressor | `RandomForestRegressor(n_estimators=200, random_state=42)` |
| Evaluation metrics | R², MAE, RMSE on test set |
| Split comparison | Same model class; only splitting strategy differs |

### Domain Context

In drug discovery, scaffold split is the standard evaluation protocol for QSAR
models intended for prospective use: the test set contains scaffolds not seen
during training, mimicking the real scenario of predicting activity on novel
chemotypes. Random split allows scaffold overlap between train and test,
inflating R² and underestimating generalization error. This phase quantifies
the gap — a key lesson for anyone building predictive models in medicinal chemistry.

---

## 2. Architecture

```
qsar-scaffold-split/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── compounds.csv
└── output/
    ├── qsar_results.csv
    ├── split_comparison.png
    └── feature_importance.png
```

---

## 3. Input Format

### `data/compounds.csv`
```csv
compound_name,smiles,pic50
cpd_001,C=CC(=O)Nc1ccc(F)cc1,7.2
cpd_002,...,6.8
...
```
- 40–50 synthetic compounds spanning pIC50 5.0–9.5
- Diverse scaffolds: benzene, naphthalene, indole, quinoline, pyrimidine, benzimidazole
- At least 6 distinct Murcko scaffolds (to make scaffold split meaningful)
- pIC50 correlated with fingerprint features (not purely random) so model can learn

---

## 4. Featurization

### ECFP4 fingerprint
```python
from rdkit.Chem import AllChem
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=False)
X = np.array(fp)  # shape (2048,)
```

- radius=2, nBits=2048, useChirality=False (standard ECFP4 for QSAR)
- Feature matrix: shape (N_compounds, 2048)

---

## 5. Scaffold Split

```python
from rdkit.Chem.Scaffolds import MurckoScaffold

def scaffold_split(df, test_fraction=0.25, random_state=42):
    # 1. Extract Murcko scaffold for each compound
    # 2. Group compounds by scaffold
    # 3. Sort scaffold groups by size descending
    # 4. Greedily assign groups to test until test_fraction reached
    # 5. Return train_idx, test_idx
```

- Target test fraction: 0.25 (≈10–12 compounds in test)
- Greedy assignment: add smallest scaffold groups to test first to reach target
  (prevents large scaffold clusters from over-filling test set)
- Print scaffold distribution: scaffold → n_compounds, assigned to train/test

---

## 6. Module Specification

### `load_compounds(path)` → pd.DataFrame
- Standard loader: compound_name, smiles, pic50 required; skip invalid SMILES
- Return df with compound_name, smiles, mol, pic50

### `featurize(df)` → np.ndarray
- Compute ECFP4 for each mol
- Return X shape (N, 2048), array of float64

### `scaffold_split(df, test_fraction, random_state)` → (train_idx, test_idx)
- Murcko scaffold extraction + greedy assignment as above
- Return numpy index arrays

### `evaluate(y_true, y_pred)` → dict
- Compute R², MAE, RMSE
- Return {"r2": float, "mae": float, "rmse": float}

### `train_and_evaluate(X, y, train_idx, test_idx, label)` → dict
- Fit `RandomForestRegressor(n_estimators=200, random_state=42)` on train
- Predict on test
- Return dict: label, metrics, y_test, y_pred, feature_importances

### `plot_split_comparison(scaffold_results, random_results, output_path)`
- 1×2 parity plot (predicted vs actual pIC50)
- Left: scaffold split, right: random split
- Each: scatter + y=x diagonal + R²/MAE annotation in plot corner
- Consistent axis limits across both panels
- Save 150 dpi

### `plot_feature_importance(importances, top_n, output_path)`
- Horizontal bar chart: top-N ECFP4 bits by importance (scaffold split model)
- x = importance, y = "bit_XXXX" label
- Save 150 dpi

### `main()`
- `--input` (required): compounds CSV
- `--test-fraction` (default: 0.25): scaffold split test fraction
- `--top-n` (default: 20): top-N bits in importance plot
- `--output-dir` (default: output)
- Print: scaffold split metrics, random split metrics, delta R² (random − scaffold)

---

## 7. Seed Data Design

### ~45 compounds across 6 scaffold groups

| Scaffold | n_compounds | pIC50 range | Scaffold group strategy |
|---|---|---|---|
| Benzene (acrylamide-aniline) | 12 | 6.0–8.5 | Largest group → train (mostly) |
| Naphthalene | 7 | 6.5–8.0 | Split across train/test |
| Indole | 7 | 7.0–9.0 | Goes to test (scaffold split) |
| Quinoline | 7 | 6.5–8.5 | Goes to test (scaffold split) |
| Pyrimidine | 6 | 5.5–7.5 | Goes to train |
| Benzimidazole | 6 | 6.0–8.0 | Goes to train |

Design rules for pIC50 correlation:
- EWG on arene (F, Cl, CF3) → higher pIC50 (+0.5–1.0 units)
- EDG on arene (OMe, Me, OH) → lower pIC50 (−0.3–0.5 units)
- Two ring systems (naphthalene, indole) → base pIC50 ~0.5 higher than benzene
- Add Gaussian noise σ=0.2 to simulate SAR noise

### Expected split outcome (greedy, test_fraction=0.25)
- Train: benzene (12) + pyrimidine (6) + benzimidazole (6) = 24 compounds
- Test: indole (7) + quinoline (7) = 14 compounds (~31% but closest to 0.25 with integer groups)
- Scaffold split R²: ~0.3–0.6 (model struggles on unseen scaffolds)
- Random split R²: ~0.6–0.85 (scaffold overlap inflates metric)

---

## 8. Verification Checklist

```bash
python main.py --input data/compounds.csv

# Expected:
# - Split comparison printed: scaffold_split R² < random_split R²
# - qsar_results.csv: compound_name + scaffold + split_assignment + y_true + y_pred_scaffold + y_pred_random
# - split_comparison.png: 2 parity plots; scaffold split shows wider scatter
# - feature_importance.png: top-20 bits with non-zero importance
# - Console: "Delta R² (random - scaffold): X.XX — random split overestimates performance"
```

---

## 9. Risks / Assumptions / Next Step

**Risks:**
- With only ~45 compounds, RF can overfit train set in both split strategies.
  Mitigate: limit n_estimators=200, min_samples_leaf=2 (no tuning needed; fixed hyperparams)
- Scaffold split greedy assignment: if one scaffold group is very large (e.g., 20 compounds),
  it may be impossible to reach test_fraction=0.25 without it. Strategy: always assign
  the single largest scaffold to train; fill test with remaining small groups.
- With synthetic pIC50 data (by design correlated), random split R² may be near 0.9+
  and scaffold split near 0.2–0.4. This is a feature, not a bug — it makes the lesson clear.
- ECFP4 bit labels ("bit_0042") are not chemically interpretable without substructure
  lookup; document this explicitly.

**Assumptions:**
- radius=2, nBits=2048, useChirality=False is standard ECFP4; no parameter sweep needed
- Greedy split uses sorted(scaffold_groups, key=len) ascending, filling test first
- Feature importances from RF are impurity-based (Gini); documented as such
- No cross-validation (single scaffold split is sufficient to demonstrate the concept)

---

## 10. Actual Results (v1.1)

### Run command
```bash
PYTHONUTF8=1 python main.py --input data/compounds.csv
```

### Scaffold distribution (greedy split, test_fraction=0.25, N=45)
| Scaffold | n | Assigned |
|---|---|---|
| c1ccccc1 (benzene) | 12 | TRAIN (forced — largest) |
| c1ccc2ccccc2c1 (naphthalene) | 7 | TRAIN |
| c1ccc2[nH]ccc2c1 (indole) | 7 | TEST (first by hetero sort) |
| c1ccc2ncccc2c1 (quinoline) | 7 | TEST |
| c1cnccn1 (pyrimidine) | 6 | TRAIN |
| c1ccc2[nH]cnc2c1 (benzimidazole) | 6 | TRAIN |

Train=31, Test=14, achieved test fraction=0.311.

### Metrics
| Metric | Scaffold Split | Random Split |
|---|---|---|
| R² | **−1.654** | 0.569 |
| MAE | 0.797 | 0.417 |
| RMSE | 0.821 | 0.498 |
| ΔR² (random − scaffold) | **+2.223** | — |

### Key insights
- Scaffold split R² = −1.65: model is **worse than predicting the mean** on unseen
  scaffolds (indole/quinoline). This is the correct and expected result — ECFP4
  fingerprints for benzene-family compounds do not transfer well to fused N-heterocycles.
- Random split R² = 0.57: partial overlap of scaffold families in train/test inflates R².
- ΔR² = +2.22: dramatic demonstration of the prospective performance gap.

### Deviations from plan
- **Scaffold split R² < 0** (spec expected 0.3–0.6): the actual gap is larger than
  anticipated. This is a feature, not a bug — with only 31 train compounds and 6
  scaffold families, the EWG/EDG signal learned on benzene scaffolds doesn't transfer
  to indole/quinoline topology. The lesson is even clearer.
- **CF3 SMILES fix**: `C(F)(F)Fc1...` chains the third F to the ring (not a CF3 branch).
  Fixed to `FC(F)(F)c1...` for ind_006, quin_006, bzim_006.
- **naphthalene SMILES**: spec gave `c1ccc2cccccc2c1` (11 atoms, wrong) for naph_001_H.
  Fixed to `c1ccc2ccccc2c1` (10 atoms, correct naphthalene) before writing compounds.csv.

**Next step:** Phase 22 — Virtual Screening Pipeline. Given a target protein pocket
(represented as a pharmacophore constraint SMARTS), screen a library of compounds
by substructure + physicochemical filters and rank survivors by docking score proxy.
