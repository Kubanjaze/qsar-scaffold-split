import sys
import os

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_compounds(path: str) -> "pd.DataFrame":
    """
    Requires columns: compound_name, smiles, pic50.
    Coerces pic50 to numeric; drops NaN pic50 rows.
    Parses SMILES; drops invalid/empty.
    Prints load counts. Returns df with compound_name, smiles, pic50, mol.
    """
    df = pd.read_csv(path)
    required = {"compound_name", "smiles", "pic50"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required}")

    n_total = len(df)

    df["pic50"] = pd.to_numeric(df["pic50"], errors="coerce")
    n_missing_pic50 = df["pic50"].isna().sum()
    df = df.dropna(subset=["pic50"]).reset_index(drop=True)

    rows = []
    n_invalid_smiles = 0
    for _, row in df.iterrows():
        smi = str(row["smiles"]).strip() if pd.notna(row["smiles"]) else ""
        if not smi:
            n_invalid_smiles += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid_smiles += 1
            print(f"  WARNING: invalid SMILES skipped: {row['compound_name']} ({smi!r})")
            continue
        rows.append({
            "compound_name": row["compound_name"],
            "smiles": smi,
            "pic50": float(row["pic50"]),
            "mol": mol,
        })

    n_valid = len(rows)
    print(
        f"Loaded {n_total} rows, {n_valid} valid, "
        f"{n_invalid_smiles} invalid/empty SMILES skipped, "
        f"{n_missing_pic50} missing pic50 dropped"
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scaffold extraction
# ---------------------------------------------------------------------------

def compute_scaffolds(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add scaffold_smiles (Murcko canonical) column to df."""
    scaffolds = []
    for _, row in df.iterrows():
        mol = row["mol"]
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        if sc and sc.GetNumAtoms() > 0:
            smi = Chem.MolToSmiles(sc, canonical=True)
        else:
            smi = "acyclic"
        scaffolds.append(smi)
    df = df.copy()
    df["scaffold_smiles"] = scaffolds
    return df


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------

def featurize_ecfp4(df: "pd.DataFrame", radius: int, nbits: int) -> "np.ndarray":
    """Return ECFP4 bit vectors as float64 array of shape (N, nbits)."""
    rows = []
    for _, row in df.iterrows():
        fp = AllChem.GetMorganFingerprintAsBitVect(
            row["mol"], radius=radius, nBits=nbits, useChirality=False
        )
        rows.append(np.array(fp, dtype=np.float64))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Scaffold split
# ---------------------------------------------------------------------------

def _hetero_count(scaffold_smiles: str) -> int:
    """Count non-carbon heavy atoms in scaffold (for tie-breaking)."""
    if scaffold_smiles == "acyclic":
        return 0
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 6)


def scaffold_split(
    df: "pd.DataFrame",
    test_fraction: float,
    random_state: int = 42,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Deterministic greedy scaffold split.
    1. Force the single largest scaffold group into TRAIN.
    2. Sort remaining groups by (size desc, hetero_count desc, smiles asc).
    3. Greedily add groups to TEST until len(test) >= ceil(test_fraction * N).
    4. Rest to TRAIN.
    Returns train_idx, test_idx as sorted numpy arrays.
    """
    groups: "dict[str, list[int]]" = defaultdict(list)
    for i, row in df.iterrows():
        groups[row["scaffold_smiles"]].append(i)

    # 1. Largest group → forced train
    largest_scaffold = max(groups, key=lambda s: len(groups[s]))
    train_idx: list = list(groups[largest_scaffold])

    remaining = {s: idxs for s, idxs in groups.items() if s != largest_scaffold}

    # 2. Sort remaining
    sorted_remaining = sorted(
        remaining.items(),
        key=lambda x: (-len(x[1]), -_hetero_count(x[0]), x[0]),
    )

    # 3. Greedy fill test
    target = math.ceil(test_fraction * len(df))
    test_idx: list = []
    test_done = False

    for scaffold, idxs in sorted_remaining:
        if not test_done:
            test_idx.extend(idxs)
            if len(test_idx) >= target:
                test_done = True
        else:
            train_idx.extend(idxs)

    # Print scaffold distribution
    print(f"\nScaffold split (target_test_frac={test_fraction:.2f}, "
          f"target_N_test={target}, N={len(df)}):")
    for scaffold, idxs in sorted(groups.items(), key=lambda x: -len(x[1])):
        assignment = "TEST" if any(i in test_idx for i in idxs) else "TRAIN"
        label = scaffold[:40] + ("..." if len(scaffold) > 40 else "")
        print(f"  [{assignment:5s}] n={len(idxs):3d}  {label}")
    achieved = len(test_idx) / len(df)
    print(f"  -> train={len(train_idx)}, test={len(test_idx)}, "
          f"achieved_test_frac={achieved:.3f}\n")

    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: "np.ndarray", y_pred: "np.ndarray") -> dict:
    """Compute R², MAE, RMSE on test set."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)}


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: "np.ndarray",
    y: "np.ndarray",
    train_idx: "np.ndarray",
    test_idx: "np.ndarray",
    label: str,
) -> dict:
    """Fit RF on train split, evaluate on test split."""
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    return {
        "label": label,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "test_idx": test_idx,
        "train_idx": train_idx,
        "feature_importances": model.feature_importances_,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_split_comparison(
    scaffold_results: dict,
    random_results: dict,
    output_path: str,
) -> None:
    """1x2 parity plot: scaffold split (left) vs random split (right)."""
    all_vals = (
        list(scaffold_results["y_test"])
        + list(scaffold_results["y_pred"])
        + list(random_results["y_test"])
        + list(random_results["y_pred"])
    )
    lo = min(all_vals) - 0.3
    hi = max(all_vals) + 0.3

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)

    for ax, results in zip(axes[0], [scaffold_results, random_results]):
        y_test = results["y_test"]
        y_pred = results["y_pred"]
        m = results["metrics"]
        label = results["label"]
        n = len(y_test)

        if n < 2:
            ax.text(0.5, 0.5, "Not enough test points", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(label)
            continue

        ax.scatter(y_test, y_pred, alpha=0.8, edgecolors="k", linewidths=0.4,
                   s=55, color="#1976d2" if "scaffold" in label else "#e64a19", zorder=3)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.6, label="y = x")

        stats_text = (
            f"R\u00b2 = {m['r2']:.3f}\n"
            f"MAE = {m['mae']:.3f}\n"
            f"RMSE = {m['rmse']:.3f}\n"
            f"N\u209c\u2091\u209b\u209c = {n}"
        )
        ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual pIC50")
        ax.set_ylabel("Predicted pIC50")
        ax.set_title(f"{label.replace('_', ' ').title()} Split")
        ax.set_aspect("equal")

    plt.suptitle("QSAR Parity Plot: Scaffold Split vs Random Split", fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_feature_importance(
    importances: "np.ndarray",
    top_n: int,
    output_path: str,
) -> None:
    """Horizontal bar chart: top-N ECFP4 bit importances (scaffold-split model)."""
    top_idx = np.argsort(importances)[-top_n:][::-1]
    top_vals = importances[top_idx]
    top_labels = [f"bit_{i:04d}" for i in top_idx]

    # Sort ascending for barh display
    order = np.argsort(top_vals)
    top_vals = top_vals[order]
    top_labels = [top_labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(top_labels, top_vals, color="#5c6bc0", edgecolor="k", linewidth=0.3)
    ax.set_xlabel("Feature Importance (impurity-based, Gini)")
    ax.set_title(
        f"Top-{top_n} ECFP4 Bit Importances\n"
        "(scaffold-split model; bit indices are not directly chemically interpretable)"
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QSAR RandomForest with scaffold split vs random split comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Compounds CSV (compound_name, smiles, pic50)")
    parser.add_argument("--test-fraction", type=float, default=0.25,
                        help="Desired test fraction for scaffold split (integer groups may overshoot)")
    parser.add_argument("--top-n", type=int, default=20, help="Top-N bits in importance plot")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius")
    parser.add_argument("--nbits", type=int, default=2048, help="Morgan fingerprint bits")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    print("Loading compounds...")
    df = load_compounds(args.input)
    if len(df) < 4:
        print("Not enough valid compounds to run. Exiting.")
        return

    # Scaffolds
    print("Computing Murcko scaffolds...")
    df = compute_scaffolds(df)

    # Featurize
    print(f"Featurizing (ECFP4 radius={args.radius}, nBits={args.nbits})...")
    X = featurize_ecfp4(df, radius=args.radius, nbits=args.nbits)
    y = df["pic50"].values

    # Scaffold split
    print("Computing scaffold split...")
    sc_train_idx, sc_test_idx = scaffold_split(df, args.test_fraction)

    # Random split — same N_test as scaffold split for fair comparison
    n_test = len(sc_test_idx)
    rand_test_frac = n_test / len(df)
    all_idx = np.arange(len(df))
    rand_train_idx, rand_test_idx = train_test_split(
        all_idx, test_size=rand_test_frac, random_state=42
    )
    print(f"Random split: train={len(rand_train_idx)}, test={len(rand_test_idx)}")

    # Train + evaluate both
    print("Training scaffold-split model...")
    sc_results = train_and_evaluate(X, y, sc_train_idx, sc_test_idx, label="scaffold")

    print("Training random-split model...")
    rand_results = train_and_evaluate(X, y, rand_train_idx, rand_test_idx, label="random")

    # Print metrics
    sc_m = sc_results["metrics"]
    rand_m = rand_results["metrics"]
    delta_r2 = rand_m["r2"] - sc_m["r2"]
    print(f"\n{'Metric':<8} {'Scaffold':>12} {'Random':>12}")
    print(f"{'R2':<8} {sc_m['r2']:>12.4f} {rand_m['r2']:>12.4f}")
    print(f"{'MAE':<8} {sc_m['mae']:>12.4f} {rand_m['mae']:>12.4f}")
    print(f"{'RMSE':<8} {sc_m['rmse']:>12.4f} {rand_m['rmse']:>12.4f}")
    print(f"\nDelta R\u00b2 (random \u2212 scaffold): {delta_r2:+.4f}"
          f"  \u2192  random split {'over' if delta_r2 > 0 else 'under'}estimates performance")

    # Build results CSV
    result_df = df[["compound_name", "smiles", "scaffold_smiles", "pic50"]].copy()

    split_sc = np.full(len(df), "train", dtype=object)
    split_sc[sc_test_idx] = "test"
    result_df["split_scaffold"] = split_sc

    pred_sc = np.full(len(df), np.nan)
    pred_sc[sc_test_idx] = sc_results["y_pred"]
    result_df["y_pred_scaffold"] = pred_sc

    split_rand = np.full(len(df), "train", dtype=object)
    split_rand[rand_test_idx] = "test"
    result_df["split_random"] = split_rand

    pred_rand = np.full(len(df), np.nan)
    pred_rand[rand_test_idx] = rand_results["y_pred"]
    result_df["y_pred_random"] = pred_rand

    csv_path = os.path.join(args.output_dir, "qsar_results.csv")
    result_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Saved: {csv_path}")

    # Plots
    print("Plotting parity comparison...")
    plot_split_comparison(
        sc_results, rand_results,
        os.path.join(args.output_dir, "split_comparison.png"),
    )

    print("Plotting feature importance...")
    plot_feature_importance(
        sc_results["feature_importances"], args.top_n,
        os.path.join(args.output_dir, "feature_importance.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
