# flake8: noqa

"""Multi-dataset evaluation for Semantic Bagging Forest.

This script evaluates the bagging ensemble on multiple MoleculeNet-style
classification datasets included in `data/`.

By default it runs one representative task per dataset to keep runtime practical.
Use `--all-tasks` to evaluate all tasks for multi-task datasets (Tox21, SIDER).
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


sys.path.append(str(Path(__file__).parent.parent))

from src.ontology.molecule_ontology import MoleculeOntology
from src.ontology.smiles_converter import MolecularFeatureExtractor
from src.sdt.logic_forest import SemanticBaggingForest


def _safe_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name))
    return s[:80].strip("_") or "task"


def _normalize_binary_labels(series: pd.Series) -> pd.Series:
    """Normalize common binary label encodings.

    - drops missing values
    - treats -1 as missing (common in some datasets)
    - supports {-1, 1} by mapping to {0, 1} after dropping missing
    """
    y = pd.to_numeric(series, errors="coerce")
    # In some MoleculeNet dumps, -1 is used as missing.
    y = y.replace(-1, np.nan)
    y = y.dropna()

    uniq = set(y.unique().tolist())
    if uniq == {1.0, 0.0}:
        return y.astype(int)
    if uniq == {1.0, -1.0}:
        # If -1 survived (shouldn't), map it.
        return y.replace(-1, 0).astype(int)
    if uniq == {1.0} or uniq == {0.0}:
        return y.astype(int)

    # Best-effort: if values are already 0/1-like
    if uniq.issubset({0.0, 1.0}):
        return y.astype(int)

    raise ValueError(f"Non-binary labels found: {sorted(uniq)}")


def populate_ontology(
    onto: MoleculeOntology,
    extractor: MolecularFeatureExtractor,
    df: pd.DataFrame,
    smiles_col: str,
    label_col: str,
    subset_name: str,
):
    instances = []
    labels = []

    for idx, row in df.iterrows():
        try:
            smi = row[smiles_col]
            feats = extractor.extract_features(smi)
            mol_id = f"Mol_{subset_name}_{idx}"
            label_val = int(row[label_col])
            inst = onto.add_molecule_instance(mol_id, feats, label=label_val)
            instances.append(inst)
            labels.append(label_val)
        except Exception:
            # Skip invalid SMILES / feature extraction failures.
            continue

    return instances, labels


def evaluate_task(
    dataset_key: str,
    dataset_name: str,
    csv_path: str,
    smiles_col: str,
    label_col: str,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    sample_size: int,
    test_size: float,
    random_state: int,
):
    df = pd.read_csv(csv_path)

    if smiles_col not in df.columns:
        raise ValueError(f"Missing smiles column '{smiles_col}' in {csv_path}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {csv_path}")

    # Normalize labels (drop missing)
    y_norm = _normalize_binary_labels(df[label_col])
    df = df.loc[y_norm.index].copy()
    df[label_col] = y_norm

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col] if df[label_col].nunique() > 1 else None,
    )

    onto_path = Path("ontology") / f"temp_bagging_{dataset_key}_{_safe_name(label_col)}.owl"
    if onto_path.exists():
        onto_path.unlink()

    onto = MoleculeOntology(str(onto_path))
    extractor = MolecularFeatureExtractor()

    train_instances, _ = populate_ontology(
        onto,
        extractor,
        train_df,
        smiles_col,
        label_col,
        subset_name="Train",
    )
    test_instances, test_labels = populate_ontology(
        onto,
        extractor,
        test_df,
        smiles_col,
        label_col,
        subset_name="Test",
    )

    # If feature extraction filtered too much, bail out.
    if len(train_instances) < max(min_samples_split, 50) or len(test_instances) < 50:
        return {
            "dataset": dataset_name,
            "task": label_col,
            "n_train": len(train_instances),
            "n_test": len(test_instances),
            "auc": np.nan,
            "acc": np.nan,
            "note": "too_few_valid_instances",
        }

    forest = SemanticBaggingForest(
        onto,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        verbose=False,
    )
    forest.fit(train_instances)

    probs = forest.predict_proba(test_instances)
    preds = forest.predict(test_instances)

    acc = accuracy_score(test_labels, preds)
    try:
        auc = roc_auc_score(test_labels, probs)
    except ValueError:
        auc = 0.5

    return {
        "dataset": dataset_name,
        "task": label_col,
        "n_train": len(train_instances),
        "n_test": len(test_instances),
        "auc": float(auc),
        "acc": float(acc),
        "note": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Semantic Bagging Forest across datasets"
    )
    parser.add_argument("--n-estimators", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Evaluate all tasks for multi-task datasets (Tox21, SIDER).",
    )
    parser.add_argument(
        "--out",
        default=str(Path("output") / "bagging_forest_benchmark.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    datasets = [
        {
            "key": "bbbp",
            "name": "BBBP",
            "path": "data/bbbp/BBBP.csv",
            "smiles": "smiles",
            "tasks": ["p_np"],
        },
        {
            "key": "bace",
            "name": "BACE",
            "path": "data/bace/bace.csv",
            "smiles": "smiles",
            "tasks": ["Class"],
        },
        {
            "key": "clintox",
            "name": "ClinTox",
            "path": "data/clintox/clintox.csv",
            "smiles": "smiles",
            "tasks": ["CT_TOX"],
        },
        {
            "key": "hiv",
            "name": "HIV",
            "path": "data/hiv/HIV.csv",
            "smiles": "smiles",
            "tasks": ["HIV_active"],
        },
        {
            "key": "tox21",
            "name": "Tox21",
            "path": "data/tox21/tox21.csv",
            "smiles": "smiles",
            "tasks": [
                "NR-AR",
                "NR-AR-LBD",
                "NR-AhR",
                "NR-Aromatase",
                "NR-ER",
                "NR-ER-LBD",
                "NR-PPAR-gamma",
                "SR-ARE",
                "SR-ATAD5",
                "SR-HSE",
                "SR-MMP",
                "SR-p53",
            ],
            "default_task": "SR-p53",
        },
        {
            "key": "sider",
            "name": "SIDER",
            "path": "data/sider/sider.csv",
            "smiles": "smiles",
            "tasks": None,  # determined from header
            "default_task": "Hepatobiliary disorders",
        },
    ]

    results = []
    for ds in datasets:
        tasks = ds.get("tasks")
        if tasks is None:
            # SIDER: all columns except smiles
            df_cols = pd.read_csv(ds["path"], nrows=1).columns.tolist()
            tasks = [c for c in df_cols if c != ds["smiles"]]

        if not args.all_tasks:
            # Use one representative task for multi-task datasets.
            if "default_task" in ds:
                tasks = [ds["default_task"]]

        for task in tasks:
            try:
                res = evaluate_task(
                    dataset_key=ds["key"],
                    dataset_name=ds["name"],
                    csv_path=ds["path"],
                    smiles_col=ds["smiles"],
                    label_col=task,
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_split=args.min_samples_split,
                    min_samples_leaf=args.min_samples_leaf,
                    sample_size=args.sample_size,
                    test_size=args.test_size,
                    random_state=args.random_state,
                )
            except Exception as e:
                res = {
                    "dataset": ds["name"],
                    "task": task,
                    "n_train": 0,
                    "n_test": 0,
                    "auc": np.nan,
                    "acc": np.nan,
                    "note": f"error: {e}",
                }

            results.append(res)
            print(
                f"{res['dataset']}/{res['task']}: "
                f"AUC={res['auc']}, ACC={res['acc']} "
                f"(train={res['n_train']}, test={res['n_test']}) {res['note']}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
