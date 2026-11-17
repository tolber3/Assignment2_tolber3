#!/usr/bin/env python3
"""Train a Gaussian Process classifier on formed ML datasets.

The inputs are the ML-ready tables written by ``BML/scripts/form_sets.py``
(``<dataset_name>_dS.parquet``/``.csv`` or ``<dataset_name>_full.parquet``/``.csv``).
A YAML config chooses the formed dataset directory, train/test dataset names,
and optional training modes. The default mode trains on compact dS features and
collapses raw bacteria labels into a binary ``Bacteria`` vs ``Particle`` task.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, RBF
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

try:
    from aim import Figure as AimFigure
    from aim import Run, Text
except Exception:  # Aim is optional at import time so non-tracking runs still work.
    AimFigure = None
    Run = None
    Text = None


# Metadata columns are useful for audit/prediction outputs, but should not be
# used as model features because they leak labels, source identity, or row IDs.
EXCLUDED_FEATURE_COLUMNS = {
    "label",
    "target",
    "group_number",
    "dataset_name",
    "dataset_subdir",
    "sample_stream",
    "device",
    "ext",
}
EXCLUDED_FEATURE_SUFFIXES = (
    "_set_number",
    "_peak_number",
    "_pair_id",
    "_pair_role",
    "_peak_time",
)
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "gpr_config.yaml"
FREQUENCY_COLUMN_RE = re.compile(r"^(?P<frequency>\d+(?:\.\d+)?)GHz(?:_|$)")


def frequency_key(value: Any) -> str:
    """Normalize GHz values so config entries like 1, 1.0, and '1GHz' match."""
    original_value = value
    if isinstance(value, str):
        value = value.strip().lower().replace("ghz", "").strip()
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid frequency value {original_value!r}; use GHz numbers or 'all'.") from exc
    if not np.isfinite(numeric_value):
        raise ValueError(f"Invalid frequency value {original_value!r}; frequency must be finite.")
    return f"{numeric_value:.12g}"


def normalize_frequency_selection(freqs: Any) -> set[str] | None:
    """Return normalized selected frequency keys, or None when all frequencies are enabled."""
    if freqs is None:
        return None
    if isinstance(freqs, str):
        raw_values = [freqs]
    elif isinstance(freqs, (list, tuple, set)):
        raw_values = list(freqs)
    else:
        raw_values = [freqs]

    if not raw_values:
        return None

    selected: set[str] = set()
    for raw_value in raw_values:
        if isinstance(raw_value, str) and raw_value.strip().lower() in {"all", "*"}:
            return None
        selected.add(frequency_key(raw_value))
    return selected


def frequency_summary(freqs: Any) -> str | list[float]:
    """Make the configured frequency selection compact and JSON-friendly."""
    selected = normalize_frequency_selection(freqs)
    if selected is None:
        return "all"
    return [float(key) for key in sorted(selected, key=float)]


def column_frequency_key(column: str) -> str | None:
    """Return the normalized GHz key for a frequency-prefixed feature column."""
    match = FREQUENCY_COLUMN_RE.match(str(column))
    if not match:
        return None
    return frequency_key(match.group("frequency"))


def selected_feature_frequencies(feature_columns: list[str]) -> list[float]:
    """List the unique GHz values represented by the selected model features."""
    keys = {key for column in feature_columns if (key := column_frequency_key(column)) is not None}
    return [float(key) for key in sorted(keys, key=float)]


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config and store its resolved path for logging/reproducibility."""
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config["_config_path"] = str(Path(path).resolve())
    return config


def get_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge optional ``training`` config values with safe defaults.

    Important modes:
    - ``table``: ``dS`` uses only magnitude/angle delta features; 
                ``full`` can use additional numeric columns if ``feature_regex`` is adjusted.
    - ``target_mode``: ``binary_bacteria_particle`` maps any bacteria-like label to ``Bacteria`` and particle-like labels to ``Particle``; 
                        ``raw`` keeps the exact label strings and prompts to remap unseen test labels.
    - ``drop_ext``: removes rows generated by peak duplication during set formation when set to true.
    - ``freqs``: ``all`` keeps every GHz feature; a list like ``[1.5, "2GHz"]`` keeps only those frequency features.
    - ``scaler``: selects ``standard`` or ``robust`` scaling when ``standardize`` is true.
    """
    training = dict(config.get("training", {}))
    training.setdefault("table", "dS")
    training.setdefault("target_column", "label")
    training.setdefault("target_mode", "binary_bacteria_particle")
    training.setdefault("drop_ext", False)
    training.setdefault("freqs", "all")
    training.setdefault("keep_non_frequency_features", False)
    training.setdefault("test_size", 0.25)
    training.setdefault("feature_regex", None)
    training.setdefault("impute_strategy", "median")
    training.setdefault("standardize", True)
    training.setdefault("scaler", "standard")
    training.setdefault("n_restarts_optimizer", 0)
    training.setdefault("max_iter_predict", 100)
    training.setdefault("multi_class", "one_vs_rest")
    training.setdefault("max_train_rows", None)
    training.setdefault("aim_enabled", True)
    training.setdefault("save_predictions", True)
    feature_importance_cfg = dict(training.get("feature_importance", {}))
    feature_importance_cfg.setdefault("enabled", True)
    feature_importance_cfg.setdefault("split", "test")
    feature_importance_cfg.setdefault("scoring", "f1_macro")
    feature_importance_cfg.setdefault("n_repeats", 5)
    feature_importance_cfg.setdefault("max_rows", None)
    feature_importance_cfg.setdefault("top_n_aim", 50)
    training["feature_importance"] = feature_importance_cfg
    training.setdefault("kernel", {"type": "rbf", "constant_value": 1.0, "length_scale": 1.0})
    return training


def processed_dataset_dir(config: dict[str, Any]) -> Path:
    """Resolve the directory containing formed ML datasets from the config."""
    paths = config.get("paths", {})
    form_sets = config.get("form_sets", {})
    scratch_root = Path(paths.get("scratch_root", ".")).expanduser()
    processed_subdir = form_sets.get("dataset_processed_subdir", paths.get("dataset_processed_subdir"))
    if not processed_subdir:
        raise ValueError("dataset_processed_subdir must be defined under form_sets or paths")
    return scratch_root / processed_subdir


def output_dirs(config: dict[str, Any], run_name: str) -> tuple[Path, Path, Path]:
    """Create checkpoint, log, and Aim repository directories for one run."""
    paths = config.get("paths", {})
    scratch_root = Path(paths.get("scratch_root", ".")).expanduser()
    checkpoint_root = scratch_root / paths.get("checkpoints_subdir", "checkpoints/Bact")
    log_root = scratch_root / paths.get("logs_subdir", "logs/Bact")
    aim_repo = scratch_root / paths.get("aim_repo_subdir", "aim-repos/Bact")
    checkpoint_dir = checkpoint_root / run_name
    log_dir = log_root / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    aim_repo.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, log_dir, aim_repo


def read_table(path: Path) -> pd.DataFrame:
    """Load a formed dataset table from parquet or CSV."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset table type: {path}")


def find_dataset_table(dataset_dir: Path, dataset_name: str, table: str) -> Path:
    """Find the parquet/CSV file for a named formed dataset and table type."""
    parquet_path = dataset_dir / f"{dataset_name}_{table}.parquet"
    csv_path = dataset_dir / f"{dataset_name}_{table}.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Could not find {dataset_name}_{table}.parquet or .csv in {dataset_dir}")


def load_named_datasets(dataset_dir: Path, dataset_names: list[str], table: str, split_name: str) -> pd.DataFrame:
    """Load and concatenate all dataset names listed for a train/test split."""
    frames = []
    for dataset_name in dataset_names:
        table_path = find_dataset_table(dataset_dir, dataset_name, table)
        df = read_table(table_path)
        df["dataset_name"] = dataset_name
        df["split_source"] = split_name
        df["source_table_path"] = str(table_path)
        frames.append(df)
        print(f"Loaded {split_name} dataset {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns from {table_path}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def normalize_label(value: Any, mode: str, label_map: dict[str, str] | None = None) -> str:
    """Convert raw labels into the configured target label space.

    ``label_map`` is applied first for exact overrides. ``raw`` mode then keeps
    the mapped string unchanged, while ``binary_bacteria_particle`` uses simple
    substring rules to make bacteria-vs-particle experiments robust across
    labels like ``Bact_K12``, ``Bact_K12B``, and ``Particle``.
    """
    raw = "" if pd.isna(value) else str(value)
    if label_map and raw in label_map:
        raw = str(label_map[raw])
    mode = (mode or "raw").lower()
    if mode in {"raw", "none", "identity"}:
        return raw
    if mode in {"binary_bacteria_particle", "binary", "bact_particle"}:
        lowered = raw.lower()
        if "bact" in lowered or "bacteria" in lowered:
            return "Bacteria"
        if "particle" in lowered or "ptcl" in lowered:
            return "Particle"
        return raw
    raise ValueError(f"Unsupported target_mode: {mode}")


def prompt_test_target_remapping(unseen_targets: list[str], train_targets: list[str]) -> dict[str, str]:
    """Ask the user to map each unseen test target to an existing train target."""
    remapping: dict[str, str] = {}
    try:
        for target in unseen_targets:
            print(f"\nWARNING: Test target {target!r} is not present in the training targets.", flush=True)
            print("Remapping will use the selected training target as this test target's evaluation label.", flush=True)
            while True:
                answer = input(f"Continue by remapping test target {target!r}? [y/n]: ").strip().lower()
                if answer in {"n", "no"}:
                    raise SystemExit("Training cancelled: test target remapping was declined.")
                if answer in {"y", "yes"}:
                    break
                print("Please enter y or n.", flush=True)

            print("Available training targets:", flush=True)
            for number, train_target in enumerate(train_targets, start=1):
                print(f"  {number}. {train_target}", flush=True)
            while True:
                selection = input(f"Select the training target number for {target!r} (1-{len(train_targets)}): ").strip()
                try:
                    number = int(selection)
                except ValueError:
                    number = 0
                if 1 <= number <= len(train_targets):
                    remapping[target] = train_targets[number - 1]
                    print(f"Remapping test target {target!r} to {remapping[target]!r}.", flush=True)
                    break
                print(f"Please enter a number from 1 to {len(train_targets)}.", flush=True)
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("Training cancelled: test target remapping input was interrupted.") from None
    return remapping


def prepare_targets(train_df: pd.DataFrame, test_df: pd.DataFrame | None, training_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None, LabelEncoder]:
    """Normalize and encode targets, prompting for unseen test target mappings."""
    target_column = training_cfg["target_column"]
    if target_column not in train_df.columns:
        raise ValueError(f"Target column {target_column!r} is missing from train data")
    if test_df is not None and not test_df.empty and target_column not in test_df.columns:
        raise ValueError(f"Target column {target_column!r} is missing from test data")

    mode = training_cfg.get("target_mode", "binary_bacteria_particle")
    label_map = training_cfg.get("label_map", {}) or {}
    y_train_text = train_df[target_column].map(lambda value: normalize_label(value, mode, label_map)).to_numpy()
    y_test_text = None if test_df is None or test_df.empty else test_df[target_column].map(lambda value: normalize_label(value, mode, label_map)).to_numpy()

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_text)
    if len(encoder.classes_) < 2:
        raise ValueError(f"Need at least two train classes after target mapping; found {encoder.classes_.tolist()}")

    y_test = None
    test_target_remap: dict[str, str] = {}
    if y_test_text is not None:
        unseen = sorted(set(y_test_text) - set(encoder.classes_))
        if unseen:
            test_target_remap = prompt_test_target_remapping(unseen, encoder.classes_.tolist())
            y_test_text = np.array([test_target_remap.get(target, target) for target in y_test_text])
        y_test = encoder.transform(y_test_text)
    training_cfg["test_target_remap"] = test_target_remap
    return y_train, y_test, encoder


def select_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame | None, training_cfg: dict[str, Any]) -> list[str]:
    """Choose common numeric model features from train/test tables.

    For the default ``dS`` table this automatically selects columns containing
    ``_dS``. Metadata and ID-like columns are excluded even if numeric. When
    ``training.freqs`` is a frequency list, only matching GHz-prefixed features
    are included; ``all`` or ``["all"]`` keeps every available frequency.
    """
    feature_regex = training_cfg.get("feature_regex")
    table = str(training_cfg.get("table", "dS")).lower()
    if feature_regex is None and table == "ds":
        feature_regex = r"_dS"

    selected_freqs = normalize_frequency_selection(training_cfg.get("freqs", "all"))
    keep_non_frequency = bool(training_cfg.get("keep_non_frequency_features", False))
    common_columns = set(train_df.columns)
    if test_df is not None and not test_df.empty:
        common_columns &= set(test_df.columns)

    numeric_columns = []
    available_freqs_after_filters: set[str] = set()
    for column in train_df.columns:
        if column not in common_columns:
            continue
        if column in EXCLUDED_FEATURE_COLUMNS:
            continue
        if column.endswith(EXCLUDED_FEATURE_SUFFIXES):
            continue
        if feature_regex and re.search(str(feature_regex), str(column)) is None:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[column]):
            continue

        column_freq = column_frequency_key(column)
        if column_freq is not None:
            available_freqs_after_filters.add(column_freq)

        if selected_freqs is not None:
            if column_freq is None and not keep_non_frequency:
                continue
            if column_freq is not None and column_freq not in selected_freqs:
                continue

        numeric_columns.append(column)

    if selected_freqs is not None:
        missing_freqs = selected_freqs - available_freqs_after_filters
        if missing_freqs:
            requested = [float(key) for key in sorted(selected_freqs, key=float)]
            missing = [float(key) for key in sorted(missing_freqs, key=float)]
            available = [float(key) for key in sorted(available_freqs_after_filters, key=float)]
            raise ValueError(
                "training.freqs requested frequencies without matching numeric feature columns after "
                "train/test/common-column and feature_regex filtering. "
                f"requested={requested}, missing={missing}, available={available}"
            )

    if not numeric_columns:
        raise ValueError("No numeric feature columns found. Check training.table, training.feature_regex, or training.freqs.")
    return numeric_columns


def maybe_drop_ext(df: pd.DataFrame, drop_ext: bool) -> pd.DataFrame:
    """Optionally remove extended rows created by cyclic peak duplication."""
    if not drop_ext or "ext" not in df.columns:
        return df
    before = len(df)
    filtered = df[~df["ext"].fillna(False).astype(bool)].copy()
    print(f"Dropped ext rows: {before - len(filtered)} of {before}")
    return filtered


def maybe_limit_train_rows(df: pd.DataFrame, y: np.ndarray, max_rows: int | None, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Optionally subsample training rows for faster Gaussian Process fitting."""
    if not max_rows or len(df) <= max_rows:
        return df, y
    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    limited_df, _, limited_y, _ = train_test_split(
        df,
        y,
        train_size=max_rows,
        stratify=stratify,
        random_state=seed,
    )
    return limited_df.reset_index(drop=True), limited_y


def describe_class_rows(metadata_df: pd.DataFrame | None, row_indices: np.ndarray) -> str:
    """Return source dataset details for rows that prevent a class-ratio split."""
    if metadata_df is None or metadata_df.empty:
        return "Source row details unavailable."

    useful_columns = [
        "dataset_name",
        "dataset_subdir",
        "source_table_path",
        "split_source",
        "label",
        "group_number",
        "sample_stream",
        "ext",
    ]
    available_columns = [column for column in useful_columns if column in metadata_df.columns]
    if not available_columns:
        return "Source row details unavailable: no dataset metadata columns were found."

    rows = metadata_df.iloc[row_indices][available_columns]
    details = []
    for row_number, row in rows.iterrows():
        values = ", ".join(f"{column}={row[column]!r}" for column in available_columns)
        details.append(f"row {row_number}: {values}")
    return "Source row(s): " + "; ".join(details)


def per_class_train_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float,
    seed: int,
    metadata_df: pd.DataFrame | None = None,
    class_names: np.ndarray | list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    """Split each class independently so ``test_size`` is applied per class.

    For example, ``test_size=0.5`` assigns approximately half of each class to
    the test split instead of only preserving the overall class proportions.
    Classes with at least two rows keep at least one train row and one test row.
    """
    test_size = float(test_size)
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"training.test_size must be between 0 and 1 for class-ratio splitting; got {test_size}")

    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    split_counts: dict[str, dict[str, int]] = {}

    for class_id in sorted(np.unique(y).tolist()):
        class_indices = np.flatnonzero(y == class_id)
        class_count = len(class_indices)
        if class_count < 2:
            class_label = (
                str(class_names[int(class_id)])
                if class_names is not None and int(class_id) < len(class_names)
                else str(class_id)
            )
            row_details = describe_class_rows(metadata_df, class_indices)
            raise ValueError(
                "Class-ratio train/test split requires at least two rows per class. "
                f"Class label {class_label!r} (class id {class_id!r}) has {class_count} row. "
                f"{row_details}. Provide explicit paths.test_datasets, remove the rare class, or add more data."
            )

        shuffled = rng.permutation(class_indices)
        test_count = int(np.floor(class_count * test_size + 0.5))
        test_count = min(max(test_count, 1), class_count - 1)
        class_test = shuffled[:test_count]
        class_train = shuffled[test_count:]
        test_indices.extend(class_test.tolist())
        train_indices.extend(class_train.tolist())
        split_counts[str(class_id)] = {
            "total": int(class_count),
            "train": int(len(class_train)),
            "test": int(len(class_test)),
        }

    train_indices = rng.permutation(np.asarray(train_indices, dtype=int))
    test_indices = rng.permutation(np.asarray(test_indices, dtype=int))
    return (
        X.iloc[train_indices].reset_index(drop=True),
        X.iloc[test_indices].reset_index(drop=True),
        y[train_indices],
        y[test_indices],
        split_counts,
    )


def build_kernel(kernel_cfg: dict[str, Any]): #in v2 use whitekernel to assume data is noisy
    """Create the sklearn Gaussian Process kernel from config values."""
    kernel_type = str(kernel_cfg.get("type", "rbf")).lower()
    if kernel_type == "rbf":
        constant_value = float(kernel_cfg.get("constant_value", 1.0))
        length_scale = float(kernel_cfg.get("length_scale", 1.0))
        return ConstantKernel(constant_value, (1e-5, 10)) * RBF(length_scale, (0.5, 1e2))
    if kernel_type == "dot_product":
        sigma_0 = float(kernel_cfg.get("sigma_0", 1.0))
        return DotProduct(sigma_0=sigma_0)
    raise ValueError(f"Unsupported kernel type: {kernel_type}")


def build_model(training_cfg: dict[str, Any], seed: int) -> Pipeline:
    """Build the preprocessing + GaussianProcessClassifier sklearn pipeline."""
    preprocess_steps = [("imputer", SimpleImputer(strategy=training_cfg.get("impute_strategy", "median")))]
    if training_cfg.get("standardize", True):
        scaler_name = str(training_cfg.get("scaler", "standard")).strip().lower()
        if scaler_name in {"standard", "standard_scaler", "zscore", "z_score"}:
            preprocess_steps.append(("scaler", StandardScaler()))
        elif scaler_name in {"robust", "robust_scaler"}:
            preprocess_steps.append(("scaler", RobustScaler()))
        elif scaler_name in {"none", "false", "off"}:
            pass
        else:
            raise ValueError(f"Unsupported training.scaler: {scaler_name!r}. Use 'standard', 'robust', or 'none'.")

    classifier = GaussianProcessClassifier(
        kernel=build_kernel(training_cfg.get("kernel", {})),
        random_state=seed,
        n_restarts_optimizer=int(training_cfg.get("n_restarts_optimizer", 0)),
        max_iter_predict=int(training_cfg.get("max_iter_predict", 100)),
        multi_class=training_cfg.get("multi_class", "one_vs_rest"),
    )
    return Pipeline([
        ("preprocess", Pipeline(preprocess_steps)),
        ("classifier", classifier),
    ])


def metrics_for_split(model: Pipeline, X: pd.DataFrame, y: np.ndarray, encoder: LabelEncoder, split: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate scalar metrics, reports, confusion matrix, and predictions."""
    pred = model.predict(X)
    pred_labels = encoder.inverse_transform(pred)
    true_labels = encoder.inverse_transform(y)
    macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    report_labels = np.arange(len(encoder.classes_))
    report_text = classification_report(y, pred, labels=report_labels, target_names=encoder.classes_, zero_division=0)
    metrics: dict[str, Any] = {
        "split": split,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": macro_f1,
        "macro_f1_score": macro_f1,
        "classification_report": classification_report(y, pred, labels=report_labels, target_names=encoder.classes_, output_dict=True, zero_division=0),
        "classification_report_text": report_text,
        "confusion_matrix": confusion_matrix(y, pred, labels=report_labels).tolist(),
        "classes": encoder.classes_.tolist(),
    }
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            metrics["log_loss"] = float(log_loss(y, proba, labels=np.arange(len(encoder.classes_))))
        except Exception as exc:
            metrics["log_loss_error"] = str(exc)

    pred_df = pd.DataFrame({
        "split": split,
        "true_label": true_labels,
        "pred_label": pred_labels,
    })
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            for idx, label in enumerate(encoder.classes_):
                pred_df[f"proba_{label}"] = proba[:, idx]
        except Exception:
            pass
    return metrics, pred_df


def to_jsonable(value: Any) -> Any:
    """Convert numpy/path/sklearn-adjacent objects into JSON-writable values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, set):
        return [to_jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def init_aim_run(config: dict[str, Any], training_cfg: dict[str, Any], aim_repo: Path, run_name: str):
    """Start an Aim run when Aim is installed and tracking is enabled."""
    if not training_cfg.get("aim_enabled", True):
        return None
    if Run is None:
        print("Aim is not importable; continuing without Aim tracking.")
        return None
    try:
        run = Run(repo=str(aim_repo), experiment=config.get("project_name", "BactGPR"))
        run.name = run_name
        run["config"] = to_jsonable(config)
        run["training"] = to_jsonable(training_cfg)
        return run
    except Exception as exc:
        print(f"Aim initialization failed; continuing without Aim tracking: {exc}")
        return None


def track_metrics(run: Any, metrics: dict[str, Any]) -> None:
    """Send scalar metrics to Aim with the split as context."""
    if run is None:
        return
    split = metrics.get("split", "")
    for metric_name in ["accuracy", "balanced_accuracy", "f1_macro", "macro_f1_score", "log_loss"]:
        if metric_name in metrics:
            run.track(metrics[metric_name], name=metric_name, context={"split": split})


def save_confusion_matrix_image(metrics: dict[str, Any], output_path: Path) -> Any:
    """Save and return a labeled confusion matrix matplotlib figure."""
    classes = [str(label) for label in metrics.get("classes", [])]
    matrix = np.asarray(metrics.get("confusion_matrix", []), dtype=int)
    fig_size = max(4.0, 1.2 * max(len(classes), 1))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{str(metrics.get('split', '')).title()} confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def save_classification_report_text(metrics: dict[str, Any], output_path: Path) -> str:
    """Save and return a readable classification summary for one split."""
    split = str(metrics.get("split", "")).title()
    report = metrics.get("classification_report_text", "")
    text = f"{split} classification summary\n\n{report}"
    output_path.write_text(text + "\n", encoding="utf-8")
    return text


def save_classification_matrix_image(metrics: dict[str, Any], output_path: Path) -> Any:
    """Save and return the classification report as a matrix/table figure."""
    report = metrics.get("classification_report", {})
    rows = []
    row_labels = []
    for label, values in report.items():
        if not isinstance(values, dict):
            continue
        rows.append([
            float(values.get("precision", 0.0)),
            float(values.get("recall", 0.0)),
            float(values.get("f1-score", 0.0)),
            float(values.get("support", 0.0)),
        ])
        row_labels.append(str(label))

    if not rows:
        return save_message_figure("No classification report values available", output_path)

    cell_text = [[f"{row[0]:.3f}", f"{row[1]:.3f}", f"{row[2]:.3f}", f"{row[3]:.0f}"] for row in rows]
    fig_height = max(2.5, 0.45 * len(rows) + 1.2)
    fig_width = 7.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(f"{str(metrics.get('split', '')).title()} classification matrix", pad=12)
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=["precision", "recall", "f1-score", "support"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def save_message_figure(message: str, output_path: Path) -> Any:
    """Create, save, and return a placeholder figure when a curve cannot be computed."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def probability_matrix_from_predictions(pred_df: pd.DataFrame, classes: list[str]) -> np.ndarray | None:
    """Return an n_rows x n_classes probability matrix from prediction columns."""
    proba_columns = [f"proba_{label}" for label in classes]
    if not set(proba_columns).issubset(pred_df.columns):
        return None
    return pred_df[proba_columns].to_numpy(dtype=float)


def plotly_message_figure(title: str, message: str) -> go.Figure:
    """Create a simple Plotly figure for unavailable metric curves."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
    fig.update_layout(title=title, xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def plotly_confusion_matrix_figure(metrics: dict[str, Any]) -> go.Figure:
    """Create a Plotly confusion matrix figure for Aim Figure tracking."""
    classes = [str(label) for label in metrics.get("classes", [])]
    matrix = np.asarray(metrics.get("confusion_matrix", []), dtype=int)
    
    # Convert the matrix to a standard list to allow clean JSON tracking serialization
    fig = go.Figure(data=go.Heatmap(z=matrix.tolist(), x=classes, y=classes, colorscale="Blues", showscale=True))
    
    # Calculate a contrast threshold based on the maximum value in the matrix
    threshold = matrix.max() / 2.0 if matrix.size else 0
    
    annotations = []
    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(classes):
            # Choose white font for high-value (dark) cells, black for low-value (light) cells
            font_color = "white" if matrix[i, j] > threshold else "black"
            
            annotations.append({
                "x": pred_label, 
                "y": true_label, 
                "text": str(matrix[i, j]), 
                "showarrow": False,
                "font": {"color": font_color}  # Applies the dynamic contrast text color
            })
            
    fig.update_layout(
        title=f"{str(metrics.get('split', '')).title()} confusion matrix",
        xaxis_title="Predicted label",
        yaxis_title="True label",
        annotations=annotations,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def plotly_classification_matrix_figure(metrics: dict[str, Any]) -> go.Figure:
    """Create a Plotly table figure from the classification report."""
    report = metrics.get("classification_report", {})
    row_labels = []
    values = {"precision": [], "recall": [], "f1-score": [], "support": []}
    for label, row in report.items():
        if not isinstance(row, dict):
            continue
        row_labels.append(str(label))
        values["precision"].append(f"{float(row.get('precision', 0.0)):.3f}")
        values["recall"].append(f"{float(row.get('recall', 0.0)):.3f}")
        values["f1-score"].append(f"{float(row.get('f1-score', 0.0)):.3f}")
        values["support"].append(f"{float(row.get('support', 0.0)):.0f}")
    if not row_labels:
        return plotly_message_figure("Classification matrix", "No classification report values available.")
    fig = go.Figure(data=[go.Table(
        header={"values": ["class", "precision", "recall", "f1-score", "support"], "fill_color": "#d9e8f5", "align": "center"},
        cells={"values": [row_labels, values["precision"], values["recall"], values["f1-score"], values["support"]], "align": "center"},
    )])
    fig.update_layout(title=f"{str(metrics.get('split', '')).title()} classification matrix")
    return fig


def plotly_roc_curve_figure(pred_df: pd.DataFrame, classes: list[str], split: str) -> go.Figure:
    """Create one-vs-rest ROC curves as a Plotly figure for binary/multiclass cases."""
    proba = probability_matrix_from_predictions(pred_df, classes)
    if proba is None or pred_df.empty:
        return plotly_message_figure(f"{split.title()} ROC curve", "ROC curve unavailable: predicted probabilities were not saved.")
    true_labels = pred_df["true_label"].astype(str).to_numpy()
    y_true = np.column_stack([(true_labels == label).astype(int) for label in classes])
    fig = go.Figure()
    plotted = False
    for idx, label in enumerate(classes):
        if y_true[:, idx].min() == y_true[:, idx].max():
            continue
        fpr, tpr, _ = roc_curve(y_true[:, idx], proba[:, idx])
        
        # Cast coordinate arrays to primitive lists for database tracking storage
        fig.add_trace(go.Scatter(x=fpr.tolist(), y=tpr.tolist(), mode="lines", name=f"{label} (AUC={auc(fpr, tpr):.3f})"))
        plotted = True
    if y_true.size and y_true.min() != y_true.max():
        fpr, tpr, _ = roc_curve(y_true.ravel(), proba.ravel())
        
        # Cast aggregate coordinate arrays to primitive lists for database tracking storage
        fig.add_trace(go.Scatter(x=fpr.tolist(), y=tpr.tolist(), mode="lines", name=f"micro avg (AUC={auc(fpr, tpr):.3f})", line={"dash": "dash", "color": "black"}))
        plotted = True
    if not plotted:
        return plotly_message_figure(f"{split.title()} ROC curve", "ROC curve unavailable: split does not contain both positive and negative examples.")
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line={"dash": "dot", "color": "gray"}))
    fig.update_layout(title=f"{split.title()} ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", xaxis_range=[0, 1], yaxis_range=[0, 1.02])
    return fig


def plotly_precision_recall_curve_figure(pred_df: pd.DataFrame, classes: list[str], split: str) -> go.Figure:
    """Create one-vs-rest precision-recall curves as a Plotly figure."""
    proba = probability_matrix_from_predictions(pred_df, classes)
    if proba is None or pred_df.empty:
        return plotly_message_figure(f"{split.title()} precision-recall curve", "Precision-recall curve unavailable: predicted probabilities were not saved.")
    true_labels = pred_df["true_label"].astype(str).to_numpy()
    y_true = np.column_stack([(true_labels == label).astype(int) for label in classes])
    fig = go.Figure()
    plotted = False
    for idx, label in enumerate(classes):
        if y_true[:, idx].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true[:, idx], proba[:, idx])
        ap = average_precision_score(y_true[:, idx], proba[:, idx])
        
        # Cast metric coordinates to primitive lists for database tracking storage
        fig.add_trace(go.Scatter(x=recall.tolist(), y=precision.tolist(), mode="lines", name=f"{label} (AP={ap:.3f})"))
        plotted = True
    if y_true.size and y_true.sum() > 0:
        precision, recall, _ = precision_recall_curve(y_true.ravel(), proba.ravel())
        ap = average_precision_score(y_true, proba, average="micro")
        
        # Cast metric coordinates to primitive lists for database tracking storage
        fig.add_trace(go.Scatter(x=recall.tolist(), y=precision.tolist(), mode="lines", name=f"micro avg (AP={ap:.3f})", line={"dash": "dash", "color": "black"}))
        plotted = True
    if not plotted:
        return plotly_message_figure(f"{split.title()} precision-recall curve", "Precision-recall curve unavailable: split has no positive examples.")
    fig.update_layout(title=f"{split.title()} precision-recall curve", xaxis_title="Recall", yaxis_title="Precision", xaxis_range=[0, 1], yaxis_range=[0, 1.02])
    return fig


def save_roc_curve_image(pred_df: pd.DataFrame, classes: list[str], split: str, output_path: Path) -> Any:
    """Save and return one-vs-rest ROC curves for binary or multiclass predictions."""
    proba = probability_matrix_from_predictions(pred_df, classes)
    if proba is None or pred_df.empty:
        return save_message_figure("ROC curve unavailable: predicted probabilities were not saved.", output_path)

    true_labels = pred_df["true_label"].astype(str).to_numpy()
    y_true = np.column_stack([(true_labels == label).astype(int) for label in classes])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plotted = False
    for idx, label in enumerate(classes):
        if y_true[:, idx].min() == y_true[:, idx].max():
            continue
        fpr, tpr, _ = roc_curve(y_true[:, idx], proba[:, idx])
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
        plotted = True

    if y_true.size and y_true.min() != y_true.max():
        fpr, tpr, _ = roc_curve(y_true.ravel(), proba.ravel())
        ax.plot(fpr, tpr, color="black", lw=1.5, linestyle="--", label=f"micro avg (AUC={auc(fpr, tpr):.3f})")
        plotted = True

    if not plotted:
        plt.close(fig)
        return save_message_figure("ROC curve unavailable: split does not contain both positive and negative examples.", output_path)

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{split.title()} ROC curve")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def save_precision_recall_curve_image(pred_df: pd.DataFrame, classes: list[str], split: str, output_path: Path) -> Any:
    """Save and return one-vs-rest precision-recall curves for binary or multiclass predictions."""
    proba = probability_matrix_from_predictions(pred_df, classes)
    if proba is None or pred_df.empty:
        return save_message_figure("Precision-recall curve unavailable: predicted probabilities were not saved.", output_path)

    true_labels = pred_df["true_label"].astype(str).to_numpy()
    y_true = np.column_stack([(true_labels == label).astype(int) for label in classes])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plotted = False
    for idx, label in enumerate(classes):
        if y_true[:, idx].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true[:, idx], proba[:, idx])
        ap = average_precision_score(y_true[:, idx], proba[:, idx])
        ax.plot(recall, precision, lw=2, label=f"{label} (AP={ap:.3f})")
        plotted = True

    if y_true.size and y_true.sum() > 0:
        precision, recall, _ = precision_recall_curve(y_true.ravel(), proba.ravel())
        ap = average_precision_score(y_true, proba, average="micro")
        ax.plot(recall, precision, color="black", lw=1.5, linestyle="--", label=f"micro avg (AP={ap:.3f})")
        plotted = True

    if not plotted:
        plt.close(fig)
        return save_message_figure("Precision-recall curve unavailable: split has no positive examples.", output_path)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{split.title()} precision-recall curve")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def log_split_evaluation_to_aim(run: Any, split: str, figures: dict[str, Any], figure_paths: dict[str, Path], report_text: str) -> None:
    """Track one split's Plotly figures and classification report text in Aim."""
    if run is None:
        return
    context = {"split": split}
    if AimFigure is not None:
        for figure_name, figure in figures.items():
            try:
                run.track(AimFigure(figure), name=figure_name, context=context)
            except Exception as exc:
                run[f"{split}_{figure_name}_warning"] = f"Aim Figure logging failed: {exc}"
    if Text is not None:
        try:
            run.track(Text(report_text), name="classification_summary", context=context)
        except Exception as exc:
            run[f"{split}_classification_summary_warning"] = f"Aim Text logging failed: {exc}"
    run[f"{split}_classification_summary"] = report_text
    for figure_name, figure_path in figure_paths.items():
        run[f"{split}_{figure_name}"] = str(figure_path)


def save_and_log_split_evaluation_artifacts(
    run: Any,
    metrics: dict[str, Any],
    pred_df: pd.DataFrame,
    split: str,
    classes: list[str],
    image_paths: dict[str, Path],
    report_path: Path,
) -> None:
    """Create filesystem and Aim artifacts for split-level evaluation outputs."""
    local_figures = {
        "confusion_matrix": save_confusion_matrix_image(metrics, image_paths["confusion_matrix"]),
        "roc_curve": save_roc_curve_image(pred_df, classes, split, image_paths["roc_curve"]),
        "precision_recall_curve": save_precision_recall_curve_image(pred_df, classes, split, image_paths["precision_recall_curve"]),
        "classification_matrix": save_classification_matrix_image(metrics, image_paths["classification_matrix"]),
    }
    aim_figures = {
        "confusion_matrix": plotly_confusion_matrix_figure(metrics),
        "roc_curve": plotly_roc_curve_figure(pred_df, classes, split),
        "precision_recall_curve": plotly_precision_recall_curve_figure(pred_df, classes, split),
        "classification_matrix": plotly_classification_matrix_figure(metrics),
    }
    report_text = save_classification_report_text(metrics, report_path)
    try:
        log_split_evaluation_to_aim(run, split, aim_figures, image_paths, report_text)
    finally:
        for figure in local_figures.values():
            plt.close(figure)


def parse_feature_metadata(feature: str) -> dict[str, Any]:
    """Extract frequency/parameter hints from a formed feature column name."""
    match = re.match(r"^(?P<frequency>\d+(?:\.\d+)?)GHz_(?P<body>.+)$", str(feature))
    if not match:
        return {"frequency_ghz": np.nan, "feature_family": "", "parameter": ""}

    body = match.group("body")
    frequency_ghz = float(match.group("frequency"))
    family = body
    parameter = ""
    if body.startswith("dS"):
        family = "dS"
        parameter = "s" + body[2:]
    elif "_" in body:
        parameter, family = body.split("_", 1)
    return {"frequency_ghz": frequency_ghz, "feature_family": family, "parameter": parameter}


def sample_importance_rows(X: pd.DataFrame, y: np.ndarray, max_rows: int | None, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Optionally limit permutation-importance rows without changing training."""
    if not max_rows or len(X) <= int(max_rows):
        return X, y
    y_series = pd.Series(y)
    stratify = y if y_series.value_counts().min() >= 2 else None
    sampled_X, _, sampled_y, _ = train_test_split(
        X,
        y,
        train_size=int(max_rows),
        stratify=stratify,
        random_state=seed,
    )
    return sampled_X.reset_index(drop=True), sampled_y


def calculate_feature_importance(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_columns: list[str],
    cfg: dict[str, Any],
    seed: int,
    split: str,
) -> pd.DataFrame:
    """Calculate permutation feature importance for the trained sklearn pipeline."""
    n_repeats = int(cfg.get("n_repeats", 5))
    scoring = str(cfg.get("scoring", "f1_macro"))
    X_eval, y_eval = sample_importance_rows(X, y, cfg.get("max_rows"), seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("once", message="Skipping features without any observed values")
        result = permutation_importance(
            model,
            X_eval,
            y_eval,
            n_repeats=n_repeats,
            random_state=seed,
            scoring=scoring,
        )

    rows = []
    for idx, feature in enumerate(feature_columns):
        metadata = parse_feature_metadata(feature)
        rows.append({
            "feature": feature,
            "importance_mean": float(result.importances_mean[idx]),
            "importance_std": float(result.importances_std[idx]),
            "importance_abs_mean": float(abs(result.importances_mean[idx])),
            "split": split,
            "scoring": scoring,
            "n_repeats": n_repeats,
            "n_rows": int(len(X_eval)),
            **metadata,
        })

    importance_df = pd.DataFrame(rows).sort_values(
        ["importance_mean", "importance_abs_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)
    importance_df.insert(0, "rank", np.arange(1, len(importance_df) + 1))
    return importance_df


def feature_importance_plotly_table(importance_df: pd.DataFrame, top_n: int = 50) -> go.Figure:
    """Create a Plotly table so Aim can track feature importance as a Figure."""
    if importance_df.empty:
        return plotly_message_figure("Feature importance", "Feature importance was not calculated.")
    display_df = importance_df.head(int(top_n)).copy()
    columns = [
        "rank",
        "feature",
        "frequency_ghz",
        "parameter",
        "importance_mean",
        "importance_std",
        "scoring",
        "split",
    ]
    columns = [column for column in columns if column in display_df.columns]
    cell_values = []
    for column in columns:
        values = display_df[column]
        if pd.api.types.is_float_dtype(values):
            cell_values.append([f"{value:.6g}" if pd.notna(value) else "" for value in values])
        else:
            cell_values.append(values.astype(str).tolist())
    fig = go.Figure(data=[go.Table(
        header={"values": columns, "fill_color": "#d9e8f5", "align": "left"},
        cells={"values": cell_values, "align": "left"},
    )])
    fig.update_layout(title=f"Top {min(int(top_n), len(display_df))} permutation feature importances", height=max(420, 24 * len(display_df) + 120))
    return fig


def feature_importance_text_table(importance_df: pd.DataFrame, top_n: int = 50) -> str:
    """Create a dependency-free text table for Aim Text tracking."""
    if importance_df.empty:
        return "Feature importance was not calculated."
    columns = ["rank", "feature", "frequency_ghz", "parameter", "importance_mean", "importance_std", "scoring", "split"]
    columns = [column for column in columns if column in importance_df.columns]
    display_df = importance_df.head(int(top_n))[columns].copy()
    return display_df.to_string(index=False)


def save_and_log_feature_importance(
    run: Any,
    importance_df: pd.DataFrame,
    artifacts: dict[str, str],
    top_n_aim: int,
) -> None:
    """Save feature-importance artifacts and track them in Aim."""
    csv_path = Path(artifacts["feature_importance"])
    json_path = Path(artifacts["feature_importance_json"])
    html_path = Path(artifacts["feature_importance_html"])
    importance_df.to_csv(csv_path, index=False)
    save_json(json_path, importance_df.to_dict(orient="records"))
    html_path.write_text(importance_df.to_html(index=False), encoding="utf-8")

    if run is None:
        return
    run["feature_importance_csv"] = str(csv_path)
    run["feature_importance_json"] = str(json_path)
    run["feature_importance_html"] = str(html_path)
    if AimFigure is not None:
        try:
            run.track(AimFigure(feature_importance_plotly_table(importance_df, top_n_aim)), name="feature_importance_table")
        except Exception as exc:
            run["feature_importance_table_warning"] = f"Aim Figure logging failed: {exc}"
    if Text is not None:
        try:
            run.track(Text(feature_importance_text_table(importance_df, top_n_aim)), name="feature_importance_table_text")
        except Exception as exc:
            run["feature_importance_text_warning"] = f"Aim Text logging failed: {exc}"


def format_completion_message(summary: dict[str, Any]) -> str:
    """Return the exact final console text printed after training completes."""
    return "Training complete\n" + json.dumps(to_jsonable(summary), indent=2)


def log_completion_to_aim(run: Any, completion_text: str) -> None:
    """Store final console output in Aim as both run metadata and text when possible."""
    if run is None:
        return
    run["training_complete_output"] = completion_text
    if Text is not None:
        try:
            run.track(Text(completion_text), name="training_complete_output")
        except Exception as exc:
            run["training_complete_text_warning"] = f"Aim Text logging failed: {exc}"


def feature_value_map(values: Any, feature_columns: list[str]) -> dict[str, Any] | list[Any] | None:
    """Map fitted vector values such as imputer/scaler statistics back to features."""
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim == 1 and len(array) == len(feature_columns):
        return {feature: to_jsonable(array[idx]) for idx, feature in enumerate(feature_columns)}
    return to_jsonable(array)


def estimator_shallow_params(estimator: Any) -> dict[str, Any]:
    """Return readable constructor-level params without serializing fitted arrays."""
    if estimator is None:
        return {}
    params = {
        "class": estimator.__class__.__name__,
        "module": estimator.__class__.__module__,
    }
    if hasattr(estimator, "get_params"):
        params["params"] = {name: to_jsonable(value) for name, value in estimator.get_params(deep=False).items()}
    return params


def fitted_estimator_summary(estimator: Any) -> dict[str, Any]:
    """Summarize a fitted sklearn estimator without storing the full model twice."""
    if estimator is None:
        return {}
    summary = estimator_shallow_params(estimator)
    for attr in ("classes_", "n_classes_", "n_features_in_", "log_marginal_likelihood_value_"):
        if hasattr(estimator, attr):
            summary[attr] = to_jsonable(getattr(estimator, attr))
    if hasattr(estimator, "kernel"):
        summary["initial_kernel"] = str(getattr(estimator, "kernel"))
    if hasattr(estimator, "kernel_"):
        summary["fitted_kernel"] = str(getattr(estimator, "kernel_"))
    if hasattr(estimator, "estimators_"):
        estimator_summaries = []
        for idx, nested_estimator in enumerate(getattr(estimator, "estimators_", [])):
            nested_summary = fitted_estimator_summary(nested_estimator)
            nested_summary["index"] = idx
            estimator_summaries.append(nested_summary)
        summary["estimators_"] = estimator_summaries
    return summary


def collect_fitted_model_settings(
    model: Pipeline,
    training_cfg: dict[str, Any],
    feature_columns: list[str],
    encoder: LabelEncoder,
    configured_freqs: str | list[float],
    feature_freqs: list[float],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict[str, Any]:
    """Collect final trained parameters/settings for logs, Aim, and notebooks."""
    preprocess = model.named_steps.get("preprocess") if hasattr(model, "named_steps") else None
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else None

    preprocess_settings: dict[str, Any] = {}
    if preprocess is not None:
        preprocess_settings = estimator_shallow_params(preprocess)
        preprocess_settings["steps"] = []
        for step_name, step_estimator in getattr(preprocess, "steps", []):
            step_summary = estimator_shallow_params(step_estimator)
            step_summary["name"] = step_name
            if step_name == "imputer":
                step_summary["statistics_by_feature"] = feature_value_map(getattr(step_estimator, "statistics_", None), feature_columns)
                step_summary["indicator_features_"] = to_jsonable(getattr(step_estimator, "indicator_", None))
            for attr in ("mean_", "var_", "scale_", "center_", "n_samples_seen_"):
                if hasattr(step_estimator, attr):
                    step_summary[attr] = feature_value_map(getattr(step_estimator, attr), feature_columns)
            preprocess_settings["steps"].append(step_summary)

    classifier_settings = fitted_estimator_summary(classifier)
    model_settings = {
        "pipeline": estimator_shallow_params(model),
        "preprocess": preprocess_settings,
        "classifier": classifier_settings,
        "training_config": to_jsonable(training_cfg),
        "configured_freqs": configured_freqs,
        "feature_frequencies_ghz": feature_freqs,
        "feature_columns": feature_columns,
        "n_features": int(len(feature_columns)),
        "classes": encoder.classes_.tolist(),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
    }
    final_kernel = classifier_settings.get("fitted_kernel") or classifier_settings.get("initial_kernel")
    if final_kernel is not None:
        model_settings["final_kernel"] = final_kernel
    if "log_marginal_likelihood_value_" in classifier_settings:
        model_settings["log_marginal_likelihood_value"] = classifier_settings["log_marginal_likelihood_value_"]
    return model_settings


def save_and_log_model_settings(run: Any, model_settings: dict[str, Any], output_path: Path) -> None:
    """Save final fitted model settings locally and in Aim."""
    save_json(output_path, model_settings)
    if run is None:
        return
    try:
        run["model_settings_json"] = str(output_path)
        run["final_trained_model_settings"] = to_jsonable(model_settings)
        if model_settings.get("final_kernel"):
            run["final_kernel"] = model_settings["final_kernel"]
        if "log_marginal_likelihood_value" in model_settings:
            run["log_marginal_likelihood_value"] = model_settings["log_marginal_likelihood_value"]
    except Exception as exc:
        run["model_settings_metadata_warning"] = f"Aim metadata logging failed: {exc}"
    if Text is not None:
        try:
            run.track(Text(json.dumps(to_jsonable(model_settings), indent=2)), name="final_trained_model_settings")
        except Exception as exc:
            run["model_settings_text_warning"] = f"Aim Text logging failed: {exc}"


def save_json(path: Path, data: Any) -> None:
    """Write JSON artifacts using the shared numpy/path conversion helper."""
    path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def build_run_name(config: dict[str, Any]) -> str:
    """Create a readable timestamped run name for logs/checkpoints/Aim."""
    project = str(config.get("project_name", "BactGPR"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{project}-{stamp}"


def train(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full training workflow and return a compact run summary."""
    seed = int(config.get("seed", 0))
    np.random.seed(seed)
    training_cfg = get_training_config(config)
    dataset_dir = processed_dataset_dir(config)
    table = training_cfg["table"]
    train_names = list(config.get("paths", {}).get("train_datasets", []))
    test_names = list(config.get("paths", {}).get("test_datasets", []))
    if not train_names:
        raise ValueError("paths.train_datasets must contain at least one formed dataset name")

    train_df = load_named_datasets(dataset_dir, train_names, table, "train")
    test_df = load_named_datasets(dataset_dir, test_names, table, "test") if test_names else pd.DataFrame()
    train_df = maybe_drop_ext(train_df, bool(training_cfg.get("drop_ext", False)))
    test_df = maybe_drop_ext(test_df, bool(training_cfg.get("drop_ext", False)))

    y_train, y_test, encoder = prepare_targets(train_df, test_df, training_cfg)
    run_name = build_run_name(config)
    checkpoint_dir, log_dir, aim_repo = output_dirs(config, run_name)
    aim_run = init_aim_run(config, training_cfg, aim_repo, run_name)
    feature_columns = select_feature_columns(train_df, test_df, training_cfg)
    configured_freqs = frequency_summary(training_cfg.get("freqs", "all"))
    feature_freqs = selected_feature_frequencies(feature_columns)
    train_df, y_train = maybe_limit_train_rows(train_df, y_train, training_cfg.get("max_train_rows"), seed)

    X_train = train_df[feature_columns]
    split_class_counts = None
    if test_df.empty:
        X_train, X_test, y_train, y_test, split_class_counts = per_class_train_test_split(
            X_train,
            y_train,
            test_size=float(training_cfg.get("test_size", 0.25)),
            seed=seed,
            metadata_df=train_df,
            class_names=encoder.classes_,
        )
        split_class_counts = {
            str(encoder.classes_[int(class_id)]): counts
            for class_id, counts in split_class_counts.items()
        }
        print(f"Internal class-ratio test split counts: {split_class_counts}")
        test_meta = pd.DataFrame(index=X_test.index)
    else:
        X_test = test_df[feature_columns]
        test_meta = test_df.reset_index(drop=True)

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Configured frequencies: {configured_freqs}")
    print(f"Feature frequencies used: {feature_freqs}")
    print(f"Classes: {encoder.classes_.tolist()}")

    model = build_model(training_cfg, seed)
    model.fit(X_train, y_train)
    model_settings = collect_fitted_model_settings(
        model,
        training_cfg,
        feature_columns,
        encoder,
        configured_freqs,
        feature_freqs,
        X_train,
        X_test,
    )

    train_metrics, train_pred_df = metrics_for_split(model, X_train, y_train, encoder, "train")
    test_metrics, test_pred_df = metrics_for_split(model, X_test, y_test, encoder, "test")
    track_metrics(aim_run, train_metrics)
    track_metrics(aim_run, test_metrics)

    if not test_df.empty:
        meta_cols = [col for col in ["dataset_name", "dataset_subdir", "group_number", "ext", "sample_stream", "label"] if col in test_meta.columns]
        test_pred_df = pd.concat([test_meta[meta_cols].reset_index(drop=True), test_pred_df.reset_index(drop=True)], axis=1)

    artifacts = {
        "model": str(checkpoint_dir / "gpr_classifier.joblib"),
        "model_bundle": str(checkpoint_dir / "model_bundle.joblib"),
        "label_encoder_classes": str(checkpoint_dir / "label_classes.json"),
        "feature_columns": str(checkpoint_dir / "feature_columns.json"),
        "metrics": str(log_dir / "metrics.json"),
        "model_settings": str(log_dir / "model_settings.json"),
        "feature_importance": str(log_dir / "feature_importance.csv"),
        "feature_importance_json": str(log_dir / "feature_importance.json"),
        "feature_importance_html": str(log_dir / "feature_importance.html"),
        "test_predictions": str(log_dir / "test_predictions.csv"),
        "train_predictions": str(log_dir / "train_predictions.csv"),
        "train_confusion_matrix": str(log_dir / "train_confusion_matrix.png"),
        "test_confusion_matrix": str(log_dir / "test_confusion_matrix.png"),
        "train_roc_curve": str(log_dir / "train_roc_curve.png"),
        "test_roc_curve": str(log_dir / "test_roc_curve.png"),
        "train_precision_recall_curve": str(log_dir / "train_precision_recall_curve.png"),
        "test_precision_recall_curve": str(log_dir / "test_precision_recall_curve.png"),
        "train_classification_matrix": str(log_dir / "train_classification_matrix.png"),
        "test_classification_matrix": str(log_dir / "test_classification_matrix.png"),
        "train_classification_report": str(log_dir / "train_classification_report.txt"),
        "test_classification_report": str(log_dir / "test_classification_report.txt"),
        "config": str(log_dir / "config.json"),
        "training_complete_output": str(log_dir / "training_complete_output.txt"),
    }
    joblib.dump(model, artifacts["model"])
    save_json(Path(artifacts["label_encoder_classes"]), encoder.classes_.tolist())
    save_json(Path(artifacts["feature_columns"]), feature_columns)
    save_json(Path(artifacts["metrics"]), {"train": train_metrics, "test": test_metrics})
    save_and_log_model_settings(aim_run, model_settings, Path(artifacts["model_settings"]))
    save_json(Path(artifacts["config"]), config)
    if training_cfg.get("save_predictions", True):
        train_pred_df.to_csv(artifacts["train_predictions"], index=False)
        test_pred_df.to_csv(artifacts["test_predictions"], index=False)

    feature_importance_df = pd.DataFrame()
    feature_importance_cfg = training_cfg.get("feature_importance", {}) or {}
    if bool(feature_importance_cfg.get("enabled", True)):
        importance_split = str(feature_importance_cfg.get("split", "test")).lower()
        if importance_split == "train":
            importance_X, importance_y = X_train, y_train
            importance_split = "train"
        else:
            importance_X, importance_y = X_test, y_test
            importance_split = "test"
        try:
            feature_importance_df = calculate_feature_importance(
                model,
                importance_X,
                importance_y,
                feature_columns,
                feature_importance_cfg,
                seed,
                importance_split,
            )
            save_and_log_feature_importance(
                aim_run,
                feature_importance_df,
                artifacts,
                int(feature_importance_cfg.get("top_n_aim", 50)),
            )
        except Exception as exc:
            print(f"Feature importance calculation failed: {exc}")
            if aim_run is not None:
                aim_run["feature_importance_error"] = str(exc)
            save_json(Path(artifacts["feature_importance_json"]), {"error": str(exc)})

    classes = [str(label) for label in encoder.classes_]
    save_and_log_split_evaluation_artifacts(
        aim_run,
        train_metrics,
        train_pred_df,
        "train",
        classes,
        {
            "confusion_matrix": Path(artifacts["train_confusion_matrix"]),
            "roc_curve": Path(artifacts["train_roc_curve"]),
            "precision_recall_curve": Path(artifacts["train_precision_recall_curve"]),
            "classification_matrix": Path(artifacts["train_classification_matrix"]),
        },
        Path(artifacts["train_classification_report"]),
    )
    save_and_log_split_evaluation_artifacts(
        aim_run,
        test_metrics,
        test_pred_df,
        "test",
        classes,
        {
            "confusion_matrix": Path(artifacts["test_confusion_matrix"]),
            "roc_curve": Path(artifacts["test_roc_curve"]),
            "precision_recall_curve": Path(artifacts["test_precision_recall_curve"]),
            "classification_matrix": Path(artifacts["test_classification_matrix"]),
        },
        Path(artifacts["test_classification_report"]),
    )

    summary = {
        "run_name": run_name,
        "dataset_dir": str(dataset_dir),
        "train_datasets": train_names,
        "test_datasets": test_names,
        "table": table,
        "configured_freqs": configured_freqs,
        "feature_frequencies_ghz": feature_freqs,
        "target_mode": training_cfg.get("target_mode"),
        "test_target_remap": training_cfg.get("test_target_remap", {}),
        "classes": encoder.classes_.tolist(),
        "n_features": len(feature_columns),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "internal_split_class_counts": split_class_counts,
        "feature_importance_rows": int(len(feature_importance_df)),
        "final_kernel": model_settings.get("final_kernel"),
        "model_settings": artifacts["model_settings"],
        "train_metrics": {k: v for k, v in train_metrics.items() if k not in {"classification_report", "classification_report_text", "confusion_matrix"}},
        "test_metrics": {k: v for k, v in test_metrics.items() if k not in {"classification_report", "classification_report_text", "confusion_matrix"}},
        "artifacts": artifacts,
    }
    model_bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "label_encoder_classes": encoder.classes_.tolist(),
        "config": to_jsonable(config),
        "training_config": to_jsonable(training_cfg),
        "summary": to_jsonable(summary),
        "model_settings": to_jsonable(model_settings),
        "configured_freqs": configured_freqs,
        "feature_frequencies_ghz": feature_freqs,
        "dataset_dir": str(dataset_dir),
        "table": table,
        "target_column": training_cfg.get("target_column"),
        "target_mode": training_cfg.get("target_mode"),
        "label_map": training_cfg.get("label_map", {}) or {},
        "train_datasets": train_names,
        "test_datasets": test_names,
        "run_name": run_name,
    }
    joblib.dump(model_bundle, artifacts["model_bundle"])
    save_json(log_dir / "summary.json", summary)
    completion_text = format_completion_message(summary)
    Path(artifacts["training_complete_output"]).write_text(completion_text + "\n", encoding="utf-8")

    # Print before closing Aim so Aim's stdout log captures the final summary too.
    print("\n" + completion_text, flush=True)

    if aim_run is not None:
        aim_run["summary"] = to_jsonable(summary)
        log_completion_to_aim(aim_run, completion_text)
        aim_run.close()
    return summary


def main() -> None:
    """CLI entrypoint used by ``python BactTrain-gpr.py --config ...``."""
    parser = argparse.ArgumentParser(description="Train a Gaussian Process classifier for formed bacteria/particle datasets")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to a GPR YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
