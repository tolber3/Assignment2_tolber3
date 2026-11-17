import argparse
import copy
import json
import math
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat


PROCESSED_DATA_COLUMNS = [
    "set",
    "lamp",
    "FiltSize",
    "absolute_time",
    "T_peaks",
    "dS_s11m",
    "dS_s11a",
    "dS_s21m",
    "dS_s21a",
    "gap",
    "index",
    "CW_Freq",
    "baseline_s11m",
    "baseline_s11a",
    "baseline_s21m",
    "baseline_s21a",
    "dT",
    "T_left",
    "T_right",
]
PARAMS = ("s11m", "s11a", "s21m", "s21a")
DS_COLUMNS = {
    "s11m": "dS_s11m",
    "s11a": "dS_s11a",
    "s21m": "dS_s21m",
    "s21a": "dS_s21a",
}
BASELINE_COLUMNS = {
    "s11m": "baseline_s11m",
    "s11a": "baseline_s11a",
    "s21m": "baseline_s21m",
    "s21a": "baseline_s21a",
}
VALID_MODES = {"pairs_only", "unpaired_only", "all"}
DEFAULT_TRAVEL_DISTANCE_UM = 120.0


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config["_config_path"] = str(Path(path).resolve())
    return config


def load_recreate_config_if_requested(config: dict[str, Any]) -> dict[str, Any]:
    remake_path = config.get("paths", {}).get("remake_processed_data", False)
    if not remake_path or str(remake_path).lower() in {"false", "none", "0"}:
        return config

    remake_path = Path(remake_path)
    with remake_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    effective_config = metadata.get("effective_config")
    if not effective_config:
        raise ValueError(f"{remake_path} does not contain an effective_config section")
    effective_config["_config_path"] = config.get("_config_path", "")
    print(f"Recreating dataset configuration from {remake_path}")
    return effective_config


def parse_set_number(path: Path) -> int | None:
    match = re.search(r"-(\d+)(?:_|\.|$)", path.name)
    return int(match.group(1)) if match else None


def frequency_ghz(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, str):
        return float(value.lower().replace("ghz", "").strip())
    value = float(value)
    return value / 1e9 if abs(value) > 1e5 else value


def frequency_key(value: Any) -> float:
    return round(frequency_ghz(value), 8)


def frequency_label(value: Any) -> str:
    value = frequency_ghz(value)
    if not np.isfinite(value):
        return "nanGHz"
    return f"{value:g}GHz"


def normalize_frequency_selection(freqs: Any) -> set[float] | None:
    """Return selected GHz keys, or None when all available frequencies should be used."""
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

    selected: set[float] = set()
    for raw_value in raw_values:
        if isinstance(raw_value, str) and raw_value.strip().lower() in {"all", "*"}:
            return None
        selected.add(frequency_key(raw_value))
    return selected


def frequency_selection_summary(freqs: Any) -> str | list[float]:
    """Make selected frequencies compact for JSON/markdown metadata."""
    selected = normalize_frequency_selection(freqs)
    if selected is None:
        return "all"
    return [float(freq) for freq in sorted(selected, key=float)]


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def get_dataset_processed_subdir(config: dict[str, Any]) -> str:
    form_cfg = config.get("form_sets", {})
    paths = config.get("paths", {})
    processed_subdir = form_cfg.get("dataset_processed_subdir", paths.get("dataset_processed_subdir"))
    if not processed_subdir:
        raise ValueError("dataset_processed_subdir must be set under form_sets")
    return str(processed_subdir)


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def read_processed_mat(mat_path: Path, warnings_log: list[dict[str, Any]]) -> pd.DataFrame:
    mat = loadmat(mat_path)
    processed = mat.get("processedData")
    if processed is None or processed.size == 0:
        warnings_log.append({"file": str(mat_path), "warning": "processedData missing or empty"})
        return pd.DataFrame(columns=PROCESSED_DATA_COLUMNS)

    processed = np.asarray(processed)
    if processed.ndim != 2:
        warnings_log.append({"file": str(mat_path), "warning": f"processedData has unexpected shape {processed.shape}"})
        return pd.DataFrame(columns=PROCESSED_DATA_COLUMNS)

    columns = PROCESSED_DATA_COLUMNS[: processed.shape[1]]
    if processed.shape[1] > len(PROCESSED_DATA_COLUMNS):
        columns = PROCESSED_DATA_COLUMNS + [f"extra_{idx}" for idx in range(processed.shape[1] - len(PROCESSED_DATA_COLUMNS))]

    df = pd.DataFrame(processed, columns=columns)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "index" not in df.columns:
        warnings_log.append({"file": str(mat_path), "warning": "processedData does not include index column"})
        return df.iloc[0:0].copy()

    before = len(df)
    df = df[df["index"].notna() & (df["index"] != 0)].copy()
    invalid = before - len(df)
    if invalid:
        warnings_log.append({"file": str(mat_path), "warning": f"ignored {invalid} invalid peak rows with index column equal to 0/NaN"})

    if len(df) < 2:
        warnings_log.append({"file": str(mat_path), "warning": f"fewer than 2 valid peaks ({len(df)})"})

    df["source_mat"] = mat_path.name
    inferred_set = parse_set_number(mat_path)
    if "set" not in df.columns or df["set"].isna().all():
        df["set"] = inferred_set
    df["set_number"] = df["set"].astype("Int64")
    df["peak_number"] = df["index"].astype("Int64")
    df["frequency_ghz"] = df["CW_Freq"].map(frequency_ghz) if "CW_Freq" in df.columns else np.nan
    df["frequency_key"] = df["frequency_ghz"].map(frequency_key)
    df["lr_timewidth"] = df["dT"] if "dT" in df.columns else df.get("T_right", np.nan) - df.get("T_left", np.nan)
    return df


def load_pairs_csv(dataset_dir: Path, set_number: int) -> pd.DataFrame | None:
    pair_path = dataset_dir / f"bact-{set_number}_pairs.csv"
    if not pair_path.exists():
        return None
    pairs = pd.read_csv(pair_path)
    for column in pairs.columns:
        if column != "valid_pair":
            pairs[column] = pd.to_numeric(pairs[column], errors="coerce")
    if "valid_pair" in pairs.columns:
        pairs["valid_pair"] = pairs["valid_pair"].map(coerce_bool)
    else:
        pairs["valid_pair"] = pairs["peak_1_index"].notna() & pairs["peak_2_index"].notna()
    return pairs


def apply_pair_status(peaks: pd.DataFrame, pairs: pd.DataFrame | None, mode: str, dataset_dir: Path, warnings_log: list[dict[str, Any]]) -> pd.DataFrame:
    peaks = peaks.copy()
    peaks["pairs_csv_present"] = pairs is not None
    peaks["paired"] = False
    peaks["pair_id"] = np.nan
    peaks["pair_role"] = ""
    peaks["pair_speed_um_per_s"] = np.nan
    peaks["pair_time_diff_s"] = np.nan

    if pairs is None:
        if mode == "unpaired_only":
            warnings_log.append({"dataset_dir": str(dataset_dir), "warning": "pairs.csv missing for set; unpaired_only includes all valid peaks for that set"})
            return peaks
        if mode == "pairs_only":
            warnings_log.append({"dataset_dir": str(dataset_dir), "warning": "pairs.csv missing for set; pairs_only excludes all peaks for that set"})
            return peaks.iloc[0:0].copy()
        return peaks

    valid_pairs = pairs[pairs["valid_pair"]].copy()
    pair_records: dict[int, dict[str, Any]] = {}
    for _, row in valid_pairs.iterrows():
        pair_id = safe_float(row.get("pair_id", np.nan))
        peak_time_diff = safe_float(row.get("peak_time_diff", np.nan))
        pair_speed = DEFAULT_TRAVEL_DISTANCE_UM / peak_time_diff if np.isfinite(peak_time_diff) and peak_time_diff > 0 else np.nan
        for role in ("peak_1", "peak_2"):
            peak_index = row.get(f"{role}_index", np.nan)
            if pd.notna(peak_index):
                pair_records[int(peak_index)] = {
                    "paired": True,
                    "pair_id": pair_id,
                    "pair_role": role,
                    "pair_time_diff_s": peak_time_diff,
                    "pair_speed_um_per_s": pair_speed,
                }

    for idx, row in peaks.iterrows():
        peak_number = row.get("peak_number", np.nan)
        if pd.isna(peak_number):
            continue
        record = pair_records.get(int(peak_number))
        if record:
            for key, value in record.items():
                peaks.at[idx, key] = value

    if mode == "pairs_only":
        peaks = peaks[peaks["paired"]].copy()
    elif mode == "unpaired_only":
        peaks = peaks[~peaks["paired"]].copy()
    elif mode != "all":
        raise ValueError(f"Unsupported mode {mode!r}; valid modes are {sorted(VALID_MODES)}")

    return peaks


def parse_plot_file_info(path: Path) -> tuple[int | None, float | None]:
    match = re.search(r"-(\d+)_([0-9.]+)GHz", path.name)
    if not match:
        return None, None
    return int(match.group(1)), float(match.group(2))


def build_plot_index(dataset_dir: Path) -> dict[tuple[int, float], Path]:
    index: dict[tuple[int, float], Path] = {}
    for path in sorted(dataset_dir.glob("*_plot_data.csv")):
        set_number, freq = parse_plot_file_info(path)
        if set_number is None or freq is None:
            continue
        index.setdefault((set_number, frequency_key(freq)), path)
    return index


def interpolate_window(trace_df: pd.DataFrame, left_time: float, right_time: float, columns: list[str]) -> pd.DataFrame | None:
    if not np.isfinite(left_time) or not np.isfinite(right_time) or right_time <= left_time:
        return None
    times_full = pd.to_numeric(trace_df["time"], errors="coerce").to_numpy(dtype=float)
    finite_time = np.isfinite(times_full)
    times_full = times_full[finite_time]
    if len(times_full) < 2:
        return None
    window_times = times_full[(times_full >= left_time) & (times_full <= right_time)]
    times = np.unique(np.concatenate(([left_time], window_times, [right_time]))).astype(float)
    if len(times) < 2:
        return None
    out = pd.DataFrame({"time": times})
    for column in columns:
        values_full = pd.to_numeric(trace_df[column], errors="coerce").to_numpy(dtype=float)[finite_time]
        finite_values = np.isfinite(values_full)
        if finite_values.sum() < 2:
            out[column] = np.nan
        else:
            out[column] = np.interp(times, times_full[finite_values], values_full[finite_values])
    return out


def interp_at_time(trace_df: pd.DataFrame, column: str, time_value: float) -> float:
    if column not in trace_df.columns or not np.isfinite(time_value):
        return np.nan
    times = pd.to_numeric(trace_df["time"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(trace_df[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(times) & np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    return float(np.interp(float(time_value), times[mask], values[mask]))


def calculate_peak_metrics_from_plot(plot_df: pd.DataFrame, peak: pd.Series) -> dict[str, float]:
    left = safe_float(peak.get("T_left", np.nan))
    right = safe_float(peak.get("T_right", np.nan))
    peak_time = safe_float(peak.get("T_peaks", np.nan))
    columns = []
    for param in PARAMS:
        columns.extend([f"{param}_signal", f"{param}_smoothed", f"{param}_baseline"])
    columns = [column for column in columns if column in plot_df.columns]
    window = interpolate_window(plot_df, left, right, columns)
    result: dict[str, float] = {}
    if window is None:
        return result
    times = window["time"].to_numpy(dtype=float)
    for param in PARAMS:
        baseline_col = f"{param}_baseline"
        if baseline_col not in window.columns:
            continue
        result[f"{param}_baseline_at_peak"] = interp_at_time(plot_df, baseline_col, peak_time)
        baseline = window[baseline_col].to_numpy(dtype=float)
        for signal_name, signal_col in (("raw", f"{param}_signal"), ("smoothed", f"{param}_smoothed")):
            if signal_col not in window.columns:
                continue
            values = window[signal_col].to_numpy(dtype=float)
            mask = np.isfinite(times) & np.isfinite(values) & np.isfinite(baseline)
            if mask.sum() >= 2:
                result[f"{param}_{signal_name}_area_vs_baseline"] = float(np.trapz(values[mask] - baseline[mask], times[mask]))
    return result


def load_voloutput_metrics(voloutput_dir: Path | None) -> pd.DataFrame:
    if not voloutput_dir:
        return pd.DataFrame()
    metrics_path = voloutput_dir / "volume_peak_metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(metrics_path)
    required = {"source_name", "set", "peak_index", "frequency_ghz"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df["set_number"] = pd.to_numeric(df["set"], errors="coerce").astype("Int64")
    df["peak_number"] = pd.to_numeric(df["peak_index"], errors="coerce").astype("Int64")
    df["frequency_key"] = df["frequency_ghz"].map(frequency_key)
    return df


def enrich_peak_metrics(
    peaks: pd.DataFrame,
    dataset_dir: Path,
    source_name: str,
    vol_metrics: pd.DataFrame,
    calculate_missing: bool,
    calc_output_dir: Path | None,
    warnings_log: list[dict[str, Any]],
) -> pd.DataFrame:
    peaks = peaks.copy()
    metric_cols = []
    for param in PARAMS:
        metric_cols.extend([
            f"{param}_baseline_at_peak",
            f"{param}_raw_area_vs_baseline",
            f"{param}_smoothed_area_vs_baseline",
        ])
    for column in metric_cols:
        if column not in peaks.columns:
            peaks[column] = np.nan

    if not vol_metrics.empty:
        source_metrics = vol_metrics[vol_metrics["source_name"] == source_name]
        lookup = {
            (int(row["set_number"]), int(row["peak_number"]), frequency_key(row["frequency_ghz"])): row
            for _, row in source_metrics.dropna(subset=["set_number", "peak_number", "frequency_ghz"]).iterrows()
        }
        for idx, peak in peaks.iterrows():
            key = (int(peak["set_number"]), int(peak["peak_number"]), frequency_key(peak["frequency_ghz"]))
            row = lookup.get(key)
            if row is None:
                continue
            for param in PARAMS:
                peaks.at[idx, f"{param}_baseline_at_peak"] = row.get(f"{param}_baseline_at_peak", peaks.at[idx, f"{param}_baseline_at_peak"])
                peaks.at[idx, f"{param}_raw_area_vs_baseline"] = row.get(f"{param}_raw_area_vs_baseline", peaks.at[idx, f"{param}_raw_area_vs_baseline"])
                peaks.at[idx, f"{param}_smoothed_area_vs_baseline"] = row.get(f"{param}_smoothed_area_vs_baseline", peaks.at[idx, f"{param}_smoothed_area_vs_baseline"])

    if calculate_missing and peaks[metric_cols].isna().any(axis=None):
        plot_index = build_plot_index(dataset_dir)
        trace_cache: dict[Path, pd.DataFrame] = {}
        calculated_rows = []
        for idx, peak in peaks.iterrows():
            if not peaks.loc[idx, metric_cols].isna().any():
                continue
            plot_path = plot_index.get((int(peak["set_number"]), frequency_key(peak["frequency_ghz"])))
            if plot_path is None:
                warnings_log.append({"dataset_dir": str(dataset_dir), "warning": f"no plot CSV for set {peak['set_number']} freq {peak['frequency_ghz']}"})
                continue
            if plot_path not in trace_cache:
                trace_cache[plot_path] = pd.read_csv(plot_path)
            calculated = calculate_peak_metrics_from_plot(trace_cache[plot_path], peak)
            calculated.update({
                "source_name": source_name,
                "set_number": int(peak["set_number"]),
                "peak_number": int(peak["peak_number"]),
                "frequency_ghz": float(peak["frequency_ghz"]),
            })
            calculated_rows.append(calculated)
            for column, value in calculated.items():
                if column in peaks.columns and pd.isna(peaks.at[idx, column]):
                    peaks.at[idx, column] = value
        if calculated_rows and calc_output_dir is not None:
            calc_output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(calculated_rows).to_csv(calc_output_dir / f"{source_name}_form_sets_peak_metrics.csv", index=False)

    return peaks


def load_dataset_peaks(
    raw_root: Path,
    dataset_subdir: str,
    dataset_info: dict[str, Any],
    mode: str,
    vol_metrics: pd.DataFrame,
    form_cfg: dict[str, Any],
    warnings_log: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    dataset_dir = raw_root / dataset_subdir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    frames = []
    original_peak_frames = []
    for mat_path in sorted(dataset_dir.glob("*.mat"), key=lambda p: (parse_set_number(p) or 0, p.name)):
        peaks = read_processed_mat(mat_path, warnings_log)
        if peaks.empty:
            continue
        peaks["dataset_subdir"] = dataset_subdir
        peaks["label"] = dataset_info.get("label", dataset_subdir)
        peaks["device"] = dataset_info.get("device", "")
        original_peak_frames.append(peaks.copy())
        set_number = int(peaks["set_number"].dropna().iloc[0]) if peaks["set_number"].notna().any() else parse_set_number(mat_path)
        pairs = load_pairs_csv(dataset_dir, int(set_number)) if set_number is not None else None
        peaks = apply_pair_status(peaks, pairs, mode, dataset_dir, warnings_log)
        if peaks.empty:
            continue
        frames.append(peaks)

    original_peak_counts = summarize_original_peaks_by_set(pd.concat(original_peak_frames, ignore_index=True)) if original_peak_frames else []

    if not frames:
        warnings_log.append({"dataset_subdir": dataset_subdir, "warning": f"no candidate peaks after mode filter {mode}"})
        return pd.DataFrame(), original_peak_counts

    all_peaks = pd.concat(frames, ignore_index=True)
    all_peaks = enrich_peak_metrics(
        all_peaks,
        dataset_dir=dataset_dir,
        source_name=dataset_subdir,
        vol_metrics=vol_metrics,
        calculate_missing=bool(form_cfg.get("calculations", {}).get("calculate_missing_in_script", False)),
        calc_output_dir=Path(form_cfg.get("calculations", {}).get("calc_output_dir")) if form_cfg.get("calculations", {}).get("calc_output_dir") else None,
        warnings_log=warnings_log,
    )
    sort_cols = ["frequency_ghz", "set_number", "pair_role", "pair_id", "peak_number", "T_peaks"]
    return all_peaks.sort_values(sort_cols).reset_index(drop=True), original_peak_counts


def get_output_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    form_cfg = config.get("form_sets", {})
    outputs = form_cfg.get("outputs")
    if outputs:
        return outputs

    datasets = form_cfg.get("datasets") or {
        "622BactVID": {"label": "Bact_B_Sub", "device": "68-10u"},
    }
    return [{
        "name": Path(get_dataset_processed_subdir(config)).name,
        "mode": form_cfg.get("mode", "all"),
        "datasets": datasets,
    }]


def sample_streams_for_mode(peaks: pd.DataFrame, mode: str) -> list[tuple[str, pd.DataFrame]]:
    if mode == "pairs_only" and "pair_role" in peaks.columns:
        streams = []
        for role in ("peak_1", "peak_2"):
            role_df = peaks[peaks["pair_role"] == role].copy()
            if not role_df.empty:
                streams.append((role, role_df))
        return streams
    return [("peaks", peaks)]


def summarize_original_peaks_by_set(peaks: pd.DataFrame) -> list[dict[str, Any]]:
    if peaks.empty:
        return []

    group_cols = ["dataset_subdir", "set_number", "frequency_ghz"]
    summary = (
        peaks.groupby(group_cols, dropna=False)
        .agg(
            original_peak_count=("peak_number", "count"),
            min_peak_number=("peak_number", "min"),
            max_peak_number=("peak_number", "max"),
            source_mats=("source_mat", lambda values: ", ".join(sorted(set(str(value) for value in values if pd.notna(value))))),
        )
        .reset_index()
        .sort_values(["dataset_subdir", "set_number", "frequency_ghz"])
    )

    records = []
    for _, row in summary.iterrows():
        records.append({
            "dataset_subdir": str(row["dataset_subdir"]),
            "set_number": int(row["set_number"]) if pd.notna(row["set_number"]) else None,
            "frequency_ghz": float(row["frequency_ghz"]) if pd.notna(row["frequency_ghz"]) else None,
            "original_peak_count": int(row["original_peak_count"]),
            "min_peak_number": int(row["min_peak_number"]) if pd.notna(row["min_peak_number"]) else None,
            "max_peak_number": int(row["max_peak_number"]) if pd.notna(row["max_peak_number"]) else None,
            "source_mats": row["source_mats"],
        })
    return records


def build_samples_for_dataset(
    peaks: pd.DataFrame,
    mode: str,
    missing_strategy: str,
    warnings_log: list[dict[str, Any]],
    freqs: Any = "all",
) -> tuple[pd.DataFrame, pd.DataFrame, list[float], list[dict[str, Any]]]:
    if peaks.empty:
        return pd.DataFrame(), pd.DataFrame(), [], []

    all_dS_rows = []
    all_full_rows = []
    available_freqs = sorted(peaks["frequency_ghz"].dropna().map(frequency_key).unique(), key=float)
    selected_freqs = normalize_frequency_selection(freqs)
    all_freqs = available_freqs if selected_freqs is None else sorted(selected_freqs, key=float)
    missing_selected = sorted(set(all_freqs) - set(available_freqs), key=float)
    if missing_selected:
        dataset_subdir = peaks["dataset_subdir"].iloc[0] if "dataset_subdir" in peaks.columns and not peaks.empty else ""
        warnings_log.append({
            "dataset_subdir": dataset_subdir,
            "warning": f"selected frequencies have no peaks in this dataset and will be blank if samples are formed: {missing_selected}",
        })
    duplicate_counts: dict[tuple[Any, ...], int] = {}

    for stream_name, stream_df in sample_streams_for_mode(peaks, mode):
        by_freq = {
            frequency_key(freq): group.sort_values(["set_number", "pair_id", "peak_number", "T_peaks"]).reset_index(drop=True)
            for freq, group in stream_df.groupby("frequency_ghz")
        }
        counts = {freq_key: len(by_freq.get(freq_key, [])) for freq_key in [frequency_key(freq) for freq in all_freqs]}
        nonzero_counts = [count for count in counts.values() if count > 0]
        if not nonzero_counts:
            continue
        min_count = min(nonzero_counts)
        max_count = max(nonzero_counts)
        for freq, count in counts.items():
            if count == 0:
                warnings_log.append({"dataset_subdir": stream_df["dataset_subdir"].iloc[0], "warning": f"frequency {freq} has no peaks for stream {stream_name}; blank columns will be used"})

        for sample_idx in range(max_count):
            dS_row: dict[str, Any] = {}
            full_row: dict[str, Any] = {}
            ext = sample_idx >= min_count
            group_number = np.nan
            first_label = stream_df["label"].iloc[0]
            first_dataset = stream_df["dataset_subdir"].iloc[0]
            first_device = stream_df["device"].iloc[0]

            dS_row.update({"label": first_label, "group_number": np.nan, "dataset_subdir": first_dataset, "ext": bool(ext), "sample_stream": stream_name})
            full_row.update({"label": first_label, "group_number": np.nan, "dataset_subdir": first_dataset, "ext": bool(ext), "device": first_device, "sample_stream": stream_name})

            for freq in all_freqs:
                freq_key = frequency_key(freq)
                freq_lbl = frequency_label(freq)
                candidates = by_freq.get(freq_key, pd.DataFrame())
                selected = None
                if not candidates.empty:
                    if sample_idx < len(candidates):
                        selected = candidates.iloc[sample_idx]
                    elif missing_strategy in {"duplicate_cycle", "duplicate_last"}:
                        # Ext rows reuse shorter frequency lists cyclically: p1, p2, p1, p2, ...
                        selected = candidates.iloc[sample_idx % len(candidates)]
                        duplicate_key = (
                            str(selected.get("dataset_subdir", "")),
                            int(selected["set_number"]) if pd.notna(selected.get("set_number", np.nan)) else None,
                            float(selected["frequency_ghz"]) if pd.notna(selected.get("frequency_ghz", np.nan)) else None,
                            int(selected["peak_number"]) if pd.notna(selected.get("peak_number", np.nan)) else None,
                            stream_name,
                        )
                        duplicate_counts[duplicate_key] = duplicate_counts.get(duplicate_key, 0) + 1
                        ext = True
                        dS_row["ext"] = True
                        full_row["ext"] = True
                    elif missing_strategy == "blank":
                        selected = None
                    else:
                        raise ValueError(f"Unsupported missing_frequency_strategy: {missing_strategy}")

                if selected is not None and pd.isna(group_number):
                    group_number = int(selected["set_number"])
                    dS_row["group_number"] = group_number
                    full_row["group_number"] = group_number

                for param, source_col in DS_COLUMNS.items():
                    col_name = f"{freq_lbl}_dS{param[1:]}"
                    value = selected.get(source_col, np.nan) if selected is not None else np.nan
                    dS_row[col_name] = value
                    full_row[col_name] = value

                full_row[f"{freq_lbl}_lr_timewidth"] = selected.get("lr_timewidth", np.nan) if selected is not None else np.nan
                full_row[f"{freq_lbl}_pair_speed"] = selected.get("pair_speed_um_per_s", np.nan) if selected is not None else np.nan
                full_row[f"{freq_lbl}_peak_time"] = selected.get("T_peaks", np.nan) if selected is not None else np.nan
                full_row[f"{freq_lbl}_peak_number"] = selected.get("peak_number", np.nan) if selected is not None else np.nan
                full_row[f"{freq_lbl}_set_number"] = selected.get("set_number", np.nan) if selected is not None else np.nan
                full_row[f"{freq_lbl}_pair_id"] = selected.get("pair_id", np.nan) if selected is not None else np.nan
                full_row[f"{freq_lbl}_pair_role"] = selected.get("pair_role", "") if selected is not None else ""

                for param in PARAMS:
                    full_row[f"{freq_lbl}_{param}_baseline_at_peak"] = selected.get(f"{param}_baseline_at_peak", np.nan) if selected is not None else np.nan
                    full_row[f"{freq_lbl}_{param}_lr_raw_minus_baseline"] = selected.get(f"{param}_raw_area_vs_baseline", np.nan) if selected is not None else np.nan
                    full_row[f"{freq_lbl}_{param}_lr_smoothed_minus_baseline"] = selected.get(f"{param}_smoothed_area_vs_baseline", np.nan) if selected is not None else np.nan

            all_dS_rows.append(dS_row)
            all_full_rows.append(full_row)

    duplication_summary = [
        {
            "dataset_subdir": dataset_subdir,
            "set_number": set_number,
            "frequency_ghz": frequency,
            "peak_number": peak_number,
            "sample_stream": sample_stream,
            "duplicated_times": count,
        }
        for (dataset_subdir, set_number, frequency, peak_number, sample_stream), count in sorted(
            duplicate_counts.items(),
            key=lambda item: (
                item[0][0],
                -1 if item[0][1] is None else item[0][1],
                math.inf if item[0][2] is None else item[0][2],
                -1 if item[0][3] is None else item[0][3],
                item[0][4],
            ),
        )
    ]
    return pd.DataFrame(all_dS_rows), pd.DataFrame(all_full_rows), [float(freq) for freq in all_freqs], duplication_summary


def parquet_engine_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except Exception:
        return False


def write_table(df: pd.DataFrame, parquet_path: Path, write_csv_fallback: bool, warnings_log: list[dict[str, Any]]) -> dict[str, str | None]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = parquet_path.with_suffix(".csv")
    result = {"parquet": str(parquet_path), "csv_fallback": None}
    if parquet_engine_available():
        df.to_parquet(parquet_path, index=False)
    else:
        warning = "No parquet engine found (install pyarrow or fastparquet). Parquet was not written."
        warnings_log.append({"file": str(parquet_path), "warning": warning})
        print(f"Warning: {warning} Writing CSV fallback: {csv_path}")
        if write_csv_fallback:
            df.to_csv(csv_path, index=False)
            result["csv_fallback"] = str(csv_path)
        return result

    if write_csv_fallback:
        df.to_csv(csv_path, index=False)
        result["csv_fallback"] = str(csv_path)
    return result


def summarize_missing_values(df: pd.DataFrame, table_name: str) -> dict[str, Any]:
    """Describe missing values by row and by source subdirectory.

    ``row_number`` is the zero-based row position in the exported table. It is
    stable for the generated parquet/CSV file and lets a bad row be found with
    ``df.iloc[row_number]``. ``dataset_subdir`` is copied from the generated row
    when present; otherwise it is left blank.
    """
    if df.empty:
        return {"table": table_name, "total_missing_values": 0, "row_count_with_missing_values": 0, "rows_with_missing_values": [], "summary_by_subdir": []}

    missing_mask = df.isna()
    row_records = []
    for row_number, row_missing in missing_mask.iterrows():
        missing_columns = row_missing.index[row_missing].tolist()
        if not missing_columns:
            continue
        row = df.loc[row_number]
        row_records.append({
            "row_number": int(row_number),
            "dataset_subdir": "" if "dataset_subdir" not in df.columns or pd.isna(row.get("dataset_subdir")) else str(row.get("dataset_subdir")),
            "group_number": None if "group_number" not in df.columns or pd.isna(row.get("group_number")) else int(row.get("group_number")),
            "sample_stream": "" if "sample_stream" not in df.columns or pd.isna(row.get("sample_stream")) else str(row.get("sample_stream")),
            "missing_value_count": int(len(missing_columns)),
            "missing_columns": [str(column) for column in missing_columns],
        })

    summary_records = []
    if "dataset_subdir" in df.columns:
        subdir_values = df["dataset_subdir"].fillna("").astype(str)
    else:
        subdir_values = pd.Series([""] * len(df), index=df.index)

    for subdir in sorted(subdir_values.unique()):
        subdir_mask = subdir_values == subdir
        subdir_missing = missing_mask.loc[subdir_mask]
        for column, count in subdir_missing.sum(axis=0).items():
            count = int(count)
            if count:
                summary_records.append({
                    "dataset_subdir": subdir,
                    "column": str(column),
                    "missing_value_count": count,
                })

    return {
        "table": table_name,
        "total_missing_values": int(missing_mask.sum().sum()),
        "row_count_with_missing_values": int(len(row_records)),
        "rows_with_missing_values": row_records,
        "summary_by_subdir": summary_records,
    }


def write_markdown(metadata: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Formed Dataset: {metadata['name']}",
        "",
        f"- Mode: `{metadata['mode']}`",
        f"- Created UTC: `{metadata['created_utc']}`",
        f"- Number of frequencies: `{len(metadata['frequencies_ghz'])}`",
        f"- Frequencies GHz: `{metadata['frequencies_ghz']}`",
        f"- dS rows: `{metadata['row_counts']['dS']}`",
        f"- Full rows: `{metadata['row_counts']['full']}`",
        "",
        "## Datasets",
    ]
    for subdir, info in metadata["datasets"].items():
        lines.append(f"- `{subdir}`: label=`{info.get('label', '')}`, device=`{info.get('device', '')}`")
    lines.extend([
        "",
        "## Original Peak Counts By Set/Frequency",
    ])
    peak_counts = metadata.get("original_peak_counts", [])
    if peak_counts:
        lines.extend([
            "| dataset_subdir | set_number | frequency_ghz | original_peak_count | min_peak_number | max_peak_number | source_mats |",
            "|---|---:|---:|---:|---:|---:|---|",
        ])
        for row in peak_counts:
            lines.append(
                f"| `{row.get('dataset_subdir', '')}` | {row.get('set_number', '')} | {row.get('frequency_ghz', '')} | "
                f"{row.get('original_peak_count', 0)} | {row.get('min_peak_number', '')} | {row.get('max_peak_number', '')} | "
                f"`{row.get('source_mats', '')}` |"
            )
    else:
        lines.append("- None")
    lines.extend([
        "",
        "## Duplicated Peaks Used For Extended Rows",
    ])
    duplicated_peaks = metadata.get("duplicated_peak_counts", [])
    if duplicated_peaks:
        lines.extend([
            "| dataset_subdir | set_number | frequency_ghz | peak_number | sample_stream | duplicated_times |",
            "|---|---:|---:|---:|---|---:|",
        ])
        for row in duplicated_peaks:
            lines.append(
                f"| `{row.get('dataset_subdir', '')}` | {row.get('set_number', '')} | {row.get('frequency_ghz', '')} | "
                f"{row.get('peak_number', '')} | `{row.get('sample_stream', '')}` | {row.get('duplicated_times', 0)} |"
            )
    else:
        lines.append("- None")
    lines.extend([
        "",
        "## Missing Values Summary",
    ])
    missing_values = metadata.get("missing_values", {})
    if missing_values:
        lines.extend([
            "| table | total_missing_values | rows_with_missing_values | summary_entries_by_subdir_column |",
            "|---|---:|---:|---:|",
        ])
        for table_name in ("dS", "full"):
            table_missing = missing_values.get(table_name, {})
            lines.append(
                f"| `{table_name}` | {table_missing.get('total_missing_values', 0)} | "
                f"{table_missing.get('row_count_with_missing_values', 0)} | "
                f"{len(table_missing.get('summary_by_subdir', []))} |"
            )
    else:
        lines.append("- None")
    lines.extend([
        "",
        "## Output Files",
        f"- dS parquet: `{metadata['outputs']['dS'].get('parquet')}`",
        f"- dS CSV fallback: `{metadata['outputs']['dS'].get('csv_fallback')}`",
        f"- Full parquet: `{metadata['outputs']['full'].get('parquet')}`",
        f"- Full CSV fallback: `{metadata['outputs']['full'].get('csv_fallback')}`",
        "",
        "## dS Columns",
        ", ".join(metadata["columns"]["dS"]),
        "",
        "## Full Columns",
        ", ".join(metadata["columns"]["full"]),
        "",
        "## Warnings",
    ])
    if metadata["warnings"]:
        for warning in metadata["warnings"]:
            lines.append(f"- `{warning}`")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_") or "formatted_dataset"


def resolve_form_config(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    form_section = config.get("form_sets", {})
    dataset_processed_subdir = get_dataset_processed_subdir(config)
    raw_root = Path(paths.get("dataset_raw_subdir", ".")).expanduser()
    output_dir = Path(paths["scratch_root"]) / dataset_processed_subdir
    run_dir = Path(paths.get("run_dir", Path(__file__).resolve().parents[1]))
    metadata_dir = Path(form_section.get("metadata_dir", run_dir / "scripts" / "data_MDs"))
    voloutput_dir = Path(form_section.get("voloutput_dir", raw_root / "VolOutput"))
    calc_output_dir = voloutput_dir / "form_sets_calc"

    form_cfg = copy.deepcopy(form_section)
    form_cfg["dataset_processed_subdir"] = dataset_processed_subdir
    form_cfg.setdefault("missing_frequency_strategy", "duplicate_cycle")
    form_cfg.setdefault("write_csv_fallback", True)
    form_cfg.setdefault("travel_distance_um", DEFAULT_TRAVEL_DISTANCE_UM)
    form_cfg.setdefault("freqs", config.get("training", {}).get("freqs", "all"))
    form_cfg.setdefault("calculations", {})
    form_cfg["calculations"].setdefault("use_voloutput", True)
    form_cfg["calculations"].setdefault("calculate_missing_in_script", False)
    form_cfg["calculations"].setdefault("calc_output_dir", str(calc_output_dir))
    form_cfg["raw_root"] = str(raw_root)
    form_cfg["output_dir"] = str(output_dir)
    form_cfg["metadata_dir"] = str(metadata_dir)
    form_cfg["voloutput_dir"] = str(voloutput_dir)
    return form_cfg


def format_processed_data(config: dict[str, Any]) -> list[dict[str, Any]]:
    global DEFAULT_TRAVEL_DISTANCE_UM
    config = load_recreate_config_if_requested(config)
    form_cfg = resolve_form_config(config)
    raw_root = Path(form_cfg["raw_root"])
    output_dir = Path(form_cfg["output_dir"])
    metadata_dir = Path(form_cfg["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    travel_distance = float(form_cfg.get("travel_distance_um", DEFAULT_TRAVEL_DISTANCE_UM))
    DEFAULT_TRAVEL_DISTANCE_UM = travel_distance

    vol_metrics = load_voloutput_metrics(Path(form_cfg["voloutput_dir"])) if form_cfg.get("calculations", {}).get("use_voloutput", True) else pd.DataFrame()
    output_metadata = []

    for output_spec in get_output_specs(config):
        name = sanitize_name(str(output_spec.get("name", "formatted_dataset")))
        mode = output_spec.get("mode", form_cfg.get("mode", "all"))
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode for output {name}: {mode}. Valid modes: {sorted(VALID_MODES)}")
        datasets = output_spec.get("datasets", {})
        if not datasets:
            raise ValueError(f"Output {name} does not define any datasets")
        output_freqs = output_spec.get("freqs", form_cfg.get("freqs", "all"))

        warnings_log: list[dict[str, Any]] = []
        dS_frames = []
        full_frames = []
        all_frequencies = set()
        original_peak_counts = []
        duplicated_peak_counts = []

        for dataset_subdir, dataset_info in datasets.items():
            peaks, dataset_peak_counts = load_dataset_peaks(raw_root, dataset_subdir, dataset_info or {}, mode, vol_metrics, form_cfg, warnings_log)
            dS_df, full_df, frequencies, dataset_duplicates = build_samples_for_dataset(
                peaks,
                mode,
                form_cfg["missing_frequency_strategy"],
                warnings_log,
                output_freqs,
            )
            original_peak_counts.extend(dataset_peak_counts)
            duplicated_peak_counts.extend(dataset_duplicates)
            if not dS_df.empty:
                dS_frames.append(dS_df)
            if not full_df.empty:
                full_frames.append(full_df)
            all_frequencies.update(frequencies)

        dS_out = pd.concat(dS_frames, ignore_index=True, sort=False) if dS_frames else pd.DataFrame()
        full_out = pd.concat(full_frames, ignore_index=True, sort=False) if full_frames else pd.DataFrame()

        dS_path = output_dir / f"{name}_dS.parquet"
        full_path = output_dir / f"{name}_full.parquet"
        dS_result = write_table(dS_out, dS_path, bool(form_cfg.get("write_csv_fallback", True)), warnings_log)
        full_result = write_table(full_out, full_path, bool(form_cfg.get("write_csv_fallback", True)), warnings_log)

        json_path = metadata_dir / f"{name}.JSON"
        md_path = output_dir / f"{name}.md"
        metadata = {
            "name": name,
            "mode": mode,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "effective_config": config,
            "form_sets_config": form_cfg,
            "datasets": datasets,
            "configured_freqs": frequency_selection_summary(output_freqs),
            "frequencies_ghz": sorted(float(freq) for freq in all_frequencies),
            "row_counts": {"dS": int(len(dS_out)), "full": int(len(full_out))},
            "missing_values": {
                "dS": summarize_missing_values(dS_out, "dS"),
                "full": summarize_missing_values(full_out, "full"),
            },
            "original_peak_counts": original_peak_counts,
            "duplicated_peak_counts": duplicated_peak_counts,
            "outputs": {"dS": dS_result, "full": full_result},
            "columns": {"dS": list(dS_out.columns), "full": list(full_out.columns)},
            "warnings": [json.dumps(item, sort_keys=True) for item in warnings_log],
            "metadata_json": str(json_path),
            "metadata_md": str(md_path),
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        write_markdown(metadata, md_path)
        output_metadata.append(metadata)

        print(f"Wrote output {name}: dS rows={len(dS_out)}, full rows={len(full_out)}")
        print(f"  Metadata JSON: {json_path}")
        print(f"  Metadata MD: {md_path}")

    manifest_name = sanitize_name(Path(form_cfg["dataset_processed_subdir"]).name)
    manifest_path = metadata_dir / f"{manifest_name}.JSON"
    manifest = {
        "name": manifest_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "effective_config": config,
        "outputs": [
            {
                "name": item["name"],
                "mode": item["mode"],
                "frequencies_ghz": item["frequencies_ghz"],
                "row_counts": item["row_counts"],
                "outputs": item["outputs"],
                "metadata_json": item["metadata_json"],
                "metadata_md": item["metadata_md"],
            }
            for item in output_metadata
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote recreate manifest: {manifest_path}")
    return output_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Form ML-ready bacteria/particle sets from processed MAD data")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    format_processed_data(config)


if __name__ == "__main__":
    main()
