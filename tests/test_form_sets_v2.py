"""Regression checks for V2 recording discovery and sidecar isolation.

Run with the bacteria analysis environment:
    python -B -m unittest discover -s tests -p 'test_form_sets_v2.py'
"""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.io import savemat
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import form_setsV2 as form


class V2SetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "ExampleVID"
        self.recordings = self.dataset / "output_v2" / "recordings"
        self.recordings.mkdir(parents=True)

    def recording(self, name, frequency=1.5, amplitude=2.0):
        folder = self.recordings / name / "EC-1_M"
        folder.mkdir(parents=True)
        peaks = np.zeros((2, len(form.PROCESSED_DATA_COLUMNS)))
        for column, value in {
            "set": 1,
            "T_peaks": [0.5, 1.5],
            "index": [1, 2],
            "CW_Freq": frequency * 1e9,
            "dT": 1,
            "T_left": [0, 1],
            "T_right": [1, 2],
        }.items():
            peaks[:, form.PROCESSED_DATA_COLUMNS.index(column)] = value
        for column in form.DS_COLUMNS.values():
            peaks[:, form.PROCESSED_DATA_COLUMNS.index(column)] = amplitude
        savemat(folder / "EC-1.mat", {"processedData": peaks})
        trace = {"time": [0, 0.5, 1, 1.5, 2]}
        for param in form.PARAMS:
            trace[f"{param}_signal"] = np.full(5, 10 + amplitude)
            trace[f"{param}_smoothed"] = np.full(5, 10 + amplitude / 2)
            trace[f"{param}_baseline"] = np.full(5, 10)
        pd.DataFrame(trace).to_csv(folder / f"EC-1_{frequency:g}GHz_plot_data.csv", index=False)
        return folder

    def load(self, mode="all", settings=None):
        log = []
        peaks, counts = form.load_dataset_peaks(
            self.root, "ExampleVID", {"label": "Bacteria", "device": "test"},
            mode, pd.DataFrame(), settings or {}, log,
        )
        return peaks, counts, log

    def test_nested_recordings_use_their_own_traces_and_metric_exports(self):
        self.recording("batch/a", amplitude=2)
        self.recording("batch/b", amplitude=6)
        calc_dir = self.root / "calculated"
        peaks, counts, _ = self.load(settings={"calculations": {
            "calculate_missing_in_script": True, "calc_output_dir": str(calc_dir),
        }})
        self.assertEqual(len(peaks), 4)
        self.assertEqual(peaks["source_mat"].nunique(), 2)
        self.assertEqual(counts[0]["original_peak_count"], 4)
        self.assertEqual(peaks["s11m_raw_area_vs_baseline"].tolist(), [2, 2, 6, 6])
        self.assertEqual(peaks["s11m_smoothed_area_vs_baseline"].tolist(), [1, 1, 3, 3])
        exports = list(calc_dir.rglob("*_form_sets_peak_metrics.csv"))
        self.assertEqual(len(exports), 2)
        self.assertEqual(sum(len(pd.read_csv(path)) for path in exports), 4)

    def test_pair_modes_ignore_legacy_pairs_and_scope_v2_pairs_to_recording(self):
        first = self.recording("a")
        self.recording("b")
        pairs = pd.DataFrame({"pair_id": [1], "peak_1_index": [1],
                              "peak_2_index": [2], "peak_time_diff": [0.5],
                              "valid_pair": [True]})
        pairs.to_csv(self.dataset / "bact-1_pairs.csv", index=False)
        savemat(self.dataset / "bact-9.mat", {"processedData": np.ones((2, 19))})
        peaks, _, _ = self.load()
        self.assertEqual(len(peaks), 4)
        self.assertFalse(peaks["paired"].any())
        self.assertTrue(self.load("pairs_only")[0].empty)
        pairs.to_csv(first / "EC-1_pairs.csv", index=False)
        paired, _, _ = self.load("pairs_only")
        self.assertEqual(len(paired), 2)
        self.assertEqual(paired["pair_speed_um_per_s"].tolist(), [240, 240])
        unpaired, _, log = self.load("unpaired_only")
        self.assertEqual(len(unpaired), 2)
        self.assertTrue(unpaired["source_mat"].str.contains("/b/").all())
        self.assertTrue(any("pairs.csv missing" in entry["warning"] for entry in log))

    def test_missing_and_empty_v2_outputs_and_custom_output_name(self):
        self.recording("a")
        self.dataset.joinpath("output_v2").rename(self.dataset / "custom_v2")
        with self.assertRaisesRegex(FileNotFoundError, "V2 recordings directory"):
            self.load()
        peaks, _, _ = self.load(settings={"v2_output_subdir": "custom_v2"})
        self.assertEqual(len(peaks), 2)
        self.recordings.mkdir(parents=True)
        empty = self.recordings / "empty"
        empty.mkdir()
        savemat(empty / "EC-2.mat", {"processedData": np.empty((0, 19))})
        peaks, _, log = self.load()
        self.assertTrue(peaks.empty)
        self.assertTrue(any("processedData missing or empty" in entry["warning"] for entry in log))

    def test_cli_exports_existing_table_format_and_can_recreate(self):
        self.recording("a", frequency=1.5, amplitude=2)
        self.recording("b", frequency=2, amplitude=6)
        config = {
            "paths": {"dataset_raw_subdir": str(self.root), "scratch_root": str(self.root)},
            "training": {"freqs": [1.5, 2]},
            "form_sets": {
                "dataset_processed_subdir": "formed",
                "metadata_dir": str(self.root / "metadata"),
                "write_csv_fallback": True,
                "outputs": [{"name": "example", "mode": "all",
                             "datasets": {"ExampleVID": {"label": "Bacteria"}}}],
            },
        }
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        result = subprocess.run(
            [sys.executable, "-B", str(Path(form.__file__).resolve()), "--config", str(config_path)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metadata = json.loads((self.root / "metadata" / "example.JSON").read_text())
        self.assertEqual(metadata["row_counts"], {"dS": 2, "full": 2})
        self.assertEqual(metadata["form_sets_config"]["input_format"], "bacteria_mad_v2")
        self.assertEqual(metadata["form_sets_config"]["voloutput_dir"], str(self.root / "VolOutput_v2"))
        table = pd.read_csv(self.root / "formed" / "example_dS.csv")
        self.assertEqual(table["1.5GHz_dS11m"].tolist(), [2, 2])
        self.assertEqual(table["2GHz_dS11m"].tolist(), [6, 6])
        self.assertEqual(table["label"].tolist(), ["Bacteria", "Bacteria"])
        self.assertTrue((self.root / "formed" / "example.md").is_file())
        if form.parquet_engine_available():
            self.assertEqual(len(pd.read_parquet(self.root / "formed" / "example_dS.parquet")), 2)
        recreate = {"paths": {"remake_processed_data": str(self.root / "metadata" / "formed.JSON")}}
        with contextlib.redirect_stdout(io.StringIO()):
            result = form.format_processed_data(recreate)
        self.assertEqual(result[0]["row_counts"], metadata["row_counts"])


if __name__ == "__main__":
    unittest.main()
