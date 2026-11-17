"""Interactive target remapping checks; run with unittest discovery."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd


TRAINER_PATH = Path(__file__).resolve().parents[1] / "src" / "BactTrain-gpr.py"
spec = importlib.util.spec_from_file_location("bact_train_gpr", TRAINER_PATH)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TargetRemappingTests(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({"label": ["Beta", "Alpha", "Beta", "Alpha"]})
        self.cfg = {"target_column": "label", "target_mode": "raw"}
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def test_maps_each_unseen_target_once_and_preserves_known_targets(self):
        test_df = pd.DataFrame({"label": ["UnknownX", "Alpha", "UnknownY", "UnknownX", "Beta"]})
        with patch("builtins.input", side_effect=["y", "2", "yes", "1"]) as prompt:
            y_train, y_test, encoder = trainer.prepare_targets(self.train_df, test_df, self.cfg)
        self.assertEqual(encoder.classes_.tolist(), ["Alpha", "Beta"])
        self.assertEqual(encoder.inverse_transform(y_train).tolist(), self.train_df["label"].tolist())
        self.assertEqual(encoder.inverse_transform(y_test).tolist(), ["Beta", "Alpha", "Alpha", "Beta", "Beta"])
        self.assertEqual(self.cfg["test_target_remap"], {"UnknownX": "Beta", "UnknownY": "Alpha"})
        self.assertEqual(test_df["label"].tolist(), ["UnknownX", "Alpha", "UnknownY", "UnknownX", "Beta"])
        self.assertEqual(prompt.call_count, 4)
        self.assertEqual(self.output.getvalue().count("WARNING:"), 2)
        self.assertEqual(self.output.getvalue().count("1. Alpha"), 2)
        self.assertEqual(self.output.getvalue().count("2. Beta"), 2)

    def test_reprompts_for_invalid_confirmation_and_number(self):
        test_df = pd.DataFrame({"label": ["Unknown"]})
        answers = ["maybe", "", " Y ", "0", "3", "two", "1.5", "", " 2 "]
        with patch("builtins.input", side_effect=answers) as prompt:
            _, y_test, encoder = trainer.prepare_targets(self.train_df, test_df, self.cfg)
        self.assertEqual(encoder.inverse_transform(y_test).tolist(), ["Beta"])
        self.assertEqual(prompt.call_count, len(answers))
        self.assertIn("Please enter y or n.", self.output.getvalue())
        self.assertIn("Please enter a number from 1 to 2.", self.output.getvalue())

    def test_declining_any_target_quits_without_saving_partial_mapping(self):
        test_df = pd.DataFrame({"label": ["UnknownX", "UnknownY"]})
        for answers in (["n"], ["y", "1", " NO "]):
            with self.subTest(answers=answers), patch("builtins.input", side_effect=answers):
                with self.assertRaisesRegex(SystemExit, "Training cancelled"):
                    trainer.prepare_targets(self.train_df, test_df, self.cfg)
            self.assertNotIn("test_target_remap", self.cfg)

    def test_eof_and_keyboard_interrupt_cancel_at_either_prompt(self):
        test_df = pd.DataFrame({"label": ["Unknown"]})
        for error in (EOFError, KeyboardInterrupt):
            for answers in ([error()], ["y", error()]):
                with self.subTest(error=error, answers=answers), patch("builtins.input", side_effect=answers):
                    with self.assertRaisesRegex(SystemExit, "input was interrupted"):
                        trainer.prepare_targets(self.train_df, test_df, self.cfg)

    def test_matching_and_normalized_targets_do_not_prompt(self):
        scenarios = [
            (self.train_df, pd.DataFrame({"label": ["Alpha", "Beta"]}), self.cfg),
            (self.train_df, pd.DataFrame({"label": ["Alias"]}), {**self.cfg, "label_map": {"Alias": "Alpha"}}),
            (pd.DataFrame({"label": ["Bact_Train", "Particle"]}),
             pd.DataFrame({"label": ["Bact_Test"]}), {**self.cfg, "target_mode": "binary_bacteria_particle"}),
            (self.train_df, None, self.cfg),
            (self.train_df, pd.DataFrame(), self.cfg),
        ]
        for train_df, test_df, cfg in scenarios:
            with self.subTest(test_df=test_df), patch("builtins.input") as prompt:
                _, y_test, _ = trainer.prepare_targets(train_df, test_df, cfg)
                prompt.assert_not_called()
                self.assertEqual(cfg["test_target_remap"], {})
                if test_df is None or test_df.empty:
                    self.assertIsNone(y_test)

    def test_declining_stops_before_creating_run_or_model(self):
        config = {
            "paths": {"train_datasets": ["train"], "test_datasets": ["test"]},
            "training": self.cfg,
        }
        test_df = pd.DataFrame({"label": ["Unknown"]})
        with patch.object(trainer, "processed_dataset_dir", return_value=Path("unused")), \
             patch.object(trainer, "load_named_datasets", side_effect=[self.train_df, test_df]), \
             patch.object(trainer, "output_dirs") as output_dirs, \
             patch.object(trainer, "init_aim_run") as aim, \
             patch.object(trainer, "build_model") as model, \
             patch("builtins.input", return_value="n"):
            with self.assertRaisesRegex(SystemExit, "remapping was declined"):
                trainer.train(config)
        output_dirs.assert_not_called()
        aim.assert_not_called()
        model.assert_not_called()

    def test_training_artifacts_record_mapping_and_original_test_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "formed"
            dataset_dir.mkdir()
            pd.DataFrame({
                "label": ["Alpha"] * 3 + ["Beta"] * 3,
                "1GHz_dS11m": [-1.2, -1, -0.8, 0.8, 1, 1.2],
            }).to_csv(dataset_dir / "train_dS.csv", index=False)
            pd.DataFrame({
                "label": ["Unknown", "Alpha"], "1GHz_dS11m": [1, -1],
            }).to_csv(dataset_dir / "test_dS.csv", index=False)
            config = {
                "paths": {"scratch_root": str(root), "train_datasets": ["train"], "test_datasets": ["test"]},
                "form_sets": {"dataset_processed_subdir": "formed"},
                "training": {**self.cfg, "aim_enabled": False, "feature_importance": {"enabled": False}},
            }
            with patch("builtins.input", side_effect=["y", "2"]), \
                 patch.object(trainer, "save_and_log_split_evaluation_artifacts"):
                summary = trainer.train(config)
            mapping = {"Unknown": "Beta"}
            self.assertEqual(summary["test_target_remap"], mapping)
            predictions = pd.read_csv(summary["artifacts"]["test_predictions"])
            self.assertEqual(predictions["label"].tolist(), ["Unknown", "Alpha"])
            self.assertEqual(predictions["true_label"].tolist(), ["Beta", "Alpha"])
            settings = json.loads(Path(summary["artifacts"]["model_settings"]).read_text())
            self.assertEqual(settings["training_config"]["test_target_remap"], mapping)
            bundle = joblib.load(summary["artifacts"]["model_bundle"])
            self.assertEqual(bundle["training_config"]["test_target_remap"], mapping)
            saved_summary = json.loads(Path(summary["artifacts"]["metrics"]).with_name("summary.json").read_text())
            self.assertEqual(saved_summary["test_target_remap"], mapping)


if __name__ == "__main__":
    unittest.main()
