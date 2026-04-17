import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest

import torch
from mmcv import Config


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools/run_a2_two_stage_training.py"


def load_runner_module(module_name: str, env_overrides: dict[str, str] | None = None):
    env_overrides = env_overrides or {}
    saved = {key: os.environ.get(key) for key in env_overrides}
    try:
        for key, value in env_overrides.items():
            os.environ[key] = value
        spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


A2 = load_runner_module("a2_two_stage_training")


def make_eval_metrics(full_idsw, full_frag, seam_fp, seam_hota, seam_idf1, non_hota, non_idf1, non_fp, non_idsw, non_frag):
    return {
        "full": {
            "HOTA": 30.0,
            "IDF1": 35.0,
            "FP": 100,
            "IDSW": full_idsw,
            "Frag": full_frag,
        },
        "seam": {
            "HOTA": seam_hota,
            "IDF1": seam_idf1,
            "FP": seam_fp,
            "IDSW": 10,
            "Frag": 10,
        },
        "non_seam": {
            "HOTA": non_hota,
            "IDF1": non_idf1,
            "FP": non_fp,
            "IDSW": non_idsw,
            "Frag": non_frag,
        },
    }


class RunSpecTest(unittest.TestCase):
    def test_run_slug_and_pair_key(self):
        spec = A2.RunSpec(
            stage="s1",
            budget=15000,
            seed=1,
            variant="baseline",
            job="all",
            gpu_request="auto",
            batch_size=4,
            effective_iters=15000,
        )
        self.assertEqual(spec.run_slug, "a2_s1_15000_seed1_baseline")
        self.assertEqual(spec.pair_key, "s1_15000_seed1")

    def test_preflight_slug_includes_effective_iters(self):
        spec = A2.RunSpec(
            stage="s1",
            budget=15000,
            seed=0,
            variant="a",
            job="train",
            gpu_request="1",
            batch_size=4,
            effective_iters=100,
        )
        self.assertTrue(spec.is_preflight)
        self.assertEqual(spec.run_slug, "a2_s1_b15000_i100_seed0_a")

    def test_env_overrides_can_redirect_state_root_and_a_config(self):
        custom_root = pathlib.Path("/tmp/custom_rolllite_root")
        custom_config = pathlib.Path("/tmp/custom_rolllite.py")
        module = load_runner_module(
            "a2_two_stage_training_env_override",
            env_overrides={
                "OMNITRACK_A2_ROOT": str(custom_root),
                "OMNITRACK_A2_A_CONFIG": str(custom_config),
                "OMNITRACK_A2_SUMMARY_TITLE": "A2 Roll-Lite Clean Stage Gate Summary",
            },
        )
        self.assertEqual(module.A2_ROOT, custom_root)
        self.assertEqual(module.A_CONFIG, custom_config)
        self.assertEqual(module.SUMMARY_TITLE, "A2 Roll-Lite Clean Stage Gate Summary")


class CommandAndConfigTest(unittest.TestCase):
    def test_train_command_includes_seed_and_deterministic(self):
        spec = A2.RunSpec(
            stage="s1",
            budget=15000,
            seed=2,
            variant="baseline",
            job="train",
            gpu_request="1",
            batch_size=4,
            effective_iters=15000,
        )
        command = A2.build_train_command(
            spec=spec,
            gpu_id="1",
            cfg_path=pathlib.Path("/tmp/train.py"),
            work_dir=pathlib.Path("/tmp/workdir"),
        )
        self.assertIn("--seed 2", command)
        self.assertIn("--deterministic", command)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", command)

    def test_train_config_has_backbone_freeze_and_whitelist(self):
        spec = A2.RunSpec(
            stage="s1",
            budget=30000,
            seed=0,
            variant="a",
            job="train",
            gpu_request="auto",
            batch_size=4,
            effective_iters=30000,
        )
        text = A2.train_config_text(
            spec,
            train_sequences=["seq_train_a", "seq_train_b"],
            val_sequences=["seq_val_a"],
        )
        self.assertIn("find_unused_parameters = True", text)
        self.assertIn("frozen_stages=4", text)
        self.assertIn("norm_eval=True", text)
        self.assertIn("lr_mult=0.0", text)
        self.assertIn("sequence_whitelist=['seq_train_a', 'seq_train_b']", text)
        self.assertIn("sequence_whitelist=['seq_val_a']", text)
        self.assertIn("max_iters=30000", text)
        self.assertIn("samples_per_gpu=4", text)

    def test_eval_config_forces_tbd_hybridsort_and_roll(self):
        spec = A2.RunSpec(
            stage="s2",
            budget=60000,
            seed=1,
            variant="baseline",
            job="proxy_val",
            gpu_request="auto",
            batch_size=4,
            effective_iters=60000,
        )
        eval_spec = A2.EvalSpec(
            name="card2_roll",
            sequence_names=("clark-center-2019-02-28_1",),
            eval_roll_px=940,
            inverse_roll_px=940,
        )
        text = A2.eval_config_text(spec, eval_spec)
        self.assertIn("tracking_mode='tbd'", text)
        self.assertIn("tbd_backend='hybridsort'", text)
        self.assertIn("eval_roll_px=940", text)
        self.assertIn("sequence_whitelist=['clark-center-2019-02-28_1']", text)

    def test_diag_command_uses_repo_absolute_script_path(self):
        spec = A2.RunSpec(
            stage="s1",
            budget=15000,
            seed=0,
            variant="baseline",
            job="proxy_val",
            gpu_request="auto",
            batch_size=4,
            effective_iters=15000,
        )
        command = A2.build_diag_command(
            spec=spec,
            eval_name="proxy_val",
            gt_dir=pathlib.Path("/tmp/gt"),
            trackers_dir=pathlib.Path("/tmp/trackers"),
            diag_out=pathlib.Path("/tmp/diag"),
        )
        self.assertIn(str(REPO_ROOT / "tools/diag_seam_metric_split.py"), command)


class GateLogicTest(unittest.TestCase):
    def build_record(self, proxy_metrics, orig_metrics, roll_metrics):
        return {
            "evaluations": {
                "proxy_val": {"metrics": proxy_metrics},
                "card2_orig": {"metrics": orig_metrics},
                "card2_roll": {"metrics": roll_metrics},
                "card2_gap": A2.card2_gap(orig_metrics, roll_metrics),
            }
        }

    def test_single_seed_gate_allows_zero_gap_tolerance_of_one(self):
        baseline = self.build_record(
            proxy_metrics=make_eval_metrics(5, 5, 10, 50.0, 60.0, 40.0, 45.0, 20, 10, 10),
            orig_metrics=make_eval_metrics(8, 8, 0, 0, 0, 0, 0, 0, 0, 0),
            roll_metrics=make_eval_metrics(8, 8, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        a_record = self.build_record(
            proxy_metrics=make_eval_metrics(4, 4, 8, 50.2, 60.1, 39.9, 44.9, 20, 10, 10),
            orig_metrics=make_eval_metrics(8, 8, 0, 0, 0, 0, 0, 0, 0, 0),
            roll_metrics=make_eval_metrics(9, 9, 1, 0, 0, 0, 0, 0, 0, 0),
        )
        gate = A2.build_single_seed_gate(baseline, a_record)
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["card2_gap"]["FP_seam"]["a_gap"], 1)

    def test_single_seed_gate_fails_when_zero_gap_tolerance_exceeded(self):
        baseline = self.build_record(
            proxy_metrics=make_eval_metrics(5, 5, 10, 50.0, 60.0, 40.0, 45.0, 20, 10, 10),
            orig_metrics=make_eval_metrics(8, 8, 0, 0, 0, 0, 0, 0, 0, 0),
            roll_metrics=make_eval_metrics(8, 8, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        a_record = self.build_record(
            proxy_metrics=make_eval_metrics(4, 4, 8, 50.2, 60.1, 39.9, 44.9, 20, 10, 10),
            orig_metrics=make_eval_metrics(8, 8, 0, 0, 0, 0, 0, 0, 0, 0),
            roll_metrics=make_eval_metrics(10, 10, 2, 0, 0, 0, 0, 0, 0, 0),
        )
        gate = A2.build_single_seed_gate(baseline, a_record)
        self.assertFalse(gate["pass"])
        joined = "\n".join(gate["reasons"])
        self.assertIn("zero-gap tolerance", joined)

    def test_budget_gate_requires_two_passes_and_positive_means(self):
        pair_records = [
            {
                "seed": 0,
                "single_seed_gate": {"pass": True},
                "baseline_gap": {"IDSW": 10, "Frag": 10, "FP_seam": 10},
                "a_gap": {"IDSW": 5, "Frag": 5, "FP_seam": 5},
                "proxy_val_seam_delta": {"HOTA": 0.5, "IDF1": 0.5},
                "proxy_val_non_seam_delta": {"HOTA": -0.1, "IDF1": -0.1},
                "proxy_val_non_seam_relative": {"FP": 0.01, "IDSW": 0.01, "Frag": 0.01},
            },
            {
                "seed": 1,
                "single_seed_gate": {"pass": True},
                "baseline_gap": {"IDSW": 8, "Frag": 8, "FP_seam": 8},
                "a_gap": {"IDSW": 6, "Frag": 6, "FP_seam": 6},
                "proxy_val_seam_delta": {"HOTA": 0.2, "IDF1": 0.1},
                "proxy_val_non_seam_delta": {"HOTA": -0.05, "IDF1": -0.05},
                "proxy_val_non_seam_relative": {"FP": 0.01, "IDSW": 0.01, "Frag": 0.01},
            },
            {
                "seed": 2,
                "single_seed_gate": {"pass": False},
                "baseline_gap": {"IDSW": 9, "Frag": 9, "FP_seam": 9},
                "a_gap": {"IDSW": 7, "Frag": 7, "FP_seam": 7},
                "proxy_val_seam_delta": {"HOTA": 0.0, "IDF1": 0.0},
                "proxy_val_non_seam_delta": {"HOTA": -0.1, "IDF1": -0.1},
                "proxy_val_non_seam_relative": {"FP": 0.01, "IDSW": 0.01, "Frag": 0.01},
            },
        ]
        gate = A2.build_budget_gate(pair_records)
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["pass_count"], 2)

    def test_refresh_pair_views_prefers_full_budget_over_preflight(self):
        baseline_preflight = self.build_record(
            proxy_metrics=make_eval_metrics(5, 5, 10, 50.0, 60.0, 40.0, 45.0, 20, 10, 10),
            orig_metrics=make_eval_metrics(8, 8, 0, 0, 0, 0, 0, 0, 0, 0),
            roll_metrics=make_eval_metrics(9, 9, 1, 0, 0, 0, 0, 0, 0, 0),
        )
        baseline_full = self.build_record(
            proxy_metrics=make_eval_metrics(5, 5, 10, 50.0, 60.0, 40.0, 45.0, 20, 10, 10),
            orig_metrics=make_eval_metrics(8, 8, 0, 0, 0, 0, 0, 0, 0, 0),
            roll_metrics=make_eval_metrics(8, 8, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        a_full = self.build_record(
            proxy_metrics=make_eval_metrics(4, 4, 8, 50.2, 60.1, 39.9, 44.9, 20, 10, 10),
            orig_metrics=make_eval_metrics(8, 8, 0, 0, 0, 0, 0, 0, 0, 0),
            roll_metrics=make_eval_metrics(9, 9, 1, 0, 0, 0, 0, 0, 0, 0),
        )

        manifest = {
            "runs": {
                "a2_s1_b15000_i100_seed0_baseline": {
                    "run_slug": "a2_s1_b15000_i100_seed0_baseline",
                    "pair_key": "s1_15000_seed0",
                    "stage": "s1",
                    "budget": 15000,
                    "effective_iters": 100,
                    "seed": 0,
                    "variant": "baseline",
                    "prepared_at": "2026-04-13T10:00:00+08:00",
                    "evaluations": baseline_preflight["evaluations"],
                },
                "a2_s1_15000_seed0_baseline": {
                    "run_slug": "a2_s1_15000_seed0_baseline",
                    "pair_key": "s1_15000_seed0",
                    "stage": "s1",
                    "budget": 15000,
                    "effective_iters": 15000,
                    "seed": 0,
                    "variant": "baseline",
                    "prepared_at": "2026-04-13T12:00:00+08:00",
                    "evaluations": baseline_full["evaluations"],
                },
                "a2_s1_15000_seed0_a": {
                    "run_slug": "a2_s1_15000_seed0_a",
                    "pair_key": "s1_15000_seed0",
                    "stage": "s1",
                    "budget": 15000,
                    "effective_iters": 15000,
                    "seed": 0,
                    "variant": "a",
                    "prepared_at": "2026-04-13T12:00:00+08:00",
                    "evaluations": a_full["evaluations"],
                },
            }
        }

        A2.refresh_pair_and_budget_views(manifest)
        pair = manifest["pairs"]["s1_15000_seed0"]
        self.assertEqual(pair["baseline_run_slug"], "a2_s1_15000_seed0_baseline")
        self.assertEqual(pair["a_run_slug"], "a2_s1_15000_seed0_a")
        self.assertTrue(pair["single_seed_gate"]["pass"])


class BNFreezeCheckTest(unittest.TestCase):
    def test_compare_backbone_bn_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = pathlib.Path(tmp_dir)
            ref_path = tmp_root / "ref.pth"
            tgt_path = tmp_root / "tgt.pth"
            torch.save(
                {
                    "state_dict": {
                        "img_backbone.layer1.0.bn1.running_mean": torch.tensor([0.0, 1.0]),
                        "img_backbone.layer1.0.bn1.running_var": torch.tensor([1.0, 2.0]),
                    }
                },
                ref_path,
            )
            torch.save(
                {
                    "state_dict": {
                        "img_backbone.layer1.0.bn1.running_mean": torch.tensor([0.0, 1.5]),
                        "img_backbone.layer1.0.bn1.running_var": torch.tensor([1.0, 2.0]),
                    }
                },
                tgt_path,
            )
            report = A2.compare_backbone_bn(ref_path, tgt_path)
            self.assertFalse(report["pass"])
            self.assertEqual(report["changed_key_count"], 1)


class ConfigIsolationTest(unittest.TestCase):
    def test_rolllite_config_only_reduces_roll_prob(self):
        official_cfg = Config.fromfile(
            str(REPO_ROOT / "projects/configs/JRDB_OmniTrack_wt_a_circular_padding_rollaug.py")
        )
        rolllite_cfg = Config.fromfile(
            str(REPO_ROOT / "projects/configs/JRDB_OmniTrack_wt_a_circular_padding_rollaug_rolllite.py")
        )

        self.assertEqual(official_cfg.data.train.data_aug_conf["roll_prob"], 1.0)
        self.assertEqual(rolllite_cfg.data.train.data_aug_conf["roll_prob"], 0.25)

        official_dict = official_cfg._cfg_dict.to_dict()
        rolllite_dict = rolllite_cfg._cfg_dict.to_dict()
        rolllite_dict["data"]["train"]["data_aug_conf"]["roll_prob"] = 1.0
        self.assertEqual(rolllite_dict, official_dict)


if __name__ == "__main__":
    unittest.main()
