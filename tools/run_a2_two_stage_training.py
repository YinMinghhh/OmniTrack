#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
A1_ROOT = REPO_ROOT / "research/seam-a/A1_proxy_subset_screening"
A2_ROOT = REPO_ROOT / "research/seam-a/A2_two_stage_training"
MANIFEST_PATH = A2_ROOT / "manifest.json"
MATRIX_PATH = A2_ROOT / "experiment_matrix.csv"
SUMMARY_PATH = A2_ROOT / "stage_gate_summary.md"
LOCK_PATH = A2_ROOT / ".manifest.lock"
COMMAND_ROOT = A2_ROOT / "commands"
CONFIG_ROOT = A2_ROOT / "generated_configs"
RUN_META_ROOT = A2_ROOT / "run_meta"

BASELINE_CONFIG = REPO_ROOT / "projects/configs/JRDB_OmniTrack.py"
A_CONFIG = REPO_ROOT / "projects/configs/JRDB_OmniTrack_wt_a_circular_padding_rollaug.py"
BASELINE_CHECKPOINT = REPO_ROOT / "ckpt/jrdb2019_baseline_iter_135900.pth"
WHITELIST_TRAIN = A1_ROOT / "train_whitelist.txt"
WHITELIST_VAL = A1_ROOT / "val_whitelist.txt"
DEFAULT_CONDA_PREFIX = (
    "source /home/SNN/anaconda3/etc/profile.d/conda.sh && "
    "conda activate /mnt/sdb/ym/envs/OmniTrack"
)

VALID_STAGES = ("s1", "s2")
VALID_BUDGETS = (15000, 30000, 60000)
VALID_VARIANTS = ("baseline", "a")
VALID_SEEDS = (0, 1, 2)
VALID_JOBS = ("train", "proxy_val", "card2_orig", "card2_roll", "all")
VALID_BATCH_SIZES = (2, 4)

DEFAULT_BATCH_SIZE = 4
DEFAULT_IMAGE_WIDTH = 3760.0
DEFAULT_SEAM_BAND_PX = 400.0
CARD2_SEQUENCE = "clark-center-2019-02-28_1"
CARD2_ROLL_PX = 940
TRACKING_MODE = "tbd"
TBD_BACKEND = "hybridsort"
SUMMARY_METRICS = ("HOTA", "IDF1", "FP", "IDSW", "Frag")
GAP_METRICS = ("IDSW", "Frag", "FP_seam")
MAX_NON_SEAM_RELATIVE_INCREASE = 0.02
MAX_NON_SEAM_ABSOLUTE_DROP = 0.2
MIN_ROLL_GAP_SHRINK = 0.20
ZERO_GAP_TOLERANCE = 1
PREFERRED_GPU_ORDER = (1, 2, 3)
SEAM_SUBSET = "seam"
NON_SEAM_SUBSET = "non_seam"
FULL_SUBSET = "full"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def quote(value: Any) -> str:
    return shlex.quote(str(value))


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


def stable_port(*parts: Any) -> int:
    seed = 173
    for part in parts:
        for ch in str(part):
            seed = (seed * 33 + ord(ch)) % 10000
    return 28000 + (seed % 1000)


@dataclass(frozen=True)
class EvalSpec:
    name: str
    sequence_names: tuple[str, ...]
    eval_roll_px: int
    inverse_roll_px: int

    @property
    def split_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class RunSpec:
    stage: str
    budget: int
    seed: int
    variant: str
    job: str
    gpu_request: str
    batch_size: int
    effective_iters: int

    @property
    def is_preflight(self) -> bool:
        return self.effective_iters != self.budget

    @property
    def run_slug(self) -> str:
        if self.is_preflight:
            return (
                f"a2_{self.stage}_b{self.budget}_i{self.effective_iters}_"
                f"seed{self.seed}_{self.variant}"
            )
        return f"a2_{self.stage}_{self.budget}_seed{self.seed}_{self.variant}"

    @property
    def pair_key(self) -> str:
        return f"{self.stage}_{self.budget}_seed{self.seed}"

    @property
    def tmux_session(self) -> str:
        return f"{self.run_slug}_{self.job}"

    @property
    def base_config(self) -> Path:
        return A_CONFIG if self.variant == "a" else BASELINE_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified runner for A2 two-stage training, eval, export, and summary."
    )
    parser.add_argument("--stage", choices=VALID_STAGES, required=True)
    parser.add_argument("--budget", type=int, choices=VALID_BUDGETS, required=True)
    parser.add_argument("--variant", choices=VALID_VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=VALID_SEEDS, required=True)
    parser.add_argument("--job", choices=VALID_JOBS, required=True)
    parser.add_argument("--gpu", default="auto", help="GPU id or 'auto'.")
    parser.add_argument("--launch", action="store_true", help="Launch the selected job in tmux.")
    parser.add_argument("--force", action="store_true", help="Replace an existing tmux session with the same name.")
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=VALID_BATCH_SIZES,
        default=DEFAULT_BATCH_SIZE,
        help="Single-GPU batch size used for train configs.",
    )
    parser.add_argument(
        "--override-max-iters",
        type=int,
        default=None,
        help="Optional effective max_iters override, e.g. 100 for the preflight pair.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare configs/commands and print them without launching tmux.",
    )
    parser.add_argument(
        "--internal-execute",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def validate_paths() -> None:
    missing = [
        path
        for path in (
            BASELINE_CONFIG,
            A_CONFIG,
            BASELINE_CHECKPOINT,
            WHITELIST_TRAIN,
            WHITELIST_VAL,
        )
        if not path.exists()
    ]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"A2 prerequisites are missing:\n{joined}")


def load_manifest() -> dict[str, Any]:
    return load_json(
        MANIFEST_PATH,
        {
            "updated_at": None,
            "runs": {},
            "pairs": {},
            "budgets": {},
        },
    )


def save_manifest(manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = now_iso()
    dump_json(MANIFEST_PATH, manifest)


@contextmanager
def manifest_transaction() -> dict[str, Any]:
    ensure_dir(A2_ROOT)
    with LOCK_PATH.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            manifest = load_manifest()
            yield manifest
            refresh_views(manifest)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def build_eval_specs(val_sequences: list[str]) -> dict[str, EvalSpec]:
    return {
        "proxy_val": EvalSpec(
            name="proxy_val",
            sequence_names=tuple(val_sequences),
            eval_roll_px=0,
            inverse_roll_px=0,
        ),
        "card2_orig": EvalSpec(
            name="card2_orig",
            sequence_names=(CARD2_SEQUENCE,),
            eval_roll_px=0,
            inverse_roll_px=0,
        ),
        "card2_roll": EvalSpec(
            name="card2_roll",
            sequence_names=(CARD2_SEQUENCE,),
            eval_roll_px=CARD2_ROLL_PX,
            inverse_roll_px=CARD2_ROLL_PX,
        ),
    }


def choose_gpu(gpu_request: str) -> str:
    if gpu_request != "auto":
        return str(int(gpu_request))

    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    candidates = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        idx_text, name, util_text, mem_used_text, mem_total_text = [
            part.strip() for part in line.split(",")
        ]
        candidates.append(
            {
                "index": int(idx_text),
                "name": name,
                "util": int(util_text),
                "mem_used": int(mem_used_text),
                "mem_total": int(mem_total_text),
            }
        )
    preferred = [item for item in candidates if item["index"] in PREFERRED_GPU_ORDER]
    pool = preferred or candidates
    preferred_rank = {gpu_id: rank for rank, gpu_id in enumerate(PREFERRED_GPU_ORDER)}
    selected = sorted(
        pool,
        key=lambda item: (
            item["util"],
            item["mem_used"],
            preferred_rank.get(item["index"], 999),
            item["index"],
        ),
    )[0]
    return str(selected["index"])


def shell_prefix() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"{DEFAULT_CONDA_PREFIX}\n"
        f"cd {quote(REPO_ROOT)}\n"
    )


def shell_with_logging(log_path: Path, commands: list[str]) -> str:
    lines = [
        shell_prefix(),
        f"exec > >(tee -a {quote(log_path)}) 2>&1",
        f"echo \"[A2] started at $(date -Iseconds)\"",
        *commands,
        "echo \"[A2] finished at $(date -Iseconds)\"",
        "",
    ]
    return "\n".join(lines)


def python_literal(value: Any) -> str:
    return repr(value)


def large_eval_interval(effective_iters: int) -> int:
    return effective_iters * 100000000


def train_config_text(spec: RunSpec, train_sequences: list[str], val_sequences: list[str]) -> str:
    return (
        f"_base_ = [{python_literal(str(spec.base_config))}]\n\n"
        f"load_from = {python_literal(relative_to_repo(BASELINE_CHECKPOINT))}\n"
        "resume_from = None\n"
        "find_unused_parameters = True\n"
        "num_gpus = 1\n"
        f"total_batch_size = {spec.batch_size}\n"
        f"batch_size = {spec.batch_size}\n"
        "model = dict(\n"
        "    img_backbone=dict(\n"
        "        frozen_stages=4,\n"
        "        norm_eval=True,\n"
        "    ),\n"
        ")\n"
        "optimizer = dict(\n"
        "    paramwise_cfg=dict(\n"
        "        custom_keys=dict(\n"
        "            img_backbone=dict(lr_mult=0.0),\n"
        "        ),\n"
        "    ),\n"
        ")\n"
        f"runner = dict(type='IterBasedRunner', max_iters={spec.effective_iters})\n"
        f"checkpoint_config = dict(interval={spec.effective_iters})\n"
        f"evaluation = dict(interval={large_eval_interval(spec.effective_iters)})\n"
        "data = dict(\n"
        f"    samples_per_gpu={spec.batch_size},\n"
        "    train=dict(\n"
        f"        sequence_whitelist={python_literal(train_sequences)},\n"
        "    ),\n"
        "    val=dict(\n"
        f"        sequence_whitelist={python_literal(val_sequences)},\n"
        "    ),\n"
        "    test=dict(\n"
        f"        sequence_whitelist={python_literal(val_sequences)},\n"
        "    ),\n"
        ")\n"
    )


def eval_config_text(spec: RunSpec, eval_spec: EvalSpec) -> str:
    return (
        f"_base_ = [{python_literal(str(spec.base_config))}]\n\n"
        "model = dict(\n"
        "    head=dict(\n"
        "        instance_bank=dict(\n"
        f"            tracking_mode={python_literal(TRACKING_MODE)},\n"
        f"            tbd_backend={python_literal(TBD_BACKEND)},\n"
        "        ),\n"
        "    ),\n"
        ")\n"
        "data = dict(\n"
        "    test=dict(\n"
        f"        sequence_whitelist={python_literal(list(eval_spec.sequence_names))},\n"
        "        data_aug_conf=dict(\n"
        f"            eval_roll_px={eval_spec.eval_roll_px},\n"
        "        ),\n"
        "    ),\n"
        ")\n"
    )


def checkpoint_path(work_dir: Path, effective_iters: int) -> Path:
    return work_dir / f"iter_{effective_iters}.pth"


def run_dirs(spec: RunSpec) -> dict[str, Path]:
    meta_root = ensure_dir(RUN_META_ROOT / spec.run_slug)
    work_dir = REPO_ROOT / "work_dirs" / spec.run_slug
    ensure_dir(work_dir)
    ensure_dir(meta_root)
    ensure_dir(COMMAND_ROOT / spec.run_slug)
    ensure_dir(CONFIG_ROOT / spec.run_slug)
    ensure_dir(meta_root / "eval")
    return {
        "meta_root": meta_root,
        "work_dir": work_dir,
        "command_root": COMMAND_ROOT / spec.run_slug,
        "config_root": CONFIG_ROOT / spec.run_slug,
    }


def eval_dirs(spec: RunSpec, eval_name: str) -> dict[str, Path]:
    base = ensure_dir(RUN_META_ROOT / spec.run_slug / "eval" / eval_name)
    return {
        "base": base,
        "raw_json": ensure_dir(base / "raw_json"),
        "trackeval_gt": ensure_dir(base / "trackeval/gt"),
        "trackeval_trackers": ensure_dir(base / "trackeval/trackers"),
        "diag_out": ensure_dir(base / "diag_seam_split"),
    }


def tracker_name(spec: RunSpec, eval_name: str) -> str:
    return f"{spec.run_slug}_{eval_name}"


def split_name(spec: RunSpec, eval_name: str) -> str:
    return f"{spec.run_slug}_{eval_name}"


def build_train_command(spec: RunSpec, gpu_id: str, cfg_path: Path, work_dir: Path) -> str:
    port = stable_port(spec.run_slug, "train")
    return (
        f"CUDA_VISIBLE_DEVICES={quote(gpu_id)} "
        f"PORT={port} "
        f"bash tools/dist_train.sh {quote(cfg_path)} 1 "
        f"--work-dir={quote(work_dir)} "
        f"--seed {spec.seed} --deterministic"
    )


def build_test_command(
    spec: RunSpec,
    eval_name: str,
    gpu_id: str,
    cfg_path: Path,
    ckpt_path: Path,
    results_pkl: Path,
    raw_json_dir: Path,
) -> str:
    port = stable_port(spec.run_slug, eval_name, "test")
    return (
        f"CUDA_VISIBLE_DEVICES={quote(gpu_id)} "
        f"PORT={port} "
        f"bash tools/dist_test.sh {quote(cfg_path)} {quote(ckpt_path)} 1 "
        f"--out {quote(results_pkl)} "
        "--format-only "
        f"--eval-options jsonfile_prefix={quote(raw_json_dir)} "
        f"--seed {spec.seed} --deterministic"
    )


def build_export_command(
    spec: RunSpec,
    eval_name: str,
    eval_spec: EvalSpec,
    pred_json: Path,
    gt_out_dir: Path,
    trackers_out_dir: Path,
) -> str:
    seq_args = " ".join(quote(seq) for seq in eval_spec.sequence_names)
    cmd = (
        "python tools/export_jrdb_trackeval_2d.py "
        f"--ann-file {quote(REPO_ROOT / 'data/JRDB2019_2d_stitched_anno_pkls/JRDB_infos_val_v1.2.pkl')} "
        f"--pred-json {quote(pred_json)} "
        f"--split-name {quote(split_name(spec, eval_name))} "
        f"--tracker-name {quote(tracker_name(spec, eval_name))} "
        f"--gt-out-dir {quote(gt_out_dir)} "
        f"--trackers-out-dir {quote(trackers_out_dir)} "
        f"--sequence-names {seq_args}"
    )
    if eval_spec.inverse_roll_px:
        cmd += f" --inverse-roll-px {eval_spec.inverse_roll_px}"
    return cmd


def build_trackeval_command(spec: RunSpec, eval_name: str, gt_dir: Path, trackers_dir: Path) -> str:
    split = split_name(spec, eval_name)
    tracker = tracker_name(spec, eval_name)
    trackeval_root = REPO_ROOT / "jrdb_toolkit/tracking_eval/TrackEval"
    return (
        f"cd {quote(trackeval_root)} && "
        "PYTHONPATH=. python scripts/run_jrdb.py "
        "--USE_PARALLEL False "
        f"--GT_FOLDER {quote(gt_dir)} "
        f"--TRACKERS_FOLDER {quote(trackers_dir)} "
        f"--SPLIT_TO_EVAL {quote(split)} "
        f"--TRACKERS_TO_EVAL {quote(tracker)} "
        "--METRICS HOTA CLEAR Identity OSPA "
        "--PRINT_ONLY_COMBINED True --PLOT_CURVES False"
    )


def build_diag_command(spec: RunSpec, eval_name: str, gt_dir: Path, trackers_dir: Path, diag_out: Path) -> str:
    return (
        f"python {quote(REPO_ROOT / 'tools/diag_seam_metric_split.py')} "
        f"--tracker-names {quote(tracker_name(spec, eval_name))} "
        f"--gt-folder {quote(gt_dir)} "
        f"--trackers-folder {quote(trackers_dir)} "
        f"--split-name {quote(split_name(spec, eval_name))} "
        f"--image-width {DEFAULT_IMAGE_WIDTH} "
        f"--seam-band-px {DEFAULT_SEAM_BAND_PX} "
        f"--out-dir {quote(diag_out)}"
    )


def build_handoff_text(spec: RunSpec, gpu_id: str, step_log: Path) -> str:
    return (
        f"session: {spec.tmux_session}\n"
        f"gpu: {gpu_id}\n"
        f"attach: tmux attach -t {spec.tmux_session}\n"
        f"capture: tmux capture-pane -pt {spec.tmux_session} | tail -n 80\n"
        f"tail: tail -f {step_log}\n"
    )


def prepare_run_assets(spec: RunSpec, gpu_request: str) -> dict[str, Any]:
    train_sequences = read_lines(WHITELIST_TRAIN)
    val_sequences = read_lines(WHITELIST_VAL)
    eval_specs = build_eval_specs(val_sequences)
    selected_gpu = choose_gpu(gpu_request)
    dirs = run_dirs(spec)

    train_cfg = dirs["config_root"] / "train.py"
    write_text(train_cfg, train_config_text(spec, train_sequences, val_sequences))

    eval_cfg_paths = {}
    step_scripts = {}
    step_logs = {}
    step_commands = {}

    train_log = dirs["meta_root"] / "train.stdout.log"
    train_script = dirs["command_root"] / "train.sh"
    train_cmd = build_train_command(spec, selected_gpu, train_cfg, dirs["work_dir"])
    write_text(train_script, shell_with_logging(train_log, [train_cmd]))
    train_script.chmod(0o755)
    step_scripts["train"] = train_script
    step_logs["train"] = train_log
    step_commands["train"] = train_cmd

    for eval_name, eval_spec in eval_specs.items():
        cfg_path = dirs["config_root"] / f"{eval_name}.py"
        eval_cfg_paths[eval_name] = cfg_path
        write_text(cfg_path, eval_config_text(spec, eval_spec))
        eval_output_dirs = eval_dirs(spec, eval_name)
        log_path = dirs["meta_root"] / f"{eval_name}.stdout.log"
        script_path = dirs["command_root"] / f"{eval_name}.sh"
        ckpt_path = checkpoint_path(dirs["work_dir"], spec.effective_iters)
        results_pkl = eval_output_dirs["base"] / "results_val.pkl"
        pred_json = eval_output_dirs["raw_json"] / "results_jrdb2d.json"
        commands = [
            build_test_command(
                spec=spec,
                eval_name=eval_name,
                gpu_id=selected_gpu,
                cfg_path=cfg_path,
                ckpt_path=ckpt_path,
                results_pkl=results_pkl,
                raw_json_dir=eval_output_dirs["raw_json"],
            ),
            build_export_command(
                spec=spec,
                eval_name=eval_name,
                eval_spec=eval_spec,
                pred_json=pred_json,
                gt_out_dir=eval_output_dirs["trackeval_gt"],
                trackers_out_dir=eval_output_dirs["trackeval_trackers"],
            ),
            build_trackeval_command(
                spec=spec,
                eval_name=eval_name,
                gt_dir=eval_output_dirs["trackeval_gt"],
                trackers_dir=eval_output_dirs["trackeval_trackers"],
            ),
            build_diag_command(
                spec=spec,
                eval_name=eval_name,
                gt_dir=eval_output_dirs["trackeval_gt"],
                trackers_dir=eval_output_dirs["trackeval_trackers"],
                diag_out=eval_output_dirs["diag_out"],
            ),
        ]
        write_text(script_path, shell_with_logging(log_path, commands))
        script_path.chmod(0o755)
        step_scripts[eval_name] = script_path
        step_logs[eval_name] = log_path
        step_commands[eval_name] = commands

    handoff_text = build_handoff_text(spec, selected_gpu, train_log)
    write_text(dirs["command_root"] / "handoff.txt", handoff_text)

    return {
        "selected_gpu": selected_gpu,
        "dirs": {
            key: str(value) for key, value in dirs.items()
        },
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "train_config": str(train_cfg),
        "eval_configs": {key: str(value) for key, value in eval_cfg_paths.items()},
        "step_scripts": {key: str(value) for key, value in step_scripts.items()},
        "step_logs": {key: str(value) for key, value in step_logs.items()},
        "step_commands": step_commands,
        "checkpoint_path": str(checkpoint_path(dirs["work_dir"], spec.effective_iters)),
        "work_dir": str(dirs["work_dir"]),
        "meta_root": str(dirs["meta_root"]),
        "handoff_path": str(dirs["command_root"] / "handoff.txt"),
    }


def ensure_batch_size_consistency(manifest: dict[str, Any], spec: RunSpec) -> None:
    if spec.is_preflight:
        return
    for record in manifest["runs"].values():
        if int(record.get("effective_iters", 0)) not in VALID_BUDGETS:
            continue
        if record.get("selected_batch_size") is None:
            continue
        if int(record["selected_batch_size"]) != spec.batch_size:
            raise ValueError(
                "A2 batch size must stay consistent across non-preflight runs. "
                f"Existing run {record['run_slug']} uses batch_size={record['selected_batch_size']}, "
                f"but current request uses batch_size={spec.batch_size}."
            )


def update_run_record(
    manifest: dict[str, Any],
    spec: RunSpec,
    assets: dict[str, Any],
) -> dict[str, Any]:
    runs = manifest["runs"]
    run_key = spec.run_slug
    record = deepcopy(runs.get(run_key, {}))
    record.update(
        {
            "run_slug": run_key,
            "pair_key": spec.pair_key,
            "stage": spec.stage,
            "budget": spec.budget,
            "effective_iters": spec.effective_iters,
            "seed": spec.seed,
            "variant": spec.variant,
            "job": spec.job,
            "gpu_request": spec.gpu_request,
            "selected_gpu": assets["selected_gpu"],
            "selected_batch_size": spec.batch_size,
            "tmux_session": spec.tmux_session,
            "work_dir": assets["work_dir"],
            "checkpoint_path": assets["checkpoint_path"],
            "meta_root": assets["meta_root"],
            "train_config": assets["train_config"],
            "eval_configs": assets["eval_configs"],
            "step_scripts": assets["step_scripts"],
            "step_logs": assets["step_logs"],
            "handoff_path": assets["handoff_path"],
            "train_sequences": assets["train_sequences"],
            "val_sequences": assets["val_sequences"],
            "base_config": relative_to_repo(spec.base_config),
            "base_checkpoint": relative_to_repo(BASELINE_CHECKPOINT),
            "prepared_at": now_iso(),
            "steps": record.get("steps", {}),
            "evaluations": record.get("evaluations", {}),
            "bn_freeze_check": record.get("bn_freeze_check"),
            "single_seed_gate": record.get("single_seed_gate"),
            "budget_gate": record.get("budget_gate"),
        }
    )
    for step_name in ("train", "proxy_val", "card2_orig", "card2_roll"):
        record["steps"].setdefault(
            step_name,
            {
                "status": "prepared",
                "log_path": assets["step_logs"].get(step_name),
                "script_path": assets["step_scripts"].get(step_name),
                "updated_at": now_iso(),
            },
        )
    runs[run_key] = record
    return record


def tmux_session_exists(session_name: str) -> bool:
    proc = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def launch_tmux(spec: RunSpec, selected_gpu: str, force: bool) -> None:
    if tmux_session_exists(spec.tmux_session):
        if not force:
            raise RuntimeError(
                f"tmux session {spec.tmux_session!r} already exists. Use --force to replace it."
            )
        subprocess.run(["tmux", "kill-session", "-t", spec.tmux_session], check=True)

    cmd_parts = [
        "python",
        str(Path(__file__).resolve()),
        "--stage",
        spec.stage,
        "--budget",
        str(spec.budget),
        "--variant",
        spec.variant,
        "--seed",
        str(spec.seed),
        "--job",
        spec.job,
        "--gpu",
        selected_gpu,
        "--batch-size",
        str(spec.batch_size),
        "--internal-execute",
    ]
    if spec.is_preflight:
        cmd_parts.extend(["--override-max-iters", str(spec.effective_iters)])
    cmd = " ".join(quote(part) for part in cmd_parts)
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", spec.tmux_session, f"bash -lc {quote(cmd)}"],
        check=True,
    )


def run_bash_script(script_path: Path) -> None:
    subprocess.run(["bash", str(script_path)], cwd=REPO_ROOT, check=True)


def load_combined_metrics_csv(path: Path) -> dict[str, dict[str, float | int]]:
    if not path.exists():
        raise FileNotFoundError(f"combined_metrics.csv not found: {path}")
    metrics: dict[str, dict[str, float | int]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subset = row["subset"]
            metrics[subset] = {
                "HOTA": float(row["HOTA"]),
                "IDF1": float(row["IDF1"]),
                "FP": int(float(row["FP"])),
                "IDSW": int(float(row["IDSW"])),
                "Frag": int(float(row["Frag"])),
            }
    return metrics


def evaluation_result_paths(spec: RunSpec, eval_name: str) -> dict[str, Path]:
    eval_root = RUN_META_ROOT / spec.run_slug / "eval" / eval_name
    return {
        "root": eval_root,
        "results_pkl": eval_root / "results_val.pkl",
        "pred_json": eval_root / "raw_json" / "results_jrdb2d.json",
        "diag_csv": eval_root / "diag_seam_split" / tracker_name(spec, eval_name) / "combined_metrics.csv",
    }


def record_step_status(
    manifest: dict[str, Any],
    spec: RunSpec,
    step_name: str,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    record = manifest["runs"][spec.run_slug]
    step = record["steps"].setdefault(step_name, {})
    step["status"] = status
    step["updated_at"] = now_iso()
    if extra:
        step.update(extra)


def compare_backbone_bn(reference_ckpt: Path, target_ckpt: Path) -> dict[str, Any]:
    ref = torch.load(reference_ckpt, map_location="cpu")
    tgt = torch.load(target_ckpt, map_location="cpu")
    ref_state = ref.get("state_dict", ref)
    tgt_state = tgt.get("state_dict", tgt)

    def extract(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        out = {}
        for key, value in state_dict.items():
            if not key.startswith("img_backbone."):
                continue
            if not (key.endswith("running_mean") or key.endswith("running_var")):
                continue
            out[key] = value.detach().cpu()
        return out

    ref_buffers = extract(ref_state)
    tgt_buffers = extract(tgt_state)
    changed = []
    for key in sorted(set(ref_buffers) | set(tgt_buffers)):
        if key not in ref_buffers or key not in tgt_buffers:
            changed.append({"key": key, "reason": "missing"})
            continue
        if not torch.equal(ref_buffers[key], tgt_buffers[key]):
            changed.append(
                {
                    "key": key,
                    "max_abs_diff": float((ref_buffers[key] - tgt_buffers[key]).abs().max().item()),
                }
            )
    return {
        "reference_checkpoint": str(reference_ckpt),
        "target_checkpoint": str(target_ckpt),
        "checked_key_count": len(ref_buffers),
        "changed_key_count": len(changed),
        "changed_keys_preview": changed[:20],
        "pass": len(changed) == 0,
    }


def relative_increase(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return math.inf if candidate > 0 else 0.0
    return (candidate - baseline) / baseline


def card2_gap(metrics_orig: dict[str, dict[str, float | int]], metrics_roll: dict[str, dict[str, float | int]]) -> dict[str, int]:
    return {
        "IDSW": abs(int(metrics_orig[FULL_SUBSET]["IDSW"]) - int(metrics_roll[FULL_SUBSET]["IDSW"])),
        "Frag": abs(int(metrics_orig[FULL_SUBSET]["Frag"]) - int(metrics_roll[FULL_SUBSET]["Frag"])),
        "FP_seam": abs(int(metrics_orig[SEAM_SUBSET]["FP"]) - int(metrics_roll[SEAM_SUBSET]["FP"])),
    }


def build_single_seed_gate(baseline_record: dict[str, Any], a_record: dict[str, Any]) -> dict[str, Any] | None:
    for eval_name in ("proxy_val", "card2_orig", "card2_roll"):
        if eval_name not in baseline_record.get("evaluations", {}):
            return None
        if eval_name not in a_record.get("evaluations", {}):
            return None

    baseline_proxy = baseline_record["evaluations"]["proxy_val"]["metrics"]
    a_proxy = a_record["evaluations"]["proxy_val"]["metrics"]
    baseline_gap = baseline_record["evaluations"]["card2_gap"]
    a_gap = a_record["evaluations"]["card2_gap"]

    reasons = []
    passed = True
    gap_check = {}
    for metric_name in GAP_METRICS:
        baseline_value = int(baseline_gap[metric_name])
        a_value = int(a_gap[metric_name])
        if baseline_value > 0:
            threshold = baseline_value * (1.0 - MIN_ROLL_GAP_SHRINK)
            metric_pass = a_value <= threshold
            shrink_ratio = (baseline_value - a_value) / baseline_value
        else:
            metric_pass = a_value <= ZERO_GAP_TOLERANCE
            shrink_ratio = None
        gap_check[metric_name] = {
            "baseline_gap": baseline_value,
            "a_gap": a_value,
            "pass": metric_pass,
            "shrink_ratio": shrink_ratio,
        }
        if not metric_pass:
            passed = False
            if baseline_value > 0:
                reasons.append(
                    f"card2 {metric_name} gap shrank insufficiently: baseline={baseline_value}, A={a_value}"
                )
            else:
                reasons.append(
                    f"card2 {metric_name} gap exceeded zero-gap tolerance: baseline=0, A={a_value}"
                )

    seam_baseline = baseline_proxy[SEAM_SUBSET]
    seam_a = a_proxy[SEAM_SUBSET]
    seam_improvements = []
    seam_worsened = []
    for metric_name in ("FP", "IDSW", "Frag"):
        if int(seam_a[metric_name]) < int(seam_baseline[metric_name]):
            seam_improvements.append(metric_name)
        elif int(seam_a[metric_name]) > int(seam_baseline[metric_name]):
            seam_worsened.append(metric_name)

    seam_hota_pass = float(seam_a["HOTA"]) >= float(seam_baseline["HOTA"])
    seam_idf1_pass = float(seam_a["IDF1"]) >= float(seam_baseline["IDF1"])
    seam_event_pass = bool(seam_improvements) and len(seam_worsened) < 3
    if not seam_hota_pass:
        passed = False
        reasons.append(
            f"proxy-val seam HOTA dropped: baseline={seam_baseline['HOTA']:.3f}, A={seam_a['HOTA']:.3f}"
        )
    if not seam_idf1_pass:
        passed = False
        reasons.append(
            f"proxy-val seam IDF1 dropped: baseline={seam_baseline['IDF1']:.3f}, A={seam_a['IDF1']:.3f}"
        )
    if not seam_event_pass:
        passed = False
        reasons.append(
            f"proxy-val seam event metrics do not show a clean win: improved={seam_improvements}, worsened={seam_worsened}"
        )

    non_seam_baseline = baseline_proxy[NON_SEAM_SUBSET]
    non_seam_a = a_proxy[NON_SEAM_SUBSET]
    non_seam_checks = {}
    for metric_name in ("HOTA", "IDF1"):
        delta = float(non_seam_a[metric_name]) - float(non_seam_baseline[metric_name])
        metric_pass = delta >= -MAX_NON_SEAM_ABSOLUTE_DROP
        non_seam_checks[metric_name] = {"delta": delta, "pass": metric_pass}
        if not metric_pass:
            passed = False
            reasons.append(
                f"proxy-val non-seam {metric_name} dropped too much: delta={delta:.3f}"
            )
    for metric_name in ("FP", "IDSW", "Frag"):
        rel = relative_increase(
            float(non_seam_a[metric_name]),
            float(non_seam_baseline[metric_name]),
        )
        metric_pass = rel <= MAX_NON_SEAM_RELATIVE_INCREASE
        non_seam_checks[metric_name] = {"relative_increase": rel, "pass": metric_pass}
        if not metric_pass:
            if math.isinf(rel):
                rel_text = "inf"
            else:
                rel_text = f"{rel:.4f}"
            passed = False
            reasons.append(
                f"proxy-val non-seam {metric_name} increased too much: relative_increase={rel_text}"
            )

    if not reasons:
        reasons.append("single-seed gate passed")

    return {
        "pass": passed,
        "reasons": reasons,
        "card2_gap": gap_check,
        "proxy_val": {
            "seam": {
                "baseline": seam_baseline,
                "a": seam_a,
                "improved": seam_improvements,
                "worsened": seam_worsened,
            },
            "non_seam": non_seam_checks,
        },
    }


def build_budget_gate(pair_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(pair_records) != len(VALID_SEEDS):
        return None
    gates = [record.get("single_seed_gate") for record in pair_records]
    if any(gate is None for gate in gates):
        return None

    pass_count = sum(bool(gate["pass"]) for gate in gates)
    mean_gap = {}
    mean_positive = True
    reasons = []
    for metric_name in GAP_METRICS:
        baseline_mean = sum(record["baseline_gap"][metric_name] for record in pair_records) / len(pair_records)
        a_mean = sum(record["a_gap"][metric_name] for record in pair_records) / len(pair_records)
        if baseline_mean > 0:
            metric_pass = a_mean < baseline_mean
        else:
            metric_pass = a_mean <= ZERO_GAP_TOLERANCE
        mean_gap[metric_name] = {
            "baseline_mean_gap": baseline_mean,
            "a_mean_gap": a_mean,
            "pass": metric_pass,
        }
        if not metric_pass:
            mean_positive = False
            reasons.append(
                f"mean card2 {metric_name} gap is not positive enough: baseline_mean={baseline_mean:.3f}, A_mean={a_mean:.3f}"
            )

    seam_hota_delta = sum(record["proxy_val_seam_delta"]["HOTA"] for record in pair_records) / len(pair_records)
    seam_idf1_delta = sum(record["proxy_val_seam_delta"]["IDF1"] for record in pair_records) / len(pair_records)
    if seam_hota_delta < 0:
        mean_positive = False
        reasons.append(f"mean seam HOTA delta is negative: {seam_hota_delta:.3f}")
    if seam_idf1_delta < 0:
        mean_positive = False
        reasons.append(f"mean seam IDF1 delta is negative: {seam_idf1_delta:.3f}")

    non_seam_metric_deltas = {
        "HOTA": sum(record["proxy_val_non_seam_delta"]["HOTA"] for record in pair_records) / len(pair_records),
        "IDF1": sum(record["proxy_val_non_seam_delta"]["IDF1"] for record in pair_records) / len(pair_records),
        "FP": sum(record["proxy_val_non_seam_relative"]["FP"] for record in pair_records) / len(pair_records),
        "IDSW": sum(record["proxy_val_non_seam_relative"]["IDSW"] for record in pair_records) / len(pair_records),
        "Frag": sum(record["proxy_val_non_seam_relative"]["Frag"] for record in pair_records) / len(pair_records),
    }
    if non_seam_metric_deltas["HOTA"] < -MAX_NON_SEAM_ABSOLUTE_DROP:
        mean_positive = False
        reasons.append(f"mean non-seam HOTA delta is too negative: {non_seam_metric_deltas['HOTA']:.3f}")
    if non_seam_metric_deltas["IDF1"] < -MAX_NON_SEAM_ABSOLUTE_DROP:
        mean_positive = False
        reasons.append(f"mean non-seam IDF1 delta is too negative: {non_seam_metric_deltas['IDF1']:.3f}")
    for metric_name in ("FP", "IDSW", "Frag"):
        rel = non_seam_metric_deltas[metric_name]
        if rel > MAX_NON_SEAM_RELATIVE_INCREASE:
            mean_positive = False
            reasons.append(
                f"mean non-seam {metric_name} relative increase is too high: {rel:.4f}"
            )

    passed = pass_count >= 2 and mean_positive
    if not reasons:
        reasons.append("budget gate passed")
    return {
        "pass": passed,
        "pass_count": pass_count,
        "required_pass_count": 2,
        "mean_gap": mean_gap,
        "mean_seam_delta": {
            "HOTA": seam_hota_delta,
            "IDF1": seam_idf1_delta,
        },
        "mean_non_seam": non_seam_metric_deltas,
        "reasons": reasons,
    }


def refresh_pair_and_budget_views(manifest: dict[str, Any]) -> None:
    pairs: dict[str, Any] = {}
    runs = manifest["runs"]
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for run in runs.values():
        grouped.setdefault(run["pair_key"], {}).setdefault(run["variant"], []).append(run)

    def select_pair_run(candidates: list[dict[str, Any]] | None) -> dict[str, Any] | None:
        if not candidates:
            return None

        def sort_key(run: dict[str, Any]) -> tuple[int, int, str]:
            budget = int(run.get("budget", 0))
            effective_iters = int(run.get("effective_iters", 0))
            is_full_budget = int(effective_iters == budget)
            updated_at = str(run.get("prepared_at") or run.get("launched_at") or "")
            return (is_full_budget, effective_iters, updated_at)

        return max(candidates, key=sort_key)

    for pair_key, items in grouped.items():
        baseline_record = select_pair_run(items.get("baseline"))
        a_record = select_pair_run(items.get("a"))
        stage, budget_text, seed_text = pair_key.split("_")
        budget = int(budget_text)
        seed = int(seed_text.replace("seed", ""))
        pair_record: dict[str, Any] = {
            "pair_key": pair_key,
            "stage": stage,
            "budget": budget,
            "seed": seed,
            "baseline_run_slug": baseline_record["run_slug"] if baseline_record else None,
            "a_run_slug": a_record["run_slug"] if a_record else None,
        }
        if baseline_record and a_record:
            gate = build_single_seed_gate(baseline_record, a_record)
            pair_record["single_seed_gate"] = gate
            if gate is not None:
                baseline_gap = baseline_record["evaluations"]["card2_gap"]
                a_gap = a_record["evaluations"]["card2_gap"]
                pair_record["baseline_gap"] = baseline_gap
                pair_record["a_gap"] = a_gap
                pair_record["proxy_val_seam_delta"] = {
                    "HOTA": float(a_record["evaluations"]["proxy_val"]["metrics"][SEAM_SUBSET]["HOTA"])
                    - float(baseline_record["evaluations"]["proxy_val"]["metrics"][SEAM_SUBSET]["HOTA"]),
                    "IDF1": float(a_record["evaluations"]["proxy_val"]["metrics"][SEAM_SUBSET]["IDF1"])
                    - float(baseline_record["evaluations"]["proxy_val"]["metrics"][SEAM_SUBSET]["IDF1"]),
                }
                pair_record["proxy_val_non_seam_delta"] = {
                    "HOTA": float(a_record["evaluations"]["proxy_val"]["metrics"][NON_SEAM_SUBSET]["HOTA"])
                    - float(baseline_record["evaluations"]["proxy_val"]["metrics"][NON_SEAM_SUBSET]["HOTA"]),
                    "IDF1": float(a_record["evaluations"]["proxy_val"]["metrics"][NON_SEAM_SUBSET]["IDF1"])
                    - float(baseline_record["evaluations"]["proxy_val"]["metrics"][NON_SEAM_SUBSET]["IDF1"]),
                }
                pair_record["proxy_val_non_seam_relative"] = {
                    metric_name: relative_increase(
                        float(a_record["evaluations"]["proxy_val"]["metrics"][NON_SEAM_SUBSET][metric_name]),
                        float(baseline_record["evaluations"]["proxy_val"]["metrics"][NON_SEAM_SUBSET][metric_name]),
                    )
                    for metric_name in ("FP", "IDSW", "Frag")
                }
                baseline_record["single_seed_gate"] = gate
                a_record["single_seed_gate"] = gate
        pairs[pair_key] = pair_record

    manifest["pairs"] = pairs

    budgets: dict[str, Any] = {}
    by_budget: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for pair in pairs.values():
        by_budget.setdefault((pair["stage"], pair["budget"]), []).append(pair)

    for (stage, budget), pair_records in by_budget.items():
        pair_records = sorted(pair_records, key=lambda item: item["seed"])
        gate = build_budget_gate(pair_records)
        budget_key = f"{stage}_{budget}"
        budgets[budget_key] = {
            "stage": stage,
            "budget": budget,
            "budget_gate": gate,
            "pair_keys": [record["pair_key"] for record in pair_records],
        }
        if gate is not None:
            for record in pair_records:
                for variant in ("baseline", "a"):
                    run_slug = record.get(f"{variant}_run_slug")
                    if run_slug and run_slug in runs:
                        runs[run_slug]["budget_gate"] = gate
    manifest["budgets"] = budgets


def flatten_run_row(run: dict[str, Any]) -> dict[str, Any]:
    row = {
        "stage": run["stage"],
        "budget": run["budget"],
        "effective_iters": run["effective_iters"],
        "seed": run["seed"],
        "variant": run["variant"],
        "gpu": run.get("selected_gpu", ""),
        "batch_size": run.get("selected_batch_size", ""),
        "work_dir": run.get("work_dir", ""),
        "checkpoint_path": run.get("checkpoint_path", ""),
        "train_status": run.get("steps", {}).get("train", {}).get("status", ""),
        "proxy_val_status": run.get("steps", {}).get("proxy_val", {}).get("status", ""),
        "card2_orig_status": run.get("steps", {}).get("card2_orig", {}).get("status", ""),
        "card2_roll_status": run.get("steps", {}).get("card2_roll", {}).get("status", ""),
        "bn_freeze_pass": "",
        "single_seed_pass": "",
        "budget_pass": "",
    }
    bn_check = run.get("bn_freeze_check")
    if bn_check is not None:
        row["bn_freeze_pass"] = bn_check.get("pass")
    single_seed_gate = run.get("single_seed_gate")
    if single_seed_gate is not None:
        row["single_seed_pass"] = single_seed_gate.get("pass")
    budget_gate = run.get("budget_gate")
    if budget_gate is not None:
        row["budget_pass"] = budget_gate.get("pass")

    for eval_name in ("proxy_val", "card2_orig", "card2_roll"):
        eval_record = run.get("evaluations", {}).get(eval_name)
        if eval_record is None:
            continue
        metrics = eval_record["metrics"]
        for subset in (FULL_SUBSET, SEAM_SUBSET, NON_SEAM_SUBSET):
            subset_metrics = metrics.get(subset)
            if subset_metrics is None:
                continue
            for metric_name in SUMMARY_METRICS:
                row[f"{eval_name}_{subset}_{metric_name}"] = subset_metrics[metric_name]

    if "evaluations" in run and "card2_gap" in run["evaluations"]:
        for metric_name in GAP_METRICS:
            row[f"card2_gap_{metric_name}"] = run["evaluations"]["card2_gap"][metric_name]
    return row


def write_experiment_matrix(manifest: dict[str, Any]) -> None:
    rows = [flatten_run_row(run) for run in sorted(manifest["runs"].values(), key=lambda item: item["run_slug"])]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if not rows:
        rows = []
        fieldnames = [
            "stage",
            "budget",
            "effective_iters",
            "seed",
            "variant",
            "gpu",
            "batch_size",
            "work_dir",
            "checkpoint_path",
            "train_status",
            "proxy_val_status",
            "card2_orig_status",
            "card2_roll_status",
            "bn_freeze_pass",
            "single_seed_pass",
            "budget_pass",
        ]
    ensure_dir(MATRIX_PATH.parent)
    with MATRIX_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_stage_summary(manifest: dict[str, Any]) -> None:
    lines = [
        "# A2 Stage Gate Summary",
        "",
        f"Updated: {now_iso()}",
        "",
    ]
    for stage in VALID_STAGES:
        lines.append(f"## {stage}")
        lines.append("")
        stage_budget_keys = [key for key in manifest["budgets"] if key.startswith(f"{stage}_")]
        if not stage_budget_keys:
            lines.append("- No budget records yet.")
            lines.append("")
            continue
        for budget in sorted({int(key.split("_")[1]) for key in stage_budget_keys}):
            budget_key = f"{stage}_{budget}"
            budget_record = manifest["budgets"].get(budget_key, {})
            gate = budget_record.get("budget_gate")
            lines.append(f"### budget {budget}")
            if gate is None:
                lines.append("- Status: pending")
            else:
                lines.append(f"- Status: {'PASS' if gate['pass'] else 'FAIL'}")
                lines.append(
                    f"- Seed pass count: {gate['pass_count']} / {gate['required_pass_count']}"
                )
                lines.append(
                    f"- Mean seam delta: HOTA={gate['mean_seam_delta']['HOTA']:.3f}, "
                    f"IDF1={gate['mean_seam_delta']['IDF1']:.3f}"
                )
                lines.append(
                    f"- Mean non-seam delta: HOTA={gate['mean_non_seam']['HOTA']:.3f}, "
                    f"IDF1={gate['mean_non_seam']['IDF1']:.3f}, "
                    f"FP={gate['mean_non_seam']['FP']:.4f}, "
                    f"IDSW={gate['mean_non_seam']['IDSW']:.4f}, "
                    f"Frag={gate['mean_non_seam']['Frag']:.4f}"
                )
                for metric_name in GAP_METRICS:
                    metric_gate = gate["mean_gap"][metric_name]
                    lines.append(
                        f"- Mean card2 {metric_name} gap: baseline={metric_gate['baseline_mean_gap']:.3f}, "
                        f"A={metric_gate['a_mean_gap']:.3f}"
                    )
                lines.append("- Reasons:")
                for reason in gate["reasons"]:
                    lines.append(f"  - {reason}")
            pair_records = [
                manifest["pairs"][pair_key]
                for pair_key in budget_record.get("pair_keys", [])
                if pair_key in manifest["pairs"]
            ]
            for pair in sorted(pair_records, key=lambda item: item["seed"]):
                gate_text = "pending"
                if pair.get("single_seed_gate") is not None:
                    gate_text = "PASS" if pair["single_seed_gate"]["pass"] else "FAIL"
                lines.append(f"- seed {pair['seed']}: {gate_text}")
            lines.append("")

        if stage == "s1":
            b15 = manifest["budgets"].get("s1_15000", {}).get("budget_gate")
            b30 = manifest["budgets"].get("s1_30000", {}).get("budget_gate")
            if b15 and b30 and b15["pass"] and b30["pass"]:
                lines.append("- Stage advancement: s2/60000 is cleared to run.")
            else:
                lines.append("- Stage advancement: hold at s1 until both 15000 and 30000 pass.")
            lines.append("")
        if stage == "s2":
            b60 = manifest["budgets"].get("s2_60000", {}).get("budget_gate")
            if b60 and b60["pass"]:
                lines.append("- Full-run decision: A-tree is eligible for full-run discussion.")
            else:
                lines.append("- Full-run decision: not cleared yet.")
            lines.append("")

    write_text(SUMMARY_PATH, "\n".join(lines))


def refresh_views(manifest: dict[str, Any]) -> None:
    refresh_pair_and_budget_views(manifest)
    save_manifest(manifest)
    write_experiment_matrix(manifest)
    write_stage_summary(manifest)


def capture_evaluation(spec: RunSpec, eval_name: str) -> dict[str, Any]:
    paths = evaluation_result_paths(spec, eval_name)
    metrics = load_combined_metrics_csv(paths["diag_csv"])
    return {
        "results_pkl": str(paths["results_pkl"]),
        "pred_json": str(paths["pred_json"]),
        "combined_metrics_csv": str(paths["diag_csv"]),
        "metrics": metrics,
        "updated_at": now_iso(),
    }


def ensure_checkpoint_exists(spec: RunSpec) -> Path:
    ckpt = checkpoint_path(REPO_ROOT / "work_dirs" / spec.run_slug, spec.effective_iters)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {spec.run_slug}: {ckpt}. "
            "Run the training step first or confirm that max_iters/checkpoint interval match."
        )
    return ckpt


def execute_train(spec: RunSpec) -> None:
    with manifest_transaction() as manifest:
        record_step_status(manifest, spec, "train", "running")
        script_path = Path(manifest["runs"][spec.run_slug]["step_scripts"]["train"])
    try:
        run_bash_script(script_path)
    except subprocess.CalledProcessError:
        with manifest_transaction() as manifest:
            record_step_status(manifest, spec, "train", "failed")
        raise
    ckpt = ensure_checkpoint_exists(spec)
    with manifest_transaction() as manifest:
        record_step_status(
            manifest,
            spec,
            "train",
            "completed",
            {"checkpoint_path": str(ckpt)},
        )
        if spec.is_preflight and spec.effective_iters == 100:
            bn_report = compare_backbone_bn(BASELINE_CHECKPOINT, ckpt)
            manifest["runs"][spec.run_slug]["bn_freeze_check"] = bn_report
            bn_report_path = Path(manifest["runs"][spec.run_slug]["meta_root"]) / "bn_freeze_check.json"
            dump_json(bn_report_path, bn_report)


def execute_eval(spec: RunSpec, eval_name: str) -> None:
    ensure_checkpoint_exists(spec)
    with manifest_transaction() as manifest:
        record_step_status(manifest, spec, eval_name, "running")
        script_path = Path(manifest["runs"][spec.run_slug]["step_scripts"][eval_name])
    try:
        run_bash_script(script_path)
    except subprocess.CalledProcessError:
        with manifest_transaction() as manifest:
            record_step_status(manifest, spec, eval_name, "failed")
        raise
    evaluation = capture_evaluation(spec, eval_name)
    with manifest_transaction() as manifest:
        manifest["runs"][spec.run_slug]["evaluations"][eval_name] = evaluation
        if (
            "card2_orig" in manifest["runs"][spec.run_slug]["evaluations"]
            and "card2_roll" in manifest["runs"][spec.run_slug]["evaluations"]
        ):
            manifest["runs"][spec.run_slug]["evaluations"]["card2_gap"] = card2_gap(
                manifest["runs"][spec.run_slug]["evaluations"]["card2_orig"]["metrics"],
                manifest["runs"][spec.run_slug]["evaluations"]["card2_roll"]["metrics"],
            )
        record_step_status(manifest, spec, eval_name, "completed")


def execute_internal(spec: RunSpec) -> None:
    if spec.job == "train":
        execute_train(spec)
        return
    if spec.job in ("proxy_val", "card2_orig", "card2_roll"):
        execute_eval(spec, spec.job)
        return
    if spec.job == "all":
        execute_train(spec)
        execute_eval(spec, "proxy_val")
        execute_eval(spec, "card2_orig")
        execute_eval(spec, "card2_roll")
        return
    raise ValueError(f"Unsupported job={spec.job!r}")


def print_prepared_summary(record: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "run_slug": record["run_slug"],
                "tmux_session": record["tmux_session"],
                "selected_gpu": record["selected_gpu"],
                "work_dir": record["work_dir"],
                "checkpoint_path": record["checkpoint_path"],
                "step_scripts": record["step_scripts"],
                "handoff_path": record["handoff_path"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    handoff = Path(record["handoff_path"]).read_text(encoding="utf-8")
    print("\n[A2 handoff]\n" + handoff)


def main() -> None:
    args = parse_args()
    validate_paths()
    ensure_dir(A2_ROOT)

    effective_iters = args.override_max_iters if args.override_max_iters is not None else args.budget
    if effective_iters <= 0:
        raise ValueError("--override-max-iters must be positive.")

    spec = RunSpec(
        stage=args.stage,
        budget=args.budget,
        seed=args.seed,
        variant=args.variant,
        job=args.job,
        gpu_request=str(args.gpu),
        batch_size=int(args.batch_size),
        effective_iters=int(effective_iters),
    )

    assets = prepare_run_assets(spec, spec.gpu_request)
    with manifest_transaction() as manifest:
        ensure_batch_size_consistency(manifest, spec)
        record = update_run_record(manifest, spec, assets)

    if args.internal_execute:
        execute_internal(spec)
        return

    if args.launch and not args.dry_run:
        launch_tmux(spec, selected_gpu=record["selected_gpu"], force=args.force)
        with manifest_transaction() as manifest:
            manifest["runs"][spec.run_slug]["launched_at"] = now_iso()
            manifest["runs"][spec.run_slug]["launch_mode"] = "tmux"
            record = manifest["runs"][spec.run_slug]
        print_prepared_summary(record)
        return

    print_prepared_summary(record)


if __name__ == "__main__":
    main()
