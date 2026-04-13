#!/usr/bin/env python3
"""Offline autopsy for JRDB nvidia-aud seam-resolver regressions.

This script compares two tracker outputs on a single JRDB stitched sequence
using TrackEval's JRDB loader and preprocessed sequence data, then replays the
CLEAR matching logic frame by frame to export event-level diagnostics.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


EPS = np.finfo("float").eps


def _patch_numpy_aliases() -> None:
    if "float" not in np.__dict__:
        np.float = float  # type: ignore[attr-defined]
    if "int" not in np.__dict__:
        np.int = int  # type: ignore[attr-defined]
    if "bool" not in np.__dict__:
        np.bool = bool  # type: ignore[attr-defined]


def _ensure_trackeval_importable(trackeval_root: Path):
    _patch_numpy_aliases()
    trackeval_root = trackeval_root.resolve()
    if str(trackeval_root) not in sys.path:
        sys.path.insert(0, str(trackeval_root))
    from trackeval.datasets.jrdb_2d_box import JRDB2DBox

    return JRDB2DBox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline autopsy for nvidia-aud-2019-04-18_0 using TrackEval "
            "preprocessed sequence data and frame-level CLEAR replay."
        )
    )
    parser.add_argument(
        "--trackeval-root",
        default="jrdb_toolkit/tracking_eval/TrackEval",
        help="Path to the TrackEval repo root.",
    )
    parser.add_argument(
        "--gt-folder",
        default=None,
        help="Override GT folder. Defaults to <trackeval-root>/data/gt/jrdb/jrdb_2d_box_val.",
    )
    parser.add_argument(
        "--trackers-folder",
        default=None,
        help=(
            "Override trackers folder. Defaults to "
            "<trackeval-root>/data/trackers/jrdb/jrdb_2d_box_val."
        ),
    )
    parser.add_argument(
        "--baseline-tracker",
        default="OmniTrackTBDDist1VisFix",
        help="Baseline tracker name under the TrackEval trackers folder.",
    )
    parser.add_argument(
        "--compare-tracker",
        default="OmniTrackTBDDist1SeamResolverV3",
        help="Comparison tracker name under the TrackEval trackers folder.",
    )
    parser.add_argument(
        "--filtered-tracker",
        default=None,
        help=(
            "Optional filtered tracker name for the single-variable rerun "
            "comparison table."
        ),
    )
    parser.add_argument(
        "--sequence",
        default="nvidia-aud-2019-04-18_0",
        help="JRDB stitched sequence to autopsy.",
    )
    parser.add_argument(
        "--split-name",
        default="val",
        help="TrackEval split name.",
    )
    parser.add_argument(
        "--class-name",
        default="pedestrian",
        help="TrackEval class name.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="CLEAR matching threshold.",
    )
    parser.add_argument(
        "--image-width",
        type=float,
        default=3760.0,
        help="JRDB stitched panorama width in pixels.",
    )
    parser.add_argument(
        "--seam-band-px",
        type=float,
        default=400.0,
        help="Seam band width in pixels, matching the resolver rule.",
    )
    parser.add_argument(
        "--fp-match-iou-threshold",
        type=float,
        default=0.95,
        help=(
            "Wrap IoU threshold used when cancelling baseline/v3 false positives "
            "that are effectively the same instance."
        ),
    )
    parser.add_argument(
        "--targeted-top-k",
        type=int,
        default=0,
        help="Export targeted per-frame JSON logs for the top-K suspicious frames.",
    )
    parser.add_argument(
        "--runtime-debug-json",
        default=None,
        help=(
            "Optional JSON/JSONL file keyed by frame. When present, targeted logs "
            "will also include runtime fields such as active track groups."
        ),
    )
    parser.add_argument(
        "--runtime-debug-results-pkl",
        default=None,
        help=(
            "Optional debug-enabled results_val.pkl. When provided together with "
            "--runtime-debug-ann-file, targeted logs will hydrate runtime fields "
            "directly from img_bbox.seam_resolver_stats."
        ),
    )
    parser.add_argument(
        "--runtime-debug-ann-file",
        default=None,
        help="Annotation pkl aligned with --runtime-debug-results-pkl.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to "
            "work_dirs/diagnostics/nvidia_aud_2019_04_18_0_autopsy_<baseline>_vs_<compare>."
        ),
    )
    return parser.parse_args()


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    converted = boxes.copy()
    converted[:, 2] = converted[:, 0] + converted[:, 2]
    converted[:, 3] = converted[:, 1] + converted[:, 3]
    return converted


def box_iou_xyxy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    boxes1 = np.asarray(boxes1, dtype=float)
    boxes2 = np.asarray(boxes2, dtype=float)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=float)

    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(bottom_right - top_left, a_min=0.0, a_max=None)
    intersection = wh[..., 0] * wh[..., 1]

    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], a_min=0.0, a_max=None) * np.clip(
        boxes1[:, 3] - boxes1[:, 1], a_min=0.0, a_max=None
    )
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], a_min=0.0, a_max=None) * np.clip(
        boxes2[:, 3] - boxes2[:, 1], a_min=0.0, a_max=None
    )
    union = area1[:, None] + area2[None, :] - intersection
    union[union <= 0.0] = 1.0
    return intersection / union


def wrap_iou_matrix_xyxy(
    boxes1: np.ndarray, boxes2: np.ndarray, image_width: float
) -> np.ndarray:
    boxes1 = np.asarray(boxes1, dtype=float).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=float).reshape(-1, 4)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=float)

    best = None
    for shift in (-float(image_width), 0.0, float(image_width)):
        shifted = boxes2.copy()
        shifted[:, [0, 2]] += shift
        current = box_iou_xyxy(boxes1, shifted)
        best = current if best is None else np.maximum(best, current)
    return best


def seam_mask_xywh(
    boxes_xywh: np.ndarray, image_width: float, seam_band_px: float
) -> np.ndarray:
    boxes_xywh = np.asarray(boxes_xywh, dtype=float).reshape(-1, 4)
    if boxes_xywh.size == 0:
        return np.zeros((0,), dtype=bool)
    x1 = boxes_xywh[:, 0]
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2]
    return (
        (x1 < float(seam_band_px))
        | (x2 > float(image_width) - float(seam_band_px))
        | (x1 < 0.0)
        | (x2 > float(image_width))
    )


def format_box(box_xywh: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in np.asarray(box_xywh)) + "]"


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_runtime_debug(runtime_debug_json: str | None) -> dict[int, dict[str, Any]]:
    if runtime_debug_json is None:
        return {}

    path = Path(runtime_debug_json)
    if not path.is_file():
        raise FileNotFoundError(f"Runtime debug file not found: {path}")

    if path.suffix == ".jsonl":
        entries = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    else:
        with path.open() as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "frames" in payload:
            entries = payload["frames"]
        elif isinstance(payload, dict):
            entries = []
            for frame_key, frame_value in payload.items():
                if isinstance(frame_value, dict):
                    frame_value = dict(frame_value)
                    frame_value.setdefault("frame_idx", int(frame_key))
                    entries.append(frame_value)
        elif isinstance(payload, list):
            entries = payload
        else:
            raise ValueError(f"Unsupported runtime debug payload: {type(payload)!r}")

    debug_by_frame: dict[int, dict[str, Any]] = {}
    for entry in entries:
        frame_idx = entry.get("frame_idx", entry.get("frame"))
        if frame_idx is None:
            continue
        debug_by_frame[int(frame_idx)] = entry
    return debug_by_frame


def load_infos(ann_file: str | Path) -> list[dict[str, Any]]:
    with Path(ann_file).open("rb") as handle:
        payload = pickle.load(handle)
    return payload["infos"] if isinstance(payload, dict) and "infos" in payload else payload


def load_runtime_debug_from_results_pkl(
    *,
    results_pkl: str | None,
    ann_file: str | None,
    sequence: str,
) -> dict[int, dict[str, Any]]:
    if results_pkl is None:
        return {}
    if ann_file is None:
        raise ValueError(
            "--runtime-debug-ann-file is required when --runtime-debug-results-pkl is set."
        )

    results_path = Path(results_pkl)
    if not results_path.is_file():
        raise FileNotFoundError(f"results_val.pkl not found: {results_path}")

    infos = load_infos(ann_file)
    with results_path.open("rb") as handle:
        results = pickle.load(handle)
    if len(results) != len(infos):
        raise ValueError(
            f"results/infos length mismatch: {len(results)} != {len(infos)}"
        )

    debug_by_frame: dict[int, dict[str, Any]] = {}
    for info, result in zip(infos, results):
        token = info["token"]
        seq_name, frame_name = token.rsplit("_", 1)
        if seq_name != sequence:
            continue
        stats = result.get("img_bbox", {}).get("seam_resolver_stats")
        if stats is None:
            continue
        debug_by_frame[int(frame_name)] = stats
    return debug_by_frame


def load_sequence_summary(
    tracker_root: Path, tracker_name: str, sequence: str
) -> dict[str, int] | None:
    summary_path = tracker_root / tracker_name / "pedestrian_summary.csv"
    if not summary_path.is_file():
        return None

    with summary_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("seq") == sequence:
                return {
                    "CLR_TP": int(float(row["CLR_TP"])),
                    "CLR_FN": int(float(row["CLR_FN"])),
                    "CLR_FP": int(float(row["CLR_FP"])),
                    "IDSW": int(float(row["IDSW"])),
                }
    return None


@dataclass
class SequenceData:
    tracker_name: str
    raw_data: dict[str, Any]
    proc_data: dict[str, Any]
    frames: list[dict[str, Any]]
    gt_eval_to_raw: dict[int, int]
    tracker_eval_to_raw: dict[int, int]


def load_sequence_data(
    dataset,
    tracker_name: str,
    sequence: str,
    class_name: str,
) -> SequenceData:
    raw_data = dataset.get_raw_seq_data(tracker_name, sequence, False)
    proc_data = dataset.get_preprocessed_seq_data(raw_data, class_name)
    frames, gt_eval_to_raw, tracker_eval_to_raw = build_preprocessed_frame_lookup(
        raw_data, proc_data, dataset, class_name
    )
    return SequenceData(
        tracker_name=tracker_name,
        raw_data=raw_data,
        proc_data=proc_data,
        frames=frames,
        gt_eval_to_raw=gt_eval_to_raw,
        tracker_eval_to_raw=tracker_eval_to_raw,
    )


def build_preprocessed_frame_lookup(
    raw_data: dict[str, Any],
    proc_data: dict[str, Any],
    dataset,
    class_name: str,
) -> tuple[list[dict[str, Any]], dict[int, int], dict[int, int]]:
    if class_name == "pedestrian":
        distractor_classes = [dataset.class_name_to_class_id["person"]]
    elif class_name == "car":
        distractor_classes = [dataset.class_name_to_class_id["van"]]
    else:
        raise ValueError(f"Unsupported class_name={class_name!r}")

    cls_id = dataset.class_name_to_class_id[class_name]
    unique_gt_ids: list[int] = []
    unique_tracker_ids: list[int] = []
    frame_records: list[dict[str, Any]] = []

    for frame_idx in range(raw_data["num_timesteps"]):
        gt_class_mask = np.sum(
            [
                np.asarray(raw_data["gt_classes"][frame_idx] == class_id)
                for class_id in [cls_id] + distractor_classes
            ],
            axis=0,
        ).astype(bool)
        gt_ids = np.asarray(raw_data["gt_ids"][frame_idx])[gt_class_mask].astype(int)
        gt_boxes = np.asarray(raw_data["gt_dets"][frame_idx], dtype=float)[gt_class_mask]
        gt_classes = np.asarray(raw_data["gt_classes"][frame_idx])[gt_class_mask].astype(int)
        gt_occlusion = np.asarray(raw_data["gt_extras"][frame_idx]["occlusion"])[
            gt_class_mask
        ]
        gt_truncation = np.asarray(raw_data["gt_extras"][frame_idx]["truncation"])[
            gt_class_mask
        ]
        gt_keep_mask = (
            (gt_occlusion <= dataset.max_occlusion + EPS)
            & (gt_truncation <= dataset.max_truncation + EPS)
            & (gt_classes == cls_id)
        )

        tracker_class_mask = np.atleast_1d(
            np.asarray(raw_data["tracker_classes"][frame_idx]) == cls_id
        ).astype(bool)
        tracker_ids = np.asarray(raw_data["tracker_ids"][frame_idx])[tracker_class_mask].astype(int)
        tracker_boxes = np.asarray(raw_data["tracker_dets"][frame_idx], dtype=float)[
            tracker_class_mask
        ]
        tracker_confidences = np.asarray(
            raw_data["tracker_confidences"][frame_idx], dtype=float
        )[tracker_class_mask]

        kept_gt_ids = gt_ids[gt_keep_mask]
        kept_gt_boxes = gt_boxes[gt_keep_mask]

        unique_gt_ids.extend(np.unique(kept_gt_ids).tolist())
        unique_tracker_ids.extend(np.unique(tracker_ids).tolist())

        frame_records.append(
            {
                "gt_raw_ids": kept_gt_ids,
                "gt_boxes": kept_gt_boxes,
                "tracker_raw_ids": tracker_ids,
                "tracker_boxes": tracker_boxes,
                "tracker_confidences": tracker_confidences,
            }
        )

    gt_eval_to_raw: dict[int, int] = {}
    tracker_eval_to_raw: dict[int, int] = {}
    gt_raw_to_eval: dict[int, int] = {}
    tracker_raw_to_eval: dict[int, int] = {}

    if unique_gt_ids:
        for eval_id, raw_id in enumerate(sorted(set(unique_gt_ids))):
            gt_raw_to_eval[int(raw_id)] = int(eval_id)
            gt_eval_to_raw[int(eval_id)] = int(raw_id)
    if unique_tracker_ids:
        for eval_id, raw_id in enumerate(sorted(set(unique_tracker_ids))):
            tracker_raw_to_eval[int(raw_id)] = int(eval_id)
            tracker_eval_to_raw[int(eval_id)] = int(raw_id)

    for frame_idx, frame in enumerate(frame_records):
        gt_eval_ids = np.asarray(
            [gt_raw_to_eval[int(raw_id)] for raw_id in frame["gt_raw_ids"]],
            dtype=int,
        )
        tracker_eval_ids = np.asarray(
            [tracker_raw_to_eval[int(raw_id)] for raw_id in frame["tracker_raw_ids"]],
            dtype=int,
        )
        frame["gt_eval_ids"] = gt_eval_ids
        frame["tracker_eval_ids"] = tracker_eval_ids

        proc_gt_ids = np.asarray(proc_data["gt_ids"][frame_idx], dtype=int)
        proc_tracker_ids = np.asarray(proc_data["tracker_ids"][frame_idx], dtype=int)
        proc_gt_boxes = np.asarray(proc_data["gt_dets"][frame_idx], dtype=float)
        proc_tracker_boxes = np.asarray(proc_data["tracker_dets"][frame_idx], dtype=float)
        if not np.array_equal(gt_eval_ids, proc_gt_ids):
            raise ValueError(
                f"GT id relabel mismatch on frame {frame_idx}: "
                f"{gt_eval_ids.tolist()} != {proc_gt_ids.tolist()}"
            )
        if not np.array_equal(tracker_eval_ids, proc_tracker_ids):
            raise ValueError(
                f"Tracker id relabel mismatch on frame {frame_idx}: "
                f"{tracker_eval_ids.tolist()} != {proc_tracker_ids.tolist()}"
            )
        if not np.allclose(frame["gt_boxes"], proc_gt_boxes):
            raise ValueError(f"GT box mismatch on frame {frame_idx}.")
        if not np.allclose(frame["tracker_boxes"], proc_tracker_boxes):
            raise ValueError(f"Tracker box mismatch on frame {frame_idx}.")

    return frame_records, gt_eval_to_raw, tracker_eval_to_raw


def simulate_clear_events(
    sequence_data: SequenceData,
    *,
    threshold: float,
    image_width: float,
    seam_band_px: float,
) -> dict[str, Any]:
    proc_data = sequence_data.proc_data
    prev_tracker_id = np.nan * np.zeros(proc_data["num_gt_ids"])
    prev_timestep_tracker_id = np.nan * np.zeros(proc_data["num_gt_ids"])
    totals = dict(tp=0, fn=0, fp=0, idsw=0, fp_seam=0, fp_non_seam=0)
    frame_events: list[dict[str, Any]] = []

    for frame_idx, (gt_ids_t, tracker_ids_t) in enumerate(
        zip(proc_data["gt_ids"], proc_data["tracker_ids"])
    ):
        gt_ids_t = np.asarray(gt_ids_t, dtype=int)
        tracker_ids_t = np.asarray(tracker_ids_t, dtype=int)
        gt_boxes_t = np.asarray(proc_data["gt_dets"][frame_idx], dtype=float)
        tracker_boxes_t = np.asarray(proc_data["tracker_dets"][frame_idx], dtype=float)
        similarity = np.asarray(proc_data["similarity_scores"][frame_idx], dtype=float)
        tracker_seam_mask = seam_mask_xywh(tracker_boxes_t, image_width, seam_band_px)

        matched_rows = np.empty((0,), dtype=int)
        matched_cols = np.empty((0,), dtype=int)
        matched_gt_ids = np.empty((0,), dtype=int)
        matched_tracker_ids = np.empty((0,), dtype=int)
        match_scores = np.empty((0,), dtype=float)
        is_idsw = np.empty((0,), dtype=bool)

        if len(gt_ids_t) == 0:
            unmatched_gt_indices = np.empty((0,), dtype=int)
            unmatched_tracker_indices = np.arange(len(tracker_ids_t), dtype=int)
            tp = 0
            fn = 0
            fp = len(unmatched_tracker_indices)
            idsw = 0
        elif len(tracker_ids_t) == 0:
            unmatched_gt_indices = np.arange(len(gt_ids_t), dtype=int)
            unmatched_tracker_indices = np.empty((0,), dtype=int)
            tp = 0
            fn = len(unmatched_gt_indices)
            fp = 0
            idsw = 0
            frame_events.append(
                build_frame_event(
                    sequence_data=sequence_data,
                    frame_idx=frame_idx,
                    gt_ids_t=gt_ids_t,
                    tracker_ids_t=tracker_ids_t,
                    gt_boxes_t=gt_boxes_t,
                    tracker_boxes_t=tracker_boxes_t,
                    tracker_seam_mask=tracker_seam_mask,
                    similarity=similarity,
                    matched_rows=matched_rows,
                    matched_cols=matched_cols,
                    matched_gt_ids=matched_gt_ids,
                    matched_tracker_ids=matched_tracker_ids,
                    match_scores=match_scores,
                    is_idsw=is_idsw,
                    unmatched_gt_indices=unmatched_gt_indices,
                    unmatched_tracker_indices=unmatched_tracker_indices,
                    tp=tp,
                    fn=fn,
                    fp=fp,
                    idsw=idsw,
                    image_width=image_width,
                    seam_band_px=seam_band_px,
                )
            )
            totals["tp"] += tp
            totals["fn"] += fn
            continue
        else:
            score_mat = (
                tracker_ids_t[np.newaxis, :]
                == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]]
            )
            score_mat = 1000.0 * score_mat + similarity
            score_mat[similarity < threshold - EPS] = 0.0

            raw_match_rows, raw_match_cols = linear_sum_assignment(-score_mat)
            actually_matched_mask = score_mat[raw_match_rows, raw_match_cols] > 0.0 + EPS
            matched_rows = raw_match_rows[actually_matched_mask]
            matched_cols = raw_match_cols[actually_matched_mask]
            matched_gt_ids = gt_ids_t[matched_rows]
            matched_tracker_ids = tracker_ids_t[matched_cols]
            match_scores = similarity[matched_rows, matched_cols]

            prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
            is_idsw = (~np.isnan(prev_matched_tracker_ids)) & (
                matched_tracker_ids != prev_matched_tracker_ids
            )
            idsw = int(np.sum(is_idsw))

            prev_tracker_id[matched_gt_ids] = matched_tracker_ids
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids

            unmatched_gt_indices = np.setdiff1d(
                np.arange(len(gt_ids_t), dtype=int), matched_rows, assume_unique=True
            )
            unmatched_tracker_indices = np.setdiff1d(
                np.arange(len(tracker_ids_t), dtype=int), matched_cols, assume_unique=True
            )
            tp = int(len(matched_rows))
            fn = int(len(unmatched_gt_indices))
            fp = int(len(unmatched_tracker_indices))

        frame_event = build_frame_event(
            sequence_data=sequence_data,
            frame_idx=frame_idx,
            gt_ids_t=gt_ids_t,
            tracker_ids_t=tracker_ids_t,
            gt_boxes_t=gt_boxes_t,
            tracker_boxes_t=tracker_boxes_t,
            tracker_seam_mask=tracker_seam_mask,
            similarity=similarity,
            matched_rows=matched_rows,
            matched_cols=matched_cols,
            matched_gt_ids=matched_gt_ids,
            matched_tracker_ids=matched_tracker_ids,
            match_scores=match_scores,
            is_idsw=is_idsw,
            unmatched_gt_indices=unmatched_gt_indices,
            unmatched_tracker_indices=unmatched_tracker_indices,
            tp=tp,
            fn=fn,
            fp=fp,
            idsw=idsw,
            image_width=image_width,
            seam_band_px=seam_band_px,
        )
        frame_events.append(frame_event)
        totals["tp"] += tp
        totals["fn"] += fn
        totals["fp"] += fp
        totals["idsw"] += idsw
        totals["fp_seam"] += frame_event["fp_seam"]
        totals["fp_non_seam"] += frame_event["fp_non_seam"]

    return {
        "tracker_name": sequence_data.tracker_name,
        "totals": totals,
        "frames": frame_events,
    }


def build_frame_event(
    *,
    sequence_data: SequenceData,
    frame_idx: int,
    gt_ids_t: np.ndarray,
    tracker_ids_t: np.ndarray,
    gt_boxes_t: np.ndarray,
    tracker_boxes_t: np.ndarray,
    tracker_seam_mask: np.ndarray,
    similarity: np.ndarray,
    matched_rows: np.ndarray,
    matched_cols: np.ndarray,
    matched_gt_ids: np.ndarray,
    matched_tracker_ids: np.ndarray,
    match_scores: np.ndarray,
    is_idsw: np.ndarray,
    unmatched_gt_indices: np.ndarray,
    unmatched_tracker_indices: np.ndarray,
    tp: int,
    fn: int,
    fp: int,
    idsw: int,
    image_width: float,
    seam_band_px: float,
) -> dict[str, Any]:
    frame_lookup = sequence_data.frames[frame_idx]
    matched_pairs: list[dict[str, Any]] = []
    for local_idx, (match_row, match_col) in enumerate(zip(matched_rows, matched_cols)):
        matched_pairs.append(
            {
                "gt_eval_id": int(matched_gt_ids[local_idx]),
                "gt_raw_id": int(frame_lookup["gt_raw_ids"][match_row]),
                "tracker_eval_id": int(matched_tracker_ids[local_idx]),
                "tracker_raw_id": int(frame_lookup["tracker_raw_ids"][match_col]),
                "iou": float(match_scores[local_idx]),
                "idsw": bool(is_idsw[local_idx]) if len(is_idsw) else False,
                "gt_box_xywh": np.asarray(gt_boxes_t[match_row], dtype=float).tolist(),
                "tracker_box_xywh": np.asarray(tracker_boxes_t[match_col], dtype=float).tolist(),
            }
        )

    unmatched_trackers: list[dict[str, Any]] = []
    for tracker_idx in unmatched_tracker_indices.tolist():
        best_gt_iou = (
            float(np.max(similarity[:, tracker_idx])) if similarity.size else 0.0
        )
        unmatched_trackers.append(
            {
                "frame_idx": frame_idx,
                "frame_number": frame_idx + 1,
                "track_id": int(frame_lookup["tracker_raw_ids"][tracker_idx]),
                "trackeval_track_id": int(tracker_ids_t[tracker_idx]),
                "box_xywh": np.asarray(tracker_boxes_t[tracker_idx], dtype=float).tolist(),
                "box": format_box(tracker_boxes_t[tracker_idx]),
                "score": float(frame_lookup["tracker_confidences"][tracker_idx]),
                "seam_flag": bool(tracker_seam_mask[tracker_idx]),
                "best_gt_iou": best_gt_iou,
            }
        )

    unmatched_gts: list[dict[str, Any]] = []
    for gt_idx in unmatched_gt_indices.tolist():
        best_tracker_iou = (
            float(np.max(similarity[gt_idx, :])) if similarity.size else 0.0
        )
        unmatched_gts.append(
            {
                "frame_idx": frame_idx,
                "frame_number": frame_idx + 1,
                "gt_id": int(frame_lookup["gt_raw_ids"][gt_idx]),
                "trackeval_gt_id": int(gt_ids_t[gt_idx]),
                "box_xywh": np.asarray(gt_boxes_t[gt_idx], dtype=float).tolist(),
                "box": format_box(gt_boxes_t[gt_idx]),
                "best_tracker_iou": best_tracker_iou,
                "seam_flag": bool(
                    seam_mask_xywh(gt_boxes_t[[gt_idx]], image_width, seam_band_px)[0]
                ),
            }
        )

    fp_seam = int(np.sum(tracker_seam_mask[unmatched_tracker_indices]))
    fp_non_seam = int(len(unmatched_tracker_indices) - fp_seam)
    return {
        "frame_idx": frame_idx,
        "frame_number": frame_idx + 1,
        "gt_dets": int(len(gt_ids_t)),
        "tracker_dets": int(len(tracker_ids_t)),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "idsw": int(idsw),
        "fp_seam": fp_seam,
        "fp_non_seam": fp_non_seam,
        "matched_pairs": matched_pairs,
        "unmatched_trackers": unmatched_trackers,
        "unmatched_gts": unmatched_gts,
        "gt_boxes_xywh": np.asarray(gt_boxes_t, dtype=float).tolist(),
        "tracker_boxes_xywh": np.asarray(tracker_boxes_t, dtype=float).tolist(),
        "tracker_seam_mask": tracker_seam_mask.tolist(),
    }


def match_common_false_positives(
    baseline_instances: list[dict[str, Any]],
    compare_instances: list[dict[str, Any]],
    *,
    image_width: float,
    iou_threshold: float,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    baseline_boxes = np.asarray([row["box_xywh"] for row in baseline_instances], dtype=float)
    compare_boxes = np.asarray([row["box_xywh"] for row in compare_instances], dtype=float)
    baseline_xyxy = xywh_to_xyxy(baseline_boxes)
    compare_xyxy = xywh_to_xyxy(compare_boxes)

    matched_baseline: dict[int, dict[str, Any]] = {}
    matched_compare: dict[int, dict[str, Any]] = {}

    baseline_by_track: dict[int, int] = {}
    for baseline_idx, row in enumerate(baseline_instances):
        baseline_by_track[int(row["track_id"])] = baseline_idx

    remaining_compare = []
    remaining_baseline = set(range(len(baseline_instances)))
    for compare_idx, row in enumerate(compare_instances):
        track_id = int(row["track_id"])
        if track_id in baseline_by_track:
            baseline_idx = baseline_by_track[track_id]
            matched_compare[compare_idx] = {
                "baseline_index": baseline_idx,
                "match_type": "track_id",
                "wrap_iou": float(
                    wrap_iou_matrix_xyxy(
                        compare_xyxy[[compare_idx]], baseline_xyxy[[baseline_idx]], image_width
                    )[0, 0]
                ),
            }
            matched_baseline[baseline_idx] = {
                "compare_index": compare_idx,
                "match_type": "track_id",
            }
            remaining_baseline.discard(baseline_idx)
        else:
            remaining_compare.append(compare_idx)

    if remaining_compare and remaining_baseline:
        remaining_compare = np.asarray(remaining_compare, dtype=int)
        remaining_baseline = np.asarray(sorted(remaining_baseline), dtype=int)
        iou_matrix = wrap_iou_matrix_xyxy(
            compare_xyxy[remaining_compare], baseline_xyxy[remaining_baseline], image_width
        )
        if iou_matrix.size:
            match_rows, match_cols = linear_sum_assignment(-iou_matrix)
            valid = iou_matrix[match_rows, match_cols] >= float(iou_threshold)
            for row_idx, col_idx in zip(match_rows[valid], match_cols[valid]):
                compare_idx = int(remaining_compare[row_idx])
                baseline_idx = int(remaining_baseline[col_idx])
                matched_compare[compare_idx] = {
                    "baseline_index": baseline_idx,
                    "match_type": "wrap_iou",
                    "wrap_iou": float(iou_matrix[row_idx, col_idx]),
                }
                matched_baseline[baseline_idx] = {
                    "compare_index": compare_idx,
                    "match_type": "wrap_iou",
                }

    return matched_baseline, matched_compare


def build_v3_only_fp_rows(
    baseline_result: dict[str, Any],
    compare_result: dict[str, Any],
    *,
    image_width: float,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline_frame, compare_frame in zip(
        baseline_result["frames"], compare_result["frames"]
    ):
        baseline_instances = baseline_frame["unmatched_trackers"]
        compare_instances = compare_frame["unmatched_trackers"]
        _, matched_compare = match_common_false_positives(
            baseline_instances,
            compare_instances,
            image_width=image_width,
            iou_threshold=iou_threshold,
        )
        for compare_idx, row in enumerate(compare_instances):
            if compare_idx in matched_compare:
                continue
            rows.append(
                {
                    "frame_idx": row["frame_idx"],
                    "frame_number": row["frame_number"],
                    "track_id": row["track_id"],
                    "trackeval_track_id": row["trackeval_track_id"],
                    "box": row["box"],
                    "x": float(row["box_xywh"][0]),
                    "y": float(row["box_xywh"][1]),
                    "w": float(row["box_xywh"][2]),
                    "h": float(row["box_xywh"][3]),
                    "score": float(row["score"]),
                    "seam_flag": bool(row["seam_flag"]),
                    "best_gt_iou": float(row["best_gt_iou"]),
                }
            )
    rows.sort(
        key=lambda row: (
            int(row["frame_idx"]),
            int(row["track_id"]),
            -float(row["best_gt_iou"]),
        )
    )
    return rows


def build_frame_comparison_rows(
    sequence: str,
    baseline_result: dict[str, Any],
    compare_result: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline_frame, compare_frame in zip(
        baseline_result["frames"], compare_result["frames"]
    ):
        tp_delta = int(compare_frame["tp"] - baseline_frame["tp"])
        fn_delta = int(compare_frame["fn"] - baseline_frame["fn"])
        fp_delta = int(compare_frame["fp"] - baseline_frame["fp"])
        idsw_delta = int(compare_frame["idsw"] - baseline_frame["idsw"])
        seam_fp_delta = int(compare_frame["fp_seam"] - baseline_frame["fp_seam"])
        non_seam_fp_delta = int(
            compare_frame["fp_non_seam"] - baseline_frame["fp_non_seam"]
        )
        rows.append(
            {
                "sequence": sequence,
                "frame_idx": baseline_frame["frame_idx"],
                "frame_number": baseline_frame["frame_number"],
                "gt_dets": baseline_frame["gt_dets"],
                "baseline_tracker_dets": baseline_frame["tracker_dets"],
                "baseline_TP": baseline_frame["tp"],
                "baseline_FN": baseline_frame["fn"],
                "baseline_FP": baseline_frame["fp"],
                "baseline_IDSW": baseline_frame["idsw"],
                "baseline_FP_seam": baseline_frame["fp_seam"],
                "baseline_FP_non_seam": baseline_frame["fp_non_seam"],
                "v3_tracker_dets": compare_frame["tracker_dets"],
                "v3_TP": compare_frame["tp"],
                "v3_FN": compare_frame["fn"],
                "v3_FP": compare_frame["fp"],
                "v3_IDSW": compare_frame["idsw"],
                "v3_FP_seam": compare_frame["fp_seam"],
                "v3_FP_non_seam": compare_frame["fp_non_seam"],
                "TP_delta": tp_delta,
                "FN_delta": fn_delta,
                "FP_delta": fp_delta,
                "IDSW_delta": idsw_delta,
                "seam_FP_delta": seam_fp_delta,
                "non_seam_FP_delta": non_seam_fp_delta,
            }
        )
    return rows


def build_priority_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        frame_rows,
        key=lambda row: (
            int(row["IDSW_delta"]),
            int(row["FP_delta"]),
            int(row["FN_delta"]),
            -int(row["TP_delta"]),
            int(row["seam_FP_delta"]),
        ),
        reverse=True,
    )
    priority_rows = []
    for rank, row in enumerate(sorted_rows, start=1):
        row_with_rank = dict(row)
        row_with_rank["priority_rank"] = rank
        priority_rows.append(row_with_rank)
    return priority_rows


def export_targeted_logs(
    *,
    output_dir: Path,
    targeted_top_k: int,
    priority_rows: list[dict[str, Any]],
    baseline_result: dict[str, Any],
    compare_result: dict[str, Any],
    image_width: float,
    seam_band_px: float,
    runtime_debug_by_frame: dict[int, dict[str, Any]],
) -> None:
    if targeted_top_k <= 0:
        return

    targeted_dir = output_dir / "targeted_logs"
    targeted_dir.mkdir(parents=True, exist_ok=True)

    for row in priority_rows[:targeted_top_k]:
        frame_idx = int(row["frame_idx"])
        baseline_frame = baseline_result["frames"][frame_idx]
        compare_frame = compare_result["frames"][frame_idx]

        compare_tracker_boxes = np.asarray(compare_frame["tracker_boxes_xywh"], dtype=float)
        compare_seam_mask = seam_mask_xywh(compare_tracker_boxes, image_width, seam_band_px)
        compare_seam_boxes = compare_tracker_boxes[compare_seam_mask]
        compare_pairwise_wrap_iou = wrap_iou_matrix_xyxy(
            xywh_to_xyxy(compare_seam_boxes),
            xywh_to_xyxy(compare_seam_boxes),
            image_width,
        )

        baseline_tracker_boxes = np.asarray(baseline_frame["tracker_boxes_xywh"], dtype=float)
        baseline_seam_mask = seam_mask_xywh(
            baseline_tracker_boxes, image_width, seam_band_px
        )
        baseline_seam_boxes = baseline_tracker_boxes[baseline_seam_mask]
        baseline_pairwise_wrap_iou = wrap_iou_matrix_xyxy(
            xywh_to_xyxy(baseline_seam_boxes),
            xywh_to_xyxy(baseline_seam_boxes),
            image_width,
        )

        payload = {
            "frame_summary": row,
            "baseline": {
                "matched_pairs": baseline_frame["matched_pairs"],
                "unmatched_trackers": baseline_frame["unmatched_trackers"],
                "unmatched_gts": baseline_frame["unmatched_gts"],
                "seam_candidates": compare_serialized_seam_candidates(
                    baseline_tracker_boxes, baseline_seam_mask
                ),
                "pairwise_wrap_iou": baseline_pairwise_wrap_iou.tolist(),
            },
            "compare": {
                "matched_pairs": compare_frame["matched_pairs"],
                "unmatched_trackers": compare_frame["unmatched_trackers"],
                "unmatched_gts": compare_frame["unmatched_gts"],
                "seam_candidates": compare_serialized_seam_candidates(
                    compare_tracker_boxes, compare_seam_mask
                ),
                "pairwise_wrap_iou": compare_pairwise_wrap_iou.tolist(),
            },
        }
        if frame_idx in runtime_debug_by_frame:
            payload["runtime_debug"] = runtime_debug_by_frame[frame_idx]
            payload["stale_active_track_hypotheses"] = infer_stale_active_track_blockers(
                runtime_debug_by_frame[frame_idx]
            )

        target_path = targeted_dir / f"frame_{frame_idx:06d}.json"
        with target_path.open("w") as handle:
            json.dump(_to_serializable(payload), handle, indent=2, sort_keys=True)


def compare_serialized_seam_candidates(
    tracker_boxes_xywh: np.ndarray, seam_mask: np.ndarray
) -> list[dict[str, Any]]:
    candidates = []
    for local_idx, box in enumerate(np.asarray(tracker_boxes_xywh)[seam_mask]):
        candidates.append(
            {
                "seam_local_index": local_idx,
                "box_xywh": np.asarray(box, dtype=float).tolist(),
                "box": format_box(box),
            }
        )
    return candidates


def validate_summary(
    *,
    tracker_root: Path,
    tracker_name: str,
    sequence: str,
    totals: dict[str, int],
) -> dict[str, Any]:
    summary = load_sequence_summary(tracker_root, tracker_name, sequence)
    result = {
        "tracker_name": tracker_name,
        "summary_available": summary is not None,
        "match": True,
        "event_totals": {
            "CLR_TP": int(totals["tp"]),
            "CLR_FN": int(totals["fn"]),
            "CLR_FP": int(totals["fp"]),
            "IDSW": int(totals["idsw"]),
        },
        "trackeval_summary": summary,
    }
    if summary is None:
        return result

    result["match"] = result["event_totals"] == summary
    return result


def suppress_metric_stdout(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def evaluate_sequence_metrics(proc_data: dict[str, Any]) -> dict[str, Any]:
    from trackeval.metrics.clear import CLEAR
    from trackeval.metrics.hota import HOTA
    from trackeval.metrics.identity import Identity

    hota = HOTA()
    clear = CLEAR({"PRINT_CONFIG": False})
    identity = Identity({"PRINT_CONFIG": False})
    hota_res = suppress_metric_stdout(hota.eval_sequence, proc_data)
    clear_res = suppress_metric_stdout(clear.eval_sequence, proc_data)
    identity_res = suppress_metric_stdout(identity.eval_sequence, proc_data)
    return {
        "HOTA": float(np.mean(hota_res["HOTA"]) * 100.0),
        "DetA": float(np.mean(hota_res["DetA"]) * 100.0),
        "AssA": float(np.mean(hota_res["AssA"]) * 100.0),
        "IDF1": float(identity_res["IDF1"] * 100.0),
        "TP": int(clear_res["CLR_TP"]),
        "FN": int(clear_res["CLR_FN"]),
        "FP": int(clear_res["CLR_FP"]),
        "IDSW": int(clear_res["IDSW"]),
        "Frag": int(clear_res["Frag"]),
    }


def infer_stale_active_track_blockers(
    runtime_debug: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not runtime_debug:
        return []

    track_debug = runtime_debug.get("track_debug") or {}
    active_track_debug = runtime_debug.get("active_track_debug") or {}
    pairwise_wrap_iou = runtime_debug.get("pairwise_wrap_iou")
    seam_labels = runtime_debug.get("seam_labels") or []
    match_iou = runtime_debug.get("match_iou")
    class_strict = bool(runtime_debug.get("class_strict", False))
    best_group_assignments = track_debug.get("best_track_group_assignments")
    active_group_ids = track_debug.get("active_track_group_ids")
    retained_tracks = active_track_debug.get("retained") or []

    if (
        pairwise_wrap_iou is None
        or match_iou is None
        or best_group_assignments is None
        or active_group_ids is None
    ):
        return []

    hypotheses = []
    num_candidates = len(pairwise_wrap_iou)
    for left_idx in range(num_candidates):
        for right_idx in range(left_idx + 1, num_candidates):
            if class_strict and seam_labels[left_idx] != seam_labels[right_idx]:
                continue
            if float(pairwise_wrap_iou[left_idx][right_idx]) < float(match_iou):
                continue

            left_group = int(best_group_assignments[left_idx])
            right_group = int(best_group_assignments[right_idx])
            if left_group < 0 or right_group < 0 or left_group == right_group:
                continue

            blocking_tracks = []
            stale_present = False
            for active_idx, group_id in enumerate(active_group_ids):
                if int(group_id) not in {left_group, right_group}:
                    continue
                meta = (
                    dict(retained_tracks[active_idx])
                    if active_idx < len(retained_tracks)
                    else {"active_track_index": active_idx}
                )
                blocking_tracks.append(meta)
                time_since_update = meta.get("time_since_update")
                if time_since_update is not None and int(time_since_update) > 0:
                    stale_present = True

            hypotheses.append(
                {
                    "candidate_pair": [int(left_idx), int(right_idx)],
                    "wrap_iou": float(pairwise_wrap_iou[left_idx][right_idx]),
                    "blocking_group_ids": [left_group, right_group],
                    "blocking_tracks": blocking_tracks,
                    "stale_active_track_condition_met": bool(stale_present),
                }
            )
    return hypotheses


def main() -> None:
    args = parse_args()

    trackeval_root = Path(args.trackeval_root).resolve()
    gt_folder = (
        Path(args.gt_folder).resolve()
        if args.gt_folder
        else trackeval_root / "data/gt/jrdb/jrdb_2d_box_val"
    )
    trackers_folder = (
        Path(args.trackers_folder).resolve()
        if args.trackers_folder
        else trackeval_root / "data/trackers/jrdb/jrdb_2d_box_val"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (
            Path("work_dirs")
            / "diagnostics"
            / f"{args.sequence}_autopsy_{args.baseline_tracker}_vs_{args.compare_tracker}"
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    JRDB2DBox = _ensure_trackeval_importable(trackeval_root)
    dataset = JRDB2DBox(
        {
            "GT_FOLDER": str(gt_folder),
            "TRACKERS_FOLDER": str(trackers_folder),
            "TRACKERS_TO_EVAL": [args.baseline_tracker, args.compare_tracker],
            "SPLIT_TO_EVAL": args.split_name,
            "PRINT_CONFIG": False,
        }
    )

    baseline_data = load_sequence_data(
        dataset, args.baseline_tracker, args.sequence, args.class_name
    )
    compare_data = load_sequence_data(
        dataset, args.compare_tracker, args.sequence, args.class_name
    )
    filtered_data = None
    if args.filtered_tracker:
        filtered_data = load_sequence_data(
            dataset, args.filtered_tracker, args.sequence, args.class_name
        )

    baseline_result = simulate_clear_events(
        baseline_data,
        threshold=args.threshold,
        image_width=args.image_width,
        seam_band_px=args.seam_band_px,
    )
    compare_result = simulate_clear_events(
        compare_data,
        threshold=args.threshold,
        image_width=args.image_width,
        seam_band_px=args.seam_band_px,
    )

    frame_rows = build_frame_comparison_rows(
        args.sequence, baseline_result, compare_result
    )
    priority_rows = build_priority_rows(frame_rows)
    v3_only_fp_rows = build_v3_only_fp_rows(
        baseline_result,
        compare_result,
        image_width=args.image_width,
        iou_threshold=args.fp_match_iou_threshold,
    )

    frame_fieldnames = [
        "sequence",
        "frame_idx",
        "frame_number",
        "gt_dets",
        "baseline_tracker_dets",
        "baseline_TP",
        "baseline_FN",
        "baseline_FP",
        "baseline_IDSW",
        "baseline_FP_seam",
        "baseline_FP_non_seam",
        "v3_tracker_dets",
        "v3_TP",
        "v3_FN",
        "v3_FP",
        "v3_IDSW",
        "v3_FP_seam",
        "v3_FP_non_seam",
        "TP_delta",
        "FN_delta",
        "FP_delta",
        "IDSW_delta",
        "seam_FP_delta",
        "non_seam_FP_delta",
    ]
    write_csv(output_dir / "frame_event_breakdown.csv", frame_rows, frame_fieldnames)
    write_csv(
        output_dir / "frame_priority_list.csv",
        priority_rows,
        ["priority_rank"] + frame_fieldnames,
    )
    write_csv(
        output_dir / "fp_instances_v3_only.csv",
        v3_only_fp_rows,
        [
            "frame_idx",
            "frame_number",
            "track_id",
            "trackeval_track_id",
            "box",
            "x",
            "y",
            "w",
            "h",
            "score",
            "seam_flag",
            "best_gt_iou",
        ],
    )

    runtime_debug_by_frame = load_runtime_debug(args.runtime_debug_json)
    runtime_debug_from_pkl = load_runtime_debug_from_results_pkl(
        results_pkl=args.runtime_debug_results_pkl,
        ann_file=args.runtime_debug_ann_file,
        sequence=args.sequence,
    )
    runtime_debug_by_frame.update(runtime_debug_from_pkl)
    export_targeted_logs(
        output_dir=output_dir,
        targeted_top_k=args.targeted_top_k,
        priority_rows=priority_rows,
        baseline_result=baseline_result,
        compare_result=compare_result,
        image_width=args.image_width,
        seam_band_px=args.seam_band_px,
        runtime_debug_by_frame=runtime_debug_by_frame,
    )

    baseline_validation = validate_summary(
        tracker_root=trackers_folder,
        tracker_name=args.baseline_tracker,
        sequence=args.sequence,
        totals=baseline_result["totals"],
    )
    compare_validation = validate_summary(
        tracker_root=trackers_folder,
        tracker_name=args.compare_tracker,
        sequence=args.sequence,
        totals=compare_result["totals"],
    )

    sequence_metric_rows = [
        {
            "tracker_label": "baseline",
            "tracker_name": args.baseline_tracker,
            **evaluate_sequence_metrics(baseline_data.proc_data),
            "FP_seam": int(baseline_result["totals"]["fp_seam"]),
            "FP_non_seam": int(baseline_result["totals"]["fp_non_seam"]),
        },
        {
            "tracker_label": "current_v3",
            "tracker_name": args.compare_tracker,
            **evaluate_sequence_metrics(compare_data.proc_data),
            "FP_seam": int(compare_result["totals"]["fp_seam"]),
            "FP_non_seam": int(compare_result["totals"]["fp_non_seam"]),
        },
    ]
    filtered_validation = None
    filtered_result = None
    if filtered_data is not None:
        filtered_result = simulate_clear_events(
            filtered_data,
            threshold=args.threshold,
            image_width=args.image_width,
            seam_band_px=args.seam_band_px,
        )
        filtered_validation = validate_summary(
            tracker_root=trackers_folder,
            tracker_name=args.filtered_tracker,
            sequence=args.sequence,
            totals=filtered_result["totals"],
        )
        sequence_metric_rows.append(
            {
                "tracker_label": "filtered",
                "tracker_name": args.filtered_tracker,
                **evaluate_sequence_metrics(filtered_data.proc_data),
                "FP_seam": int(filtered_result["totals"]["fp_seam"]),
                "FP_non_seam": int(filtered_result["totals"]["fp_non_seam"]),
            }
        )

    write_csv(
        output_dir / "sequence_metric_comparison.csv",
        sequence_metric_rows,
        [
            "tracker_label",
            "tracker_name",
            "HOTA",
            "DetA",
            "AssA",
            "IDF1",
            "TP",
            "FN",
            "FP",
            "IDSW",
            "Frag",
            "FP_seam",
            "FP_non_seam",
        ],
    )

    run_summary = {
        "sequence": args.sequence,
        "baseline_tracker": args.baseline_tracker,
        "compare_tracker": args.compare_tracker,
        "filtered_tracker": args.filtered_tracker,
        "threshold": args.threshold,
        "image_width": args.image_width,
        "seam_band_px": args.seam_band_px,
        "fp_match_iou_threshold": args.fp_match_iou_threshold,
        "baseline_totals": baseline_result["totals"],
        "compare_totals": compare_result["totals"],
        "summary_validation": {
            "baseline": baseline_validation,
            "compare": compare_validation,
            "filtered": filtered_validation,
        },
        "output_files": [
            "frame_event_breakdown.csv",
            "frame_priority_list.csv",
            "fp_instances_v3_only.csv",
            "sequence_metric_comparison.csv",
        ],
    }
    with (output_dir / "run_summary.json").open("w") as handle:
        json.dump(_to_serializable(run_summary), handle, indent=2, sort_keys=True)

    print(f"Wrote diagnostics to: {output_dir}")
    print(
        "Baseline totals:",
        baseline_validation["event_totals"],
        "| summary match:",
        baseline_validation["match"],
    )
    print(
        "Compare totals:",
        compare_validation["event_totals"],
        "| summary match:",
        compare_validation["match"],
    )
    if priority_rows:
        top_row = priority_rows[0]
        print(
            "Top suspicious frame:",
            f"frame={top_row['frame_idx']}",
            f"IDSW_delta={top_row['IDSW_delta']}",
            f"FP_delta={top_row['FP_delta']}",
            f"FN_delta={top_row['FN_delta']}",
            f"TP_delta={top_row['TP_delta']}",
            f"seam_FP_delta={top_row['seam_FP_delta']}",
        )


if __name__ == "__main__":
    main()
