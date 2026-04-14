#!/usr/bin/env python
import argparse
import contextlib
import csv
import importlib.util
import io
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


if "float" not in np.__dict__:
    np.float = float
if "int" not in np.__dict__:
    np.int = int
if "bool" not in np.__dict__:
    np.bool = bool


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TRACKEVAL_ROOT = REPO_ROOT / "jrdb_toolkit/tracking_eval/TrackEval"
if str(TRACKEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKEVAL_ROOT))

GEOMETRY_MODULE_PATH = (
    REPO_ROOT
    / "projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/association_geometry.py"
)
GEOMETRY_SPEC = importlib.util.spec_from_file_location(
    "diag_association_geometry",
    GEOMETRY_MODULE_PATH,
)
GEOMETRY_MODULE = importlib.util.module_from_spec(GEOMETRY_SPEC)
GEOMETRY_SPEC.loader.exec_module(GEOMETRY_MODULE)
high_lat_mask_xywh = GEOMETRY_MODULE.high_lat_mask_xywh
seam_mask_xywh = GEOMETRY_MODULE.seam_mask_xywh
from trackeval.datasets.jrdb_2d_box import JRDB2DBox  # noqa: E402
from trackeval.metrics.clear import CLEAR  # noqa: E402
from trackeval.metrics.hota import HOTA  # noqa: E402
from trackeval.metrics.identity import Identity  # noqa: E402


SUBSET_NAMES = ("full", "seam", "non_seam", "high_lat", "seam_high_lat")
LEAKAGE_RULES = (
    (0.01, "mechanism_local"),
    (0.05, "descriptive_with_caveat"),
    (math.inf, "descriptive_only_no_strong_causal_claim"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute seam-conditioned TrackEval metric splits without modifying TrackEval core."
    )
    parser.add_argument(
        "--tracker-names",
        nargs="+",
        required=True,
        help="Tracker names under the TrackEval tracker root.",
    )
    parser.add_argument(
        "--gt-folder",
        default="jrdb_toolkit/tracking_eval/TrackEval/data/gt/jrdb/jrdb_2d_box_val",
        help="TrackEval GT root.",
    )
    parser.add_argument(
        "--trackers-folder",
        default="jrdb_toolkit/tracking_eval/TrackEval/data/trackers/jrdb/jrdb_2d_box_val",
        help="TrackEval tracker root.",
    )
    parser.add_argument(
        "--split-name",
        default="val",
        help="JRDB split tag for the TrackEval dataset config.",
    )
    parser.add_argument(
        "--tracker-sub-folder",
        default="data",
        help="Tracker sub-folder used by TrackEval.",
    )
    parser.add_argument(
        "--class-name",
        default="pedestrian",
        help="Class name passed into TrackEval preprocessing.",
    )
    parser.add_argument(
        "--image-width",
        type=float,
        default=3760.0,
        help="ERP stitched image width used by the seam rule.",
    )
    parser.add_argument(
        "--image-height",
        type=float,
        default=480.0,
        help="ERP stitched image height used by the high-lat rule.",
    )
    parser.add_argument(
        "--seam-band-px",
        type=float,
        default=400.0,
        help="Band width for seam-conditioned membership.",
    )
    parser.add_argument(
        "--high-lat-deg",
        type=float,
        default=45.0,
        help="Absolute latitude threshold used for the high-lat slice.",
    )
    parser.add_argument(
        "--bad-case-top-k",
        type=int,
        default=5,
        help="Number of worst seam/high-lat sequences to shortlist.",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/diagnostics/seam_metric_split",
        help="Directory where diagnostic csv files will be written.",
    )
    return parser.parse_args()


def make_dataset(args, tracker_name):
    config = {
        "GT_FOLDER": str((REPO_ROOT / args.gt_folder).resolve()),
        "TRACKERS_FOLDER": str((REPO_ROOT / args.trackers_folder).resolve()),
        "TRACKERS_TO_EVAL": [tracker_name],
        "CLASSES_TO_EVAL": [args.class_name],
        "SPLIT_TO_EVAL": args.split_name,
        "TRACKER_SUB_FOLDER": args.tracker_sub_folder,
        "PRINT_CONFIG": False,
        "INPUT_AS_ZIP": False,
    }
    return JRDB2DBox(config)


def suppress_metric_stdout(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)
def full_mask_xywh(boxes_xywh):
    boxes_xywh = np.asarray(boxes_xywh, dtype=float).reshape(-1, 4)
    return np.ones((boxes_xywh.shape[0],), dtype=bool)


def remap_ids(id_list):
    unique_ids = []
    for ids_t in id_list:
        if len(ids_t) > 0:
            unique_ids.extend(np.asarray(ids_t, dtype=int).tolist())
    if not unique_ids:
        return [np.empty((0,), dtype=int) for _ in id_list], 0

    unique_ids = np.unique(unique_ids)
    remap = {int(track_id): idx for idx, track_id in enumerate(unique_ids.tolist())}
    out = []
    for ids_t in id_list:
        if len(ids_t) == 0:
            out.append(np.empty((0,), dtype=int))
            continue
        out.append(np.asarray([remap[int(track_id)] for track_id in ids_t], dtype=int))
    return out, len(unique_ids)


def filter_preprocessed_subset(data, gt_keep_masks, tracker_keep_masks):
    subset = {
        "gt_ids": [],
        "tracker_ids": [],
        "gt_dets": [],
        "tracker_dets": [],
        "tracker_confidences": [],
        "similarity_scores": [],
        "num_timesteps": data["num_timesteps"],
        "seq": data["seq"],
    }

    raw_gt_ids = []
    raw_tracker_ids = []
    num_gt_dets = 0
    num_tracker_dets = 0

    for t in range(data["num_timesteps"]):
        gt_keep = np.asarray(gt_keep_masks[t], dtype=bool)
        tracker_keep = np.asarray(tracker_keep_masks[t], dtype=bool)

        gt_ids_t = np.asarray(data["gt_ids"][t])[gt_keep]
        tracker_ids_t = np.asarray(data["tracker_ids"][t])[tracker_keep]
        gt_dets_t = np.asarray(data["gt_dets"][t])[gt_keep]
        tracker_dets_t = np.asarray(data["tracker_dets"][t])[tracker_keep]
        tracker_confidences_t = np.asarray(data["tracker_confidences"][t])[tracker_keep]
        similarity_t = np.asarray(data["similarity_scores"][t])[gt_keep][:, tracker_keep]

        raw_gt_ids.append(gt_ids_t)
        raw_tracker_ids.append(tracker_ids_t)
        subset["gt_dets"].append(gt_dets_t)
        subset["tracker_dets"].append(tracker_dets_t)
        subset["tracker_confidences"].append(tracker_confidences_t)
        subset["similarity_scores"].append(similarity_t)
        num_gt_dets += int(len(gt_ids_t))
        num_tracker_dets += int(len(tracker_ids_t))

    subset["gt_ids"], subset["num_gt_ids"] = remap_ids(raw_gt_ids)
    subset["tracker_ids"], subset["num_tracker_ids"] = remap_ids(raw_tracker_ids)
    subset["num_gt_dets"] = num_gt_dets
    subset["num_tracker_dets"] = num_tracker_dets
    return subset


def build_subset_memberships(
    data, image_width, image_height, seam_band_px, high_lat_deg
):
    gt_seam_masks = [
        seam_mask_xywh(np.asarray(dets), image_width, seam_band_px)
        for dets in data["gt_dets"]
    ]
    tracker_seam_masks = [
        seam_mask_xywh(np.asarray(dets), image_width, seam_band_px)
        for dets in data["tracker_dets"]
    ]
    gt_high_lat_masks = [
        high_lat_mask_xywh(
            np.asarray(dets),
            image_width=image_width,
            image_height=image_height,
            high_lat_deg=high_lat_deg,
        )
        for dets in data["gt_dets"]
    ]
    tracker_high_lat_masks = [
        high_lat_mask_xywh(
            np.asarray(dets),
            image_width=image_width,
            image_height=image_height,
            high_lat_deg=high_lat_deg,
        )
        for dets in data["tracker_dets"]
    ]

    return {
        "full": {
            "gt": [full_mask_xywh(dets) for dets in data["gt_dets"]],
            "tracker": [full_mask_xywh(dets) for dets in data["tracker_dets"]],
        },
        "seam": {"gt": gt_seam_masks, "tracker": tracker_seam_masks},
        "non_seam": {
            "gt": [~mask for mask in gt_seam_masks],
            "tracker": [~mask for mask in tracker_seam_masks],
        },
        "high_lat": {"gt": gt_high_lat_masks, "tracker": tracker_high_lat_masks},
        "seam_high_lat": {
            "gt": [seam & high_lat for seam, high_lat in zip(gt_seam_masks, gt_high_lat_masks)],
            "tracker": [
                seam & high_lat
                for seam, high_lat in zip(tracker_seam_masks, tracker_high_lat_masks)
            ],
        },
    }


def build_subset_data(data, subset_name, subset_memberships):
    if subset_name == "full":
        return deepcopy(data)
    if subset_name in subset_memberships:
        return filter_preprocessed_subset(
            data,
            subset_memberships[subset_name]["gt"],
            subset_memberships[subset_name]["tracker"],
        )
    raise ValueError(f"Unsupported subset_name={subset_name!r}")


def clear_matches(data, threshold=0.5):
    prev_tracker_id = np.nan * np.zeros(data["num_gt_ids"])
    prev_timestep_tracker_id = np.nan * np.zeros(data["num_gt_ids"])
    matched_pairs = []
    eps = np.finfo("float").eps

    for frame_idx, (gt_ids_t, tracker_ids_t) in enumerate(
        zip(data["gt_ids"], data["tracker_ids"])
    ):
        gt_ids_t = np.asarray(gt_ids_t, dtype=int)
        tracker_ids_t = np.asarray(tracker_ids_t, dtype=int)
        if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
            continue

        similarity = np.asarray(data["similarity_scores"][frame_idx], dtype=float)
        score_mat = (
            tracker_ids_t[np.newaxis, :]
            == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]]
        )
        score_mat = 1000 * score_mat + similarity
        score_mat[similarity < threshold - eps] = 0

        match_rows, match_cols = linear_sum_assignment(-score_mat)
        actually_matched_mask = score_mat[match_rows, match_cols] > 0 + eps
        match_rows = match_rows[actually_matched_mask]
        match_cols = match_cols[actually_matched_mask]

        if len(match_rows) == 0:
            continue

        matched_gt_ids = gt_ids_t[match_rows]
        matched_tracker_ids = tracker_ids_t[match_cols]
        matched_pairs.append(
            {
                "frame": int(frame_idx),
                "gt_local_indices": match_rows.astype(int),
                "tracker_local_indices": match_cols.astype(int),
                "gt_ids": matched_gt_ids.astype(int),
                "tracker_ids": matched_tracker_ids.astype(int),
            }
        )

        prev_tracker_id[matched_gt_ids] = matched_tracker_ids
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids

    return matched_pairs


def leakage_bucket(rate):
    for upper_bound, label in LEAKAGE_RULES:
        if rate <= upper_bound:
            return label
    raise AssertionError("Unreachable leakage bucket.")


def compute_membership_leakage(full_data, gt_seam_masks, tracker_seam_masks):
    matched_pairs = clear_matches(full_data)
    total_matches = 0
    cross_subset_matches = 0
    for match in matched_pairs:
        frame_idx = match["frame"]
        gt_flags = gt_seam_masks[frame_idx][match["gt_local_indices"]]
        tracker_flags = tracker_seam_masks[frame_idx][match["tracker_local_indices"]]
        total_matches += int(len(gt_flags))
        cross_subset_matches += int(np.count_nonzero(gt_flags != tracker_flags))

    leakage = (
        float(cross_subset_matches) / float(total_matches) if total_matches > 0 else 0.0
    )
    return {
        "matched_pairs": int(total_matches),
        "cross_subset_matches": int(cross_subset_matches),
        "membership_leakage": leakage,
        "interpretation": leakage_bucket(leakage),
    }


def write_rows_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def combined_coverage_rows(coverage_rows, tracker_name):
    combined_rows = []
    full_rows = [
        row
        for row in coverage_rows
        if row["tracker"] == tracker_name and row["subset"] == "full"
    ]
    full_coverage = {
        "tracker": tracker_name,
        "subset": "full",
        "seq": "COMBINED",
        "gt_dets": int(sum(row["gt_dets"] for row in full_rows)),
        "tracker_dets": int(sum(row["tracker_dets"] for row in full_rows)),
        "gt_ids": int(sum(row["gt_ids"] for row in full_rows)),
        "tracker_ids": int(sum(row["tracker_ids"] for row in full_rows)),
        "gt_fraction_of_full": 1.0,
        "tracker_fraction_of_full": 1.0,
    }
    combined_rows.append(full_coverage)

    for subset_name in SUBSET_NAMES:
        if subset_name == "full":
            continue
        subset_rows = [
            row
            for row in coverage_rows
            if row["tracker"] == tracker_name and row["subset"] == subset_name
        ]
        combined = {
            "tracker": tracker_name,
            "subset": subset_name,
            "seq": "COMBINED",
            "gt_dets": int(sum(row["gt_dets"] for row in subset_rows)),
            "tracker_dets": int(sum(row["tracker_dets"] for row in subset_rows)),
            "gt_ids": int(sum(row["gt_ids"] for row in subset_rows)),
            "tracker_ids": int(sum(row["tracker_ids"] for row in subset_rows)),
        }
        combined["gt_fraction_of_full"] = (
            float(combined["gt_dets"]) / float(full_coverage["gt_dets"])
            if full_coverage["gt_dets"] > 0
            else 0.0
        )
        combined["tracker_fraction_of_full"] = (
            float(combined["tracker_dets"]) / float(full_coverage["tracker_dets"])
            if full_coverage["tracker_dets"] > 0
            else 0.0
        )
        combined_rows.append(combined)
    return combined_rows


def rank_bad_case_sequences(per_sequence_metric_rows, coverage_rows, tracker_name, top_k):
    seq_rows = {}
    for row in per_sequence_metric_rows:
        if row["tracker"] != tracker_name or row["seq"] == "COMBINED":
            continue
        seq_entry = seq_rows.setdefault(row["seq"], {"seq": row["seq"]})
        subset_name = row["subset"]
        seq_entry[f"{subset_name}_HOTA"] = float(row["HOTA"])
        seq_entry[f"{subset_name}_AssA"] = float(row["AssA"])
        seq_entry[f"{subset_name}_IDF1"] = float(row["IDF1"])

    for row in coverage_rows:
        if row["tracker"] != tracker_name or row["seq"] == "COMBINED":
            continue
        seq_entry = seq_rows.setdefault(row["seq"], {"seq": row["seq"]})
        subset_name = row["subset"]
        seq_entry[f"{subset_name}_gt_dets"] = int(row["gt_dets"])
        seq_entry[f"{subset_name}_tracker_dets"] = int(row["tracker_dets"])

    ranking = []
    for seq_name, row in seq_rows.items():
        subset_scores = []
        active_subset_count = 0
        for subset_name in ("seam", "high_lat"):
            gt_dets = int(row.get(f"{subset_name}_gt_dets", 0))
            score = row.get(f"{subset_name}_HOTA")
            if gt_dets > 0 and score is not None:
                subset_scores.append(float(score))
                active_subset_count += 1
        if active_subset_count == 0:
            continue
        ranking.append(
            {
                **row,
                "seq": seq_name,
                "bad_case_score": float(np.mean(subset_scores)),
                "active_subset_count": active_subset_count,
            }
        )

    ranking.sort(
        key=lambda row: (
            float(row["bad_case_score"]),
            -int(row["active_subset_count"]),
            row["seq"],
        )
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
        row["selected"] = rank <= int(top_k)
    return ranking


def evaluate_metric_bundle(data):
    hota = HOTA()
    clear = CLEAR({"PRINT_CONFIG": False})
    identity = Identity({"PRINT_CONFIG": False})
    return {
        "HOTA": suppress_metric_stdout(hota.eval_sequence, data),
        "CLEAR": suppress_metric_stdout(clear.eval_sequence, data),
        "Identity": suppress_metric_stdout(identity.eval_sequence, data),
    }


def combine_metric_bundle(per_sequence_bundle):
    hota = HOTA()
    clear = CLEAR({"PRINT_CONFIG": False})
    identity = Identity({"PRINT_CONFIG": False})
    return {
        "HOTA": suppress_metric_stdout(
            hota.combine_sequences,
            {seq: metrics["HOTA"] for seq, metrics in per_sequence_bundle.items()},
        ),
        "CLEAR": suppress_metric_stdout(
            clear.combine_sequences,
            {seq: metrics["CLEAR"] for seq, metrics in per_sequence_bundle.items()},
        ),
        "Identity": suppress_metric_stdout(
            identity.combine_sequences,
            {seq: metrics["Identity"] for seq, metrics in per_sequence_bundle.items()},
        ),
    }


def summarise_metric_bundle(metric_bundle):
    return {
        "HOTA": float(np.mean(metric_bundle["HOTA"]["HOTA"]) * 100.0),
        "DetA": float(np.mean(metric_bundle["HOTA"]["DetA"]) * 100.0),
        "AssA": float(np.mean(metric_bundle["HOTA"]["AssA"]) * 100.0),
        "IDF1": float(metric_bundle["Identity"]["IDF1"] * 100.0),
        "FP": int(metric_bundle["CLEAR"]["CLR_FP"]),
        "IDSW": int(metric_bundle["CLEAR"]["IDSW"]),
    }


def validate_full_against_summary(tracker_dir, full_summary):
    summary_path = tracker_dir / "pedestrian_summary.csv"
    with summary_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row["seq"] == "COMBINED"]
    if len(rows) != 1:
        raise ValueError(f"Unexpected summary format in {summary_path}.")

    expected = rows[0]
    for key in ("HOTA", "DetA", "AssA", "IDF1"):
        if round(full_summary[key], 3) != round(float(expected[key]), 3):
            raise AssertionError(
                f"Full split mismatch for {key}: computed={full_summary[key]:.6f}, "
                f"summary={float(expected[key]):.6f}"
            )
    for key, summary_key in (("FP", "CLR_FP"), ("IDSW", "IDSW")):
        if int(full_summary[key]) != int(expected[summary_key]):
            raise AssertionError(
                f"Full split mismatch for {key}: computed={full_summary[key]}, "
                f"summary={expected[summary_key]}"
            )


def main():
    args = parse_args()
    out_root = (REPO_ROOT / args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Running seam-conditioned split for trackers={args.tracker_names}, "
        f"band={args.seam_band_px}, image_width={args.image_width}, "
        f"high_lat_deg={args.high_lat_deg}"
    )

    for tracker_name in args.tracker_names:
        dataset = make_dataset(args, tracker_name)
        tracker_dir = (REPO_ROOT / args.trackers_folder / tracker_name).resolve()
        tracker_out_dir = out_root / tracker_name
        tracker_out_dir.mkdir(parents=True, exist_ok=True)

        per_sequence_metric_rows = []
        coverage_rows = []
        leakage_rows = []
        per_subset_seq_metrics = {subset_name: {} for subset_name in SUBSET_NAMES}

        for seq in dataset.seq_list:
            raw_data = dataset.get_raw_seq_data(tracker_name, seq, is_3d=False)
            preprocessed = dataset.get_preprocessed_seq_data(raw_data, args.class_name)

            subset_cache = {}
            subset_memberships = build_subset_memberships(
                preprocessed,
                image_width=args.image_width,
                image_height=args.image_height,
                seam_band_px=args.seam_band_px,
                high_lat_deg=args.high_lat_deg,
            )
            for subset_name in SUBSET_NAMES:
                subset_data = build_subset_data(
                    preprocessed,
                    subset_name,
                    subset_memberships,
                )
                subset_cache[subset_name] = subset_data
                per_subset_seq_metrics[subset_name][seq] = evaluate_metric_bundle(
                    subset_data
                )
                summary = summarise_metric_bundle(per_subset_seq_metrics[subset_name][seq])
                per_sequence_metric_rows.append(
                    {
                        "tracker": tracker_name,
                        "subset": subset_name,
                        "seq": seq,
                        **summary,
                    }
                )

            full_gt = int(preprocessed["num_gt_dets"])
            full_tracker = int(preprocessed["num_tracker_dets"])
            seam_gt = int(subset_cache["seam"]["num_gt_dets"])
            seam_tracker = int(subset_cache["seam"]["num_tracker_dets"])
            non_seam_gt = int(subset_cache["non_seam"]["num_gt_dets"])
            non_seam_tracker = int(subset_cache["non_seam"]["num_tracker_dets"])

            if seam_gt + non_seam_gt != full_gt:
                raise AssertionError(
                    f"GT detection conservation failed for {tracker_name}/{seq}: "
                    f"{seam_gt} + {non_seam_gt} != {full_gt}"
                )
            if seam_tracker + non_seam_tracker != full_tracker:
                raise AssertionError(
                    f"Tracker detection conservation failed for {tracker_name}/{seq}: "
                    f"{seam_tracker} + {non_seam_tracker} != {full_tracker}"
                )

            for subset_name in SUBSET_NAMES:
                subset_data = subset_cache[subset_name]
                coverage_rows.append(
                    {
                        "tracker": tracker_name,
                        "subset": subset_name,
                        "seq": seq,
                        "gt_dets": int(subset_data["num_gt_dets"]),
                        "tracker_dets": int(subset_data["num_tracker_dets"]),
                        "gt_ids": int(subset_data["num_gt_ids"]),
                        "tracker_ids": int(subset_data["num_tracker_ids"]),
                        "gt_fraction_of_full": (
                            float(subset_data["num_gt_dets"]) / float(full_gt)
                            if full_gt > 0
                            else 0.0
                        ),
                        "tracker_fraction_of_full": (
                            float(subset_data["num_tracker_dets"]) / float(full_tracker)
                            if full_tracker > 0
                            else 0.0
                        ),
                    }
                )

            for subset_name in ("seam", "high_lat", "seam_high_lat"):
                leakage_rows.append(
                    {
                        "tracker": tracker_name,
                        "scope": "per_sequence",
                        "subset": subset_name,
                        "seq": seq,
                        **compute_membership_leakage(
                            preprocessed,
                            subset_memberships[subset_name]["gt"],
                            subset_memberships[subset_name]["tracker"],
                        ),
                    }
                )

        combined_metric_rows = []
        for subset_name in SUBSET_NAMES:
            combined = combine_metric_bundle(per_subset_seq_metrics[subset_name])
            summary = summarise_metric_bundle(combined)
            combined_metric_rows.append(
                {
                    "tracker": tracker_name,
                    "subset": subset_name,
                    "seq": "COMBINED",
                    **summary,
                }
            )

        validate_full_against_summary(tracker_dir, combined_metric_rows[0])

        for subset_name in ("seam", "high_lat", "seam_high_lat"):
            subset_rows = [
                row
                for row in leakage_rows
                if row["tracker"] == tracker_name and row["subset"] == subset_name
            ]
            total_matched_pairs = int(sum(row["matched_pairs"] for row in subset_rows))
            total_cross_matches = int(
                sum(row["cross_subset_matches"] for row in subset_rows)
            )
            combined_leakage = (
                float(total_cross_matches) / float(total_matched_pairs)
                if total_matched_pairs > 0
                else 0.0
            )
            leakage_rows.append(
                {
                    "tracker": tracker_name,
                    "scope": "combined",
                    "subset": subset_name,
                    "seq": "COMBINED",
                    "matched_pairs": total_matched_pairs,
                    "cross_subset_matches": total_cross_matches,
                    "membership_leakage": combined_leakage,
                    "interpretation": leakage_bucket(combined_leakage),
                }
            )

        tracker_coverage_rows = [
            row for row in coverage_rows if row["tracker"] == tracker_name
        ]
        combined_coverage = combined_coverage_rows(coverage_rows, tracker_name)
        bad_case_ranking = rank_bad_case_sequences(
            per_sequence_metric_rows=per_sequence_metric_rows,
            coverage_rows=coverage_rows,
            tracker_name=tracker_name,
            top_k=args.bad_case_top_k,
        )

        write_rows_csv(tracker_out_dir / "combined_metrics.csv", combined_metric_rows)
        write_rows_csv(
            tracker_out_dir / "per_sequence_metrics.csv",
            [row for row in per_sequence_metric_rows if row["tracker"] == tracker_name],
        )
        write_rows_csv(
            tracker_out_dir / "coverage.csv",
            [*tracker_coverage_rows, *combined_coverage],
        )
        write_rows_csv(
            tracker_out_dir / "membership_leakage.csv",
            [row for row in leakage_rows if row["tracker"] == tracker_name],
        )
        write_rows_csv(
            tracker_out_dir / "bad_case_candidates.csv",
            bad_case_ranking,
        )

        seam_combined = next(
            row
            for row in leakage_rows
            if row["tracker"] == tracker_name
            and row["scope"] == "combined"
            and row["subset"] == "seam"
        )
        print(
            f"[{tracker_name}] wrote diagnostics to {tracker_out_dir}. "
            f"Full HOTA={combined_metric_rows[0]['HOTA']:.3f}, "
            f"seam leakage={seam_combined['membership_leakage']:.4%} "
            f"({seam_combined['interpretation']})"
        )


if __name__ == "__main__":
    main()
