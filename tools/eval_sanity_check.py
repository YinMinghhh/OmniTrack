import argparse
import csv
import json
import os
from collections import defaultdict

try:
    from tools.jrdb_eval_splits import (
        DEFAULT_GT_SOURCE_DIR,
        detect_split_from_pred_json,
        extract_sequences_from_pred_json,
        get_split_sequences,
    )
except ImportError:
    from jrdb_eval_splits import (
        DEFAULT_GT_SOURCE_DIR,
        detect_split_from_pred_json,
        extract_sequences_from_pred_json,
        get_split_sequences,
    )


TRACKER_NAME = "JRDB-train"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity checks for OmniTrack JRDB evaluation pipeline."
    )
    parser.add_argument("--pred-json", default=None, help="Path to results_jrdb2d.json")
    parser.add_argument("--workspace", default=None, help="Path to evaluation workspace")
    parser.add_argument(
        "--gt-source-dir",
        default=DEFAULT_GT_SOURCE_DIR,
        help="Directory containing JRDB stitched GT json files.",
    )
    parser.add_argument(
        "--image-width",
        type=float,
        default=3760.0,
        help="Reference stitched-image width used for heuristic warnings.",
    )
    parser.add_argument(
        "--warn-width-threshold",
        type=float,
        default=1200.0,
        help="Warn if too many boxes have width larger than this value.",
    )
    return parser.parse_args()


def _read_seqmap(seqmap_path):
    seq_rows = []
    with open(seqmap_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < 4:
                continue
            seq_rows.append((cols[0], cols[1], cols[2], cols[3]))
    return seq_rows


def _read_eval_split(gt_root):
    split_path = os.path.join(gt_root, "eval_split.txt")
    if not os.path.isfile(split_path):
        raise FileNotFoundError("Missing eval split marker: %s" % split_path)
    with open(split_path, "r", encoding="utf-8") as f:
        value = f.read().strip()
    if value not in {"train", "val", "all"}:
        raise ValueError("Unsupported eval split marker value: %r" % value)
    return value


def _read_combined_summary(summary_path):
    if not os.path.isfile(summary_path):
        return None

    with open(summary_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("seq") == "COMBINED":
                return {
                    "HOTA": row.get("HOTA", ""),
                    "MOTA": row.get("MOTA", ""),
                    "IDF1": row.get("IDF1", ""),
                }
    return None


def check_pred_json(pred_json_path, gt_source_dir, errors, warnings):
    print(f"[SANITY] Checking prediction JSON: {pred_json_path}")
    if not os.path.isfile(pred_json_path):
        errors.append(f"Prediction JSON not found: {pred_json_path}")
        return

    with open(pred_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "results" not in data or not isinstance(data["results"], dict):
        errors.append("Prediction JSON must contain dict key 'results'.")
        return

    frames = data["results"]
    frame_count = len(frames)
    obj_count = 0
    invalid_token_count = 0
    none_id_count = 0
    seq_stats = defaultdict(
        lambda: {
            "frames": 0,
            "objects": 0,
            "none_id": 0,
        }
    )

    for token, objs in frames.items():
        seq_name = token.rsplit("_", 1)[0] if "_" in token else "__invalid__"
        seq_stats[seq_name]["frames"] += 1
        if "_" not in token:
            invalid_token_count += 1
        else:
            frame = token.rsplit("_", 1)[-1]
            if not frame.isdigit():
                invalid_token_count += 1

        if not isinstance(objs, list):
            warnings.append(f"Frame {token} has non-list predictions.")
            continue
        obj_count += len(objs)
        seq_stats[seq_name]["objects"] += len(objs)
        for obj in objs:
            tid = obj.get("tracking_id", None)
            if tid is None or tid == "None":
                none_id_count += 1
                seq_stats[seq_name]["none_id"] += 1

    print(f"[SANITY] JSON frames={frame_count}, objects={obj_count}, none_id={none_id_count}")
    for seq_name in sorted(seq_stats.keys()):
        stats = seq_stats[seq_name]
        none_ratio = 0.0 if stats["objects"] == 0 else (100.0 * stats["none_id"] / stats["objects"])
        print(
            "[IDCHAIN][SANITY][SEQ] "
            f"{seq_name} "
            f"frames={stats['frames']} "
            f"objects={stats['objects']} "
            f"none_id={stats['none_id']} "
            f"none_ratio={none_ratio:.2f}%"
        )

    try:
        pred_sequences = extract_sequences_from_pred_json(pred_json_path)
    except ValueError as exc:
        errors.append(str(exc))
        pred_sequences = []

    try:
        resolved_split = detect_split_from_pred_json(pred_json_path, gt_source_dir)
        target_sequences = get_split_sequences(resolved_split, gt_source_dir)
        missing_pred = sorted(set(target_sequences) - set(pred_sequences))
        extra_pred = sorted(set(pred_sequences) - set(target_sequences))
        print(
            "[SANITY][SPLIT] "
            f"target_split={resolved_split} "
            f"target_seq_count={len(target_sequences)} "
            f"pred_seq_count={len(pred_sequences)} "
            f"missing_pred_seq_count={len(missing_pred)} "
            f"extra_pred_seq_count={len(extra_pred)}"
        )
    except ValueError as exc:
        errors.append(str(exc))
        print(
            "[SANITY][SPLIT] "
            f"target_split=unmatched "
            f"target_seq_count=0 "
            f"pred_seq_count={len(pred_sequences)} "
            f"missing_pred_seq_count=0 "
            f"extra_pred_seq_count=0"
        )

    if frame_count == 0:
        errors.append("Prediction JSON has zero frames in 'results'.")
    if obj_count == 0:
        errors.append("Prediction JSON has zero objects.")
    if invalid_token_count > 0:
        warnings.append(f"{invalid_token_count} frame tokens do not match '<seq>_<frame>' pattern.")


def _sample_boxes_from_txt_dir(txt_dir, warn_width_threshold):
    parsed = 0
    bad_cols = 0
    non_positive_wh = 0
    very_large_w = 0
    per_seq_lines = defaultdict(int)

    empty_files = 0
    for fname in sorted(os.listdir(txt_dir)):
        if not fname.endswith(".txt"):
            continue
        seq = os.path.splitext(fname)[0]
        path = os.path.join(txt_dir, fname)
        if os.path.getsize(path) == 0:
            empty_files += 1
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                cols = line.split()
                per_seq_lines[seq] += 1
                if len(cols) < 10:
                    bad_cols += 1
                    continue
                try:
                    width = float(cols[8])
                    height = float(cols[9])
                except ValueError:
                    bad_cols += 1
                    continue
                parsed += 1
                if width <= 0 or height <= 0:
                    non_positive_wh += 1
                if width > warn_width_threshold:
                    very_large_w += 1

    return {
        "parsed": parsed,
        "bad_cols": bad_cols,
        "non_positive_wh": non_positive_wh,
        "very_large_w": very_large_w,
        "per_seq_lines": per_seq_lines,
        "empty_files": empty_files,
    }


def check_workspace(workspace, gt_source_dir, image_width, warn_width_threshold, errors, warnings):
    gt_root = os.path.join(workspace, "gt")
    pred_root = os.path.join(workspace, "pred", TRACKER_NAME)
    label_dir = os.path.join(gt_root, "label_02")

    print(f"[SANITY] Checking workspace: {workspace}")
    required_dirs = [gt_root, label_dir, pred_root]
    for path in required_dirs:
        if not os.path.isdir(path):
            errors.append(f"Missing required directory: {path}")
    if errors:
        return

    try:
        resolved_split = _read_eval_split(gt_root)
    except (FileNotFoundError, ValueError) as exc:
        errors.append(str(exc))
        return

    seqmap_path = os.path.join(gt_root, f"evaluate_tracking.seqmap.{resolved_split}")
    if not os.path.isfile(seqmap_path):
        errors.append(f"Missing seqmap: {seqmap_path}")
        return

    seqmap_rows = _read_seqmap(seqmap_path)
    seqmap_seqs = {row[0] for row in seqmap_rows}
    gt_seqs = {os.path.splitext(name)[0] for name in os.listdir(label_dir) if name.endswith(".txt")}
    pred_seqs = {os.path.splitext(name)[0] for name in os.listdir(pred_root) if name.endswith(".txt")}
    canonical_seqs = set(get_split_sequences(resolved_split, gt_source_dir))

    missing_in_pred = sorted(seqmap_seqs - pred_seqs)
    missing_in_gt = sorted(seqmap_seqs - gt_seqs)
    extra_pred = sorted(pred_seqs - seqmap_seqs)
    missing_in_seqmap = sorted(canonical_seqs - seqmap_seqs)
    extra_in_seqmap = sorted(seqmap_seqs - canonical_seqs)

    pred_sample = _sample_boxes_from_txt_dir(pred_root, warn_width_threshold=warn_width_threshold)
    gt_sample = _sample_boxes_from_txt_dir(label_dir, warn_width_threshold=warn_width_threshold)

    print(
        "[SANITY][WORKSPACE] "
        f"target_split={resolved_split} "
        f"seqmap_seq_count={len(seqmap_seqs)} "
        f"pred_txt_seq_count={len(pred_seqs)} "
        f"empty_pred_seq_count={pred_sample['empty_files']} "
        f"missing_pred_seq_count={len(missing_in_pred)} "
        f"extra_pred_seq_count={len(extra_pred)}"
    )
    print(
        f"[SANITY] parsed_pred_boxes={pred_sample['parsed']}, malformed_pred_lines={pred_sample['bad_cols']}, "
        f"parsed_gt_boxes={gt_sample['parsed']}, malformed_gt_lines={gt_sample['bad_cols']}"
    )

    if missing_in_seqmap or extra_in_seqmap:
        errors.append(
            "Seqmap does not match canonical split %s (missing=%d, extra=%d)."
            % (resolved_split, len(missing_in_seqmap), len(extra_in_seqmap))
        )
    if missing_in_pred:
        errors.append(f"{len(missing_in_pred)} sequences in seqmap are missing in predictions.")
    if missing_in_gt:
        errors.append(f"{len(missing_in_gt)} sequences in seqmap are missing in GT label_02.")
    if extra_pred:
        warnings.append(f"{len(extra_pred)} prediction sequences are not listed in seqmap.")

    if pred_sample["parsed"] == 0:
        errors.append(f"No parseable prediction boxes found in {pred_root}.")
        return
    if pred_sample["bad_cols"] > 0:
        warnings.append(f"{pred_sample['bad_cols']} prediction lines have malformed columns.")
    if gt_sample["bad_cols"] > 0:
        warnings.append(f"{gt_sample['bad_cols']} GT lines have malformed columns.")
    if pred_sample["non_positive_wh"] > 0:
        warnings.append(f"{pred_sample['non_positive_wh']} prediction boxes have non-positive width/height.")
    if gt_sample["non_positive_wh"] > 0:
        warnings.append(f"{gt_sample['non_positive_wh']} GT boxes have non-positive width/height.")

    pred_large_ratio = pred_sample["very_large_w"] / pred_sample["parsed"]
    if pred_large_ratio > 0.1:
        warnings.append(
            "More than 10% prediction boxes have very large width. "
            "Check whether converter writes xywh or x1y1x2y2."
        )
    elif pred_sample["very_large_w"] > 0:
        warnings.append(
            f"{pred_sample['very_large_w']} prediction boxes exceed width threshold={warn_width_threshold}."
        )

    if gt_sample["parsed"] > 0:
        gt_large_ratio = gt_sample["very_large_w"] / gt_sample["parsed"]
        if gt_large_ratio > 0.1:
            warnings.append(
                "More than 10% GT boxes have very large width. "
                "Check GT formatter box semantics."
            )

    if pred_sample["empty_files"] > 0:
        warnings.append(f"{pred_sample['empty_files']} prediction sequence files are empty.")
    if pred_sample["empty_files"] == len(pred_seqs) and pred_seqs:
        errors.append("All prediction sequence files are empty.")

    for seq, _, start, end in seqmap_rows[:5]:
        print(f"[SANITY] seqmap sample: {seq} start={start} end={end}")

    if warn_width_threshold > image_width:
        warnings.append(
            "warn_width_threshold is larger than image_width; width warnings may be less sensitive."
        )

    summary_metrics = _read_combined_summary(os.path.join(pred_root, "pedestrian_summary.csv"))
    if summary_metrics is not None:
        print(
            "[SANITY][COMBINED] "
            f"HOTA={summary_metrics['HOTA']} "
            f"MOTA={summary_metrics['MOTA']} "
            f"IDF1={summary_metrics['IDF1']}"
        )


def main():
    args = parse_args()
    errors = []
    warnings = []

    if args.pred_json:
        check_pred_json(args.pred_json, args.gt_source_dir, errors, warnings)
    if args.workspace:
        check_workspace(
            args.workspace,
            gt_source_dir=args.gt_source_dir,
            image_width=args.image_width,
            warn_width_threshold=args.warn_width_threshold,
            errors=errors,
            warnings=warnings,
        )

    if not args.pred_json and not args.workspace:
        errors.append("Nothing to check. Use --pred-json and/or --workspace.")

    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[ERROR] {error}")

    if errors:
        print(f"[SANITY] FAILED with {len(errors)} error(s), {len(warnings)} warning(s).")
        raise SystemExit(1)
    print(f"[SANITY] PASSED with {len(warnings)} warning(s).")


if __name__ == "__main__":
    main()
