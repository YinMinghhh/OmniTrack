import argparse
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity checks for OmniTrack JRDB evaluation pipeline."
    )
    parser.add_argument("--pred-json", default=None, help="Path to results_jrdb2d.json")
    parser.add_argument("--workspace", default=None, help="Path to evaluation workspace")
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


def check_pred_json(pred_json_path, errors, warnings):
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

    for token, objs in frames.items():
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
        for obj in objs:
            tid = obj.get("tracking_id", None)
            if tid is None or tid == "None":
                none_id_count += 1

    print(f"[SANITY] JSON frames={frame_count}, objects={obj_count}, none_id={none_id_count}")
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
                    w = float(cols[8])
                    h = float(cols[9])
                except ValueError:
                    bad_cols += 1
                    continue
                parsed += 1
                if w <= 0 or h <= 0:
                    non_positive_wh += 1
                if w > warn_width_threshold:
                    very_large_w += 1

    return {
        "parsed": parsed,
        "bad_cols": bad_cols,
        "non_positive_wh": non_positive_wh,
        "very_large_w": very_large_w,
        "per_seq_lines": per_seq_lines,
        "empty_files": empty_files,
    }


def check_workspace(workspace, image_width, warn_width_threshold, errors, warnings):
    gt_root = os.path.join(workspace, "gt")
    pred_root = os.path.join(workspace, "pred", "JRDB-train")
    label_dir = os.path.join(gt_root, "label_02")
    seqmap_path = os.path.join(gt_root, "evaluate_tracking.seqmap.train")

    print(f"[SANITY] Checking workspace: {workspace}")
    required_dirs = [gt_root, label_dir, pred_root]
    for p in required_dirs:
        if not os.path.isdir(p):
            errors.append(f"Missing required directory: {p}")
    if not os.path.isfile(seqmap_path):
        errors.append(f"Missing seqmap: {seqmap_path}")
    if errors:
        return

    seqmap_rows = _read_seqmap(seqmap_path)
    seqmap_seqs = {r[0] for r in seqmap_rows}
    gt_seqs = {os.path.splitext(x)[0] for x in os.listdir(label_dir) if x.endswith(".txt")}
    pred_seqs = {os.path.splitext(x)[0] for x in os.listdir(pred_root) if x.endswith(".txt")}

    print(f"[SANITY] seqmap={len(seqmap_seqs)}, gt={len(gt_seqs)}, pred={len(pred_seqs)}")

    missing_in_pred = sorted(seqmap_seqs - pred_seqs)
    missing_in_gt = sorted(seqmap_seqs - gt_seqs)
    extra_pred = sorted(pred_seqs - seqmap_seqs)

    if missing_in_pred:
        errors.append(f"{len(missing_in_pred)} sequences in seqmap are missing in predictions.")
    if missing_in_gt:
        errors.append(f"{len(missing_in_gt)} sequences in seqmap are missing in GT label_02.")
    if extra_pred:
        warnings.append(f"{len(extra_pred)} prediction sequences are not listed in seqmap.")

    pred_sample = _sample_boxes_from_txt_dir(pred_root, warn_width_threshold=warn_width_threshold)
    gt_sample = _sample_boxes_from_txt_dir(label_dir, warn_width_threshold=warn_width_threshold)

    parsed = pred_sample["parsed"]
    print(
        f"[SANITY] parsed_pred_boxes={parsed}, malformed_pred_lines={pred_sample['bad_cols']}, "
        f"parsed_gt_boxes={gt_sample['parsed']}, malformed_gt_lines={gt_sample['bad_cols']}"
    )

    if parsed == 0:
        errors.append("No parseable prediction boxes found in workspace/pred/JRDB-train.")
        return
    if pred_sample["bad_cols"] > 0:
        warnings.append(f"{pred_sample['bad_cols']} prediction lines have malformed columns.")
    if gt_sample["bad_cols"] > 0:
        warnings.append(f"{gt_sample['bad_cols']} GT lines have malformed columns.")
    if pred_sample["non_positive_wh"] > 0:
        warnings.append(f"{pred_sample['non_positive_wh']} prediction boxes have non-positive width/height.")
    if gt_sample["non_positive_wh"] > 0:
        warnings.append(f"{gt_sample['non_positive_wh']} GT boxes have non-positive width/height.")

    pred_large_ratio = pred_sample["very_large_w"] / parsed
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


def main():
    args = parse_args()
    errors = []
    warnings = []

    if args.pred_json:
        check_pred_json(args.pred_json, errors, warnings)
    if args.workspace:
        check_workspace(
            args.workspace,
            image_width=args.image_width,
            warn_width_threshold=args.warn_width_threshold,
            errors=errors,
            warnings=warnings,
        )

    if not args.pred_json and not args.workspace:
        errors.append("Nothing to check. Use --pred-json and/or --workspace.")

    for w in warnings:
        print(f"[WARN] {w}")
    for e in errors:
        print(f"[ERROR] {e}")

    if errors:
        print(f"[SANITY] FAILED with {len(errors)} error(s), {len(warnings)} warning(s).")
        raise SystemExit(1)
    print(f"[SANITY] PASSED with {len(warnings)} warning(s).")


if __name__ == "__main__":
    main()
