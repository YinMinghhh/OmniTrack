import argparse
import glob
import json
import os

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--box_format",
        default="xywh",
        choices=["xywh", "xyxy"],
        help="BBox format written to columns 6:10 in GT txt.",
    )
    parser.add_argument(
        "--split",
        default="auto",
        choices=["auto", "train", "val", "all"],
        help="Target evaluation split. auto detects split from prediction JSON.",
    )
    parser.add_argument(
        "--pred-json",
        default=None,
        help="Path to results_jrdb2d.json. Required when --split auto.",
    )
    parser.add_argument(
        "--gt-source-dir",
        default=DEFAULT_GT_SOURCE_DIR,
        help="Directory containing JRDB stitched GT json files.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def collect_txt_sequences(txt_dir):
    if not os.path.isdir(txt_dir):
        return []
    sequences = [
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(os.path.join(txt_dir, "*.txt"))
    ]
    sequences.sort()
    return sequences


def resolve_split(split, pred_json_path, gt_source_dir):
    if split == "auto":
        if not pred_json_path:
            raise ValueError("--pred-json is required when --split auto.")
        return detect_split_from_pred_json(pred_json_path, gt_source_dir)
    return split


def load_pred_sequences(pred_json_path, pred_dir):
    if pred_json_path:
        return extract_sequences_from_pred_json(pred_json_path)
    return collect_txt_sequences(pred_dir)


def validate_pred_sequences(resolved_split, target_sequences, pred_sequences):
    target_set = set(target_sequences)
    pred_set = set(pred_sequences)
    missing = sorted(target_set - pred_set)
    extra = sorted(pred_set - target_set)
    return missing, extra


def convert_gt(workspace_gt, target_sequences, box_format, gt_source_dir):
    print("[1/3] Converting GT...")
    out_dir = os.path.join(workspace_gt, "label_02")
    ensure_dir(out_dir)

    for seq in target_sequences:
        json_f = os.path.join(gt_source_dir, "%s.json" % seq)
        if not os.path.isfile(json_f):
            raise FileNotFoundError("GT json not found for sequence %s: %s" % (seq, json_f))

        with open(json_f, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = []
        for frame, objs in data.get("labels", {}).items():
            fid = int(frame.split(".")[0])
            for obj in objs:
                label_id = obj["label_id"]
                try:
                    tid = int(label_id.split(":")[-1]) if ":" in label_id else int(label_id)
                except Exception:
                    tid = abs(hash(label_id)) % 100000
                box = obj["box"]
                left, top = float(box[0]), float(box[1])
                width, height = float(box[2]), float(box[3])
                if box_format == "xywh":
                    x3, x4 = width, height
                else:
                    x3, x4 = left + width, top + height
                lines.append(
                    f"{fid} {tid} Pedestrian 0 0 0 {left:.2f} {top:.2f} {x3:.2f} {x4:.2f} -1 -1 -1 -1000 -1000 -1000 0 1.0\n"
                )

        lines.sort(key=lambda line: int(line.split(" ")[0]))
        with open(os.path.join(out_dir, "%s.txt" % seq), "w", encoding="utf-8") as f:
            f.writelines(lines)


def generate_seqmap(workspace_gt, resolved_split, target_sequences):
    print("[2/3] Generating Seqmap...")
    gt_dir = os.path.join(workspace_gt, "label_02")
    seqmap_path = os.path.join(workspace_gt, "evaluate_tracking.seqmap.%s" % resolved_split)
    lines = []

    for seq in target_sequences:
        gt_path = os.path.join(gt_dir, "%s.txt" % seq)
        with open(gt_path, "r", encoding="utf-8") as txt:
            content = txt.readlines()
        frames = [int(line.split(" ")[0]) for line in content] if content else [0]
        lines.append(f"{seq} empty {min(frames)} {max(frames)}\n")

    lines.sort()
    with open(seqmap_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    with open(os.path.join(workspace_gt, "eval_split.txt"), "w", encoding="utf-8") as f:
        f.write(resolved_split + "\n")


def cleanup_workspace(workspace_gt, workspace_pred, target_sequences, pred_sequences):
    label_dir = os.path.join(workspace_gt, "label_02")
    ensure_dir(label_dir)
    ensure_dir(workspace_pred)

    target_set = set(target_sequences)
    pred_set = set(pred_sequences)
    removed_gt = 0
    removed_pred = 0
    removed_seqmaps = 0

    for path in glob.glob(os.path.join(label_dir, "*.txt")):
        seq = os.path.splitext(os.path.basename(path))[0]
        if seq not in target_set:
            os.remove(path)
            removed_gt += 1

    for path in glob.glob(os.path.join(workspace_pred, "*.txt")):
        seq = os.path.splitext(os.path.basename(path))[0]
        if seq not in target_set or seq not in pred_set:
            os.remove(path)
            removed_pred += 1

    for path in glob.glob(os.path.join(workspace_gt, "evaluate_tracking.seqmap.*")):
        os.remove(path)
        removed_seqmaps += 1

    eval_split_path = os.path.join(workspace_gt, "eval_split.txt")
    if os.path.isfile(eval_split_path):
        os.remove(eval_split_path)

    print(
        "[EVAL][CLEANUP] removed_gt_txt=%d removed_pred_txt=%d removed_seqmaps=%d"
        % (removed_gt, removed_pred, removed_seqmaps)
    )


def fill_missing(workspace_pred, target_sequences):
    print("[3/3] Filling missing predictions...")
    existing_pred = set(collect_txt_sequences(workspace_pred))
    padded = []

    for seq in target_sequences:
        if seq in existing_pred:
            continue
        with open(os.path.join(workspace_pred, "%s.txt" % seq), "w", encoding="utf-8"):
            pass
        padded.append(seq)
        print("  -> Created empty file for missing seq: %s" % seq)

    return padded


def main():
    args = parse_args()
    gt_ws = os.path.join(args.workspace, "gt")
    pred_ws = os.path.join(args.workspace, "pred", TRACKER_NAME)
    ensure_dir(gt_ws)
    ensure_dir(pred_ws)

    resolved_split = resolve_split(args.split, args.pred_json, args.gt_source_dir)
    target_sequences = get_split_sequences(resolved_split, args.gt_source_dir)
    pred_sequences = load_pred_sequences(args.pred_json, pred_ws)
    missing_pred, extra_pred = validate_pred_sequences(
        resolved_split,
        target_sequences,
        pred_sequences,
    )

    print("[INFO] Writing GT boxes as: %s" % args.box_format)
    print(
        "[EVAL][SPLIT] target_split=%s target_seq_count=%d pred_seq_count=%d missing_pred_seq_count=%d extra_pred_seq_count=%d"
        % (
            resolved_split,
            len(target_sequences),
            len(pred_sequences),
            len(missing_pred),
            len(extra_pred),
        )
    )

    if args.pred_json and (missing_pred or extra_pred):
        raise ValueError(
            "Prediction sequences do not match target split %s: missing=%d extra=%d"
            % (resolved_split, len(missing_pred), len(extra_pred))
        )

    cleanup_workspace(gt_ws, pred_ws, target_sequences, pred_sequences)
    convert_gt(gt_ws, target_sequences, args.box_format, args.gt_source_dir)
    generate_seqmap(gt_ws, resolved_split, target_sequences)
    padded_sequences = fill_missing(pred_ws, target_sequences)

    print(
        "[EVAL][PAD] did_pad_empty_files=%s padded_empty_count=%d"
        % ("True" if padded_sequences else "False", len(padded_sequences))
    )
    print("[SUCCESS] Evaluation environment ready.")


if __name__ == "__main__":
    main()
