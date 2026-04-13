import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path


ENUM_OCCLUSION = (
    "Fully_visible",
    "Mostly_visible",
    "Severely_occluded",
    "Fully_occluded",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export JRDB 2D GT/predictions to TrackEval KITTI-style files."
    )
    parser.add_argument(
        "--ann-file",
        required=True,
        help="JRDB pkl used for the evaluation split, e.g. JRDB_infos_val_v1.2.pkl",
    )
    parser.add_argument(
        "--gt-labels-dir",
        default="data/JRDB2019/train_dataset_with_activity/labels/labels_2d_stitched",
        help="Directory containing raw JRDB stitched 2D labels.",
    )
    parser.add_argument(
        "--pred-json",
        default=None,
        help="results_jrdb2d.json produced by tools/test.py",
    )
    parser.add_argument(
        "--split-name",
        default="val",
        help="Split tag used in evaluate_tracking.seqmap.<split-name>.",
    )
    parser.add_argument(
        "--gt-out-dir",
        default="jrdb_toolkit/tracking_eval/TrackEval/data/gt/jrdb/jrdb_2d_box_val",
        help="Output GT directory for TrackEval.",
    )
    parser.add_argument(
        "--trackers-out-dir",
        default="jrdb_toolkit/tracking_eval/TrackEval/data/trackers/jrdb/jrdb_2d_box_val",
        help="Output tracker root directory for TrackEval.",
    )
    parser.add_argument(
        "--tracker-name",
        default="OmniTrack",
        help="Tracker name under the TrackEval trackers directory.",
    )
    parser.add_argument(
        "--sequence-names",
        nargs="+",
        default=None,
        help="Optional JRDB stitched sequence names to export.",
    )
    parser.add_argument(
        "--inverse-roll-px",
        type=float,
        default=0.0,
        help="Inverse horizontal roll applied to predictions before export.",
    )
    parser.add_argument(
        "--image-width",
        type=float,
        default=3760.0,
        help="Original stitched panorama width used for inverse roll.",
    )
    parser.add_argument(
        "--skip-gt",
        action="store_true",
        help="Only export predictions.",
    )
    parser.add_argument(
        "--skip-pred",
        action="store_true",
        help="Only export GT.",
    )
    return parser.parse_args()


def load_infos(ann_file):
    with open(ann_file, "rb") as f:
        obj = pickle.load(f)
    infos = obj["infos"] if isinstance(obj, dict) and "infos" in obj else obj
    return infos


def normalize_sequence_names(sequence_names):
    if sequence_names is None:
        return None
    normalized = {
        str(sequence).strip()
        for sequence in sequence_names
        if str(sequence).strip()
    }
    return normalized or None


def build_split_index(infos, sequence_names=None):
    sequence_names = normalize_sequence_names(sequence_names)
    seq_to_frames = defaultdict(set)
    for info in infos:
        token = info["token"]
        seq_name, frame_name = token.rsplit("_", 1)
        if sequence_names is not None and seq_name not in sequence_names:
            continue
        seq_to_frames[seq_name].add(int(frame_name))
    return {seq: sorted(frames) for seq, frames in seq_to_frames.items()}


def safe_track_id(value):
    if value is None:
        return -1
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or not value.is_integer():
            return -1
        return int(value)
    try:
        text = str(value).strip()
        if not text:
            return -1
        return int(text)
    except ValueError:
        try:
            numeric = float(str(value).strip())
        except ValueError:
            return -1
        if math.isnan(numeric) or not numeric.is_integer():
            return -1
        return int(numeric)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def inverse_roll_x(x, inverse_roll_px, image_width):
    if image_width <= 0:
        return float(x)
    return float((float(x) - float(inverse_roll_px)) % float(image_width))


def write_seqmap(gt_out_dir, split_name, seq_to_frames):
    lines = []
    for seq_name in sorted(seq_to_frames):
        num_frames = max(seq_to_frames[seq_name]) + 1
        lines.append(f"{seq_name} empty 000000 {num_frames}\n")
    seqmap_path = gt_out_dir / f"evaluate_tracking.seqmap.{split_name}"
    write_lines(seqmap_path, lines)
    return seqmap_path


def convert_gt(seq_to_frames, gt_labels_dir, gt_out_dir):
    label_out_dir = gt_out_dir / "label_02"
    summary = {}
    for seq_name, keep_frames in sorted(seq_to_frames.items()):
        label_path = gt_labels_dir / f"{seq_name}.json"
        raw = read_json(label_path)
        keep_frame_set = set(keep_frames)
        lines = []
        det_count = 0

        for frame_name, annos in sorted(raw["labels"].items()):
            frame_id = int(frame_name.split(".")[0])
            if frame_id not in keep_frame_set:
                continue
            for anno in annos:
                label_id = anno["label_id"]
                if not label_id.startswith("pedestrian:"):
                    continue
                attrs = anno.get("attributes", {})
                truncated = int(str(attrs.get("truncated", "false")).lower() == "true")
                occlusion_name = attrs.get("occlusion", ENUM_OCCLUSION[0])
                occlusion = ENUM_OCCLUSION.index(occlusion_name)
                x, y, w, h = anno["box"]
                track_id = int(label_id.split(":")[-1]) + 1
                line = (
                    f"{frame_id} {track_id} Pedestrian "
                    f"{truncated} {occlusion} -1 "
                    f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} "
                    "-1 -1 -1 -1 -1 -1 -1\n"
                )
                lines.append(line)
                det_count += 1

        write_lines(label_out_dir / f"{seq_name}.txt", lines)
        summary[seq_name] = {"frames": len(keep_frames), "detections": det_count}
    return summary


def convert_pred(
    seq_to_frames,
    pred_json,
    trackers_out_dir,
    tracker_name,
    inverse_roll_px=0.0,
    image_width=3760.0,
):
    tracker_dir = trackers_out_dir / tracker_name / "data"
    raw = read_json(pred_json)
    pred_results = raw["results"]
    seq_lines = defaultdict(list)
    seq_frames = {seq: set(frames) for seq, frames in seq_to_frames.items()}
    skip_invalid_id = 0

    for sample_token, annos in sorted(pred_results.items()):
        seq_name, frame_name = sample_token.rsplit("_", 1)
        if seq_name not in seq_frames:
            continue
        frame_id = int(frame_name)
        if frame_id not in seq_frames[seq_name]:
            continue

        for anno in annos:
            track_id = safe_track_id(anno.get("tracking_id"))
            if track_id < 0:
                skip_invalid_id += 1
                continue
            x, y = anno["x1y1"]
            w, h = anno["size"]
            if inverse_roll_px:
                x = inverse_roll_x(x, inverse_roll_px, image_width)
            score = float(anno.get("detection_score", 1.0))
            if math.isnan(score):
                score = 1.0
            line = (
                f"{frame_id} {track_id} Pedestrian "
                f"0 0 -1 "
                f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} "
                f"-1 -1 -1 -1 -1 -1 -1 {score:.6f}\n"
            )
            seq_lines[seq_name].append(line)

    summary = {}
    for seq_name in sorted(seq_to_frames):
        lines = seq_lines.get(seq_name, [])
        write_lines(tracker_dir / f"{seq_name}.txt", lines)
        summary[seq_name] = {"detections": len(lines)}

    return summary, skip_invalid_id, tracker_dir


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ann_file = (repo_root / args.ann_file).resolve()
    gt_labels_dir = (repo_root / args.gt_labels_dir).resolve()
    gt_out_dir = (repo_root / args.gt_out_dir).resolve()
    trackers_out_dir = (repo_root / args.trackers_out_dir).resolve()
    pred_json = None if args.pred_json is None else (repo_root / args.pred_json).resolve()

    infos = load_infos(ann_file)
    seq_to_frames = build_split_index(infos, sequence_names=args.sequence_names)

    output_summary = {
        "split_name": args.split_name,
        "ann_file": str(ann_file),
        "num_sequences": len(seq_to_frames),
        "num_frames": sum(len(frames) for frames in seq_to_frames.values()),
    }

    if not args.skip_gt:
        gt_out_dir.mkdir(parents=True, exist_ok=True)
        seqmap_path = write_seqmap(gt_out_dir, args.split_name, seq_to_frames)
        gt_summary = convert_gt(seq_to_frames, gt_labels_dir, gt_out_dir)
        output_summary["gt_out_dir"] = str(gt_out_dir)
        output_summary["seqmap_path"] = str(seqmap_path)
        output_summary["gt_summary"] = gt_summary

    if not args.skip_pred:
        if pred_json is None:
            raise ValueError("--pred-json is required unless --skip-pred is set.")
        pred_summary, skip_invalid_id, tracker_dir = convert_pred(
            seq_to_frames=seq_to_frames,
            pred_json=pred_json,
            trackers_out_dir=trackers_out_dir,
            tracker_name=args.tracker_name,
            inverse_roll_px=args.inverse_roll_px,
            image_width=args.image_width,
        )
        output_summary["trackers_out_dir"] = str(trackers_out_dir)
        output_summary["tracker_name"] = args.tracker_name
        output_summary["tracker_data_dir"] = str(tracker_dir)
        output_summary["pred_summary"] = pred_summary
        output_summary["skipped_invalid_track_ids"] = skip_invalid_id
        output_summary["inverse_roll_px"] = float(args.inverse_roll_px)
        output_summary["image_width"] = float(args.image_width)

    print(json.dumps(output_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
