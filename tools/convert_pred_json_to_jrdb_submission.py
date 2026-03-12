import argparse
import csv
import json
import math
import os
import pickle
import zipfile
from collections import defaultdict

try:
    from tools.jrdb_eval_splits import extract_sequences_from_pred_json
except ImportError:
    from jrdb_eval_splits import extract_sequences_from_pred_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert OmniTrack JRDB JSON to official benchmark submission files."
    )
    parser.add_argument(
        "--pred-json",
        required=True,
        help="Path to results_jrdb2d.json generated from JRDB test inference.",
    )
    parser.add_argument(
        "--ann-file",
        required=True,
        help="Path to JRDB_infos_test_v1.2.pkl used for inference.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Submission txt output directory, usually <root>/CIWT/data.",
    )
    parser.add_argument(
        "--zip-output",
        default=None,
        help="Optional output zip path for benchmark upload.",
    )
    parser.add_argument(
        "--zip-source-dir",
        default=None,
        help="Directory to archive when --zip-output is set, usually <root>/CIWT.",
    )
    parser.add_argument(
        "--sequence-map-output",
        default=None,
        help="Optional CSV mapping from numbered submission files to JRDB sequence names.",
    )
    parser.add_argument(
        "--box-format",
        default="xywh",
        choices=["xywh", "xyxy"],
        help="Columns 7:10 written as xywh or x0y0x1y1.",
    )
    return parser.parse_args()


def _extract_sequence_and_frame(token):
    if not isinstance(token, str) or "_" not in token:
        raise ValueError("Invalid JRDB token: %r" % (token,))
    seq_name, frame = token.rsplit("_", 1)
    if not frame.isdigit():
        raise ValueError("Invalid JRDB token frame suffix: %r" % (token,))
    return seq_name, int(frame)


def load_target_sequences_from_ann_file(ann_file):
    with open(ann_file, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or "infos" not in data:
        raise ValueError("Annotation file must contain dict key 'infos': %s" % ann_file)

    infos = data["infos"]
    if not isinstance(infos, list) or not infos:
        raise ValueError("Annotation file has no infos: %s" % ann_file)

    sequences = set()
    for info in infos:
        token = info.get("token")
        if token is None:
            raise ValueError("Annotation info item is missing token in %s" % ann_file)
        seq_name, _ = _extract_sequence_and_frame(token)
        sequences.add(seq_name)

    ordered = sorted(sequences)
    if not ordered:
        raise ValueError("Could not derive JRDB sequences from annotation file: %s" % ann_file)
    return ordered


def load_flat_predictions(pred_json_path):
    with open(pred_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    content = data.get("results", data) if isinstance(data, dict) else data
    flat_list = []

    if isinstance(content, dict):
        for sample_token, objs in content.items():
            if not isinstance(objs, list):
                raise ValueError("Prediction JSON frame %r does not contain a list." % sample_token)
            for obj in objs:
                if "sample_token" not in obj:
                    obj = dict(obj)
                    obj["sample_token"] = sample_token
                flat_list.append(obj)
    elif isinstance(content, list):
        flat_list = content
    else:
        raise ValueError(
            "Prediction JSON must be a dict with 'results' or a list of annotations."
        )

    return flat_list


def cleanup_output_dir(output_dir):
    removed = 0
    if not os.path.isdir(output_dir):
        return removed

    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if name.endswith(".txt") and os.path.isfile(path):
            os.remove(path)
            removed += 1
    return removed


def write_sequence_map(sequence_map_output, target_sequences):
    parent_dir = os.path.dirname(sequence_map_output)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(sequence_map_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["submission_file", "sequence_name"])
        for index, seq_name in enumerate(target_sequences):
            writer.writerow(["%04d.txt" % index, seq_name])


def build_submission_lines(flat_predictions, target_sequences, box_format):
    target_set = set(target_sequences)
    seq_outputs = defaultdict(list)
    seq_stats = defaultdict(
        lambda: {
            "input_objs": 0,
            "drop_none_tid": 0,
            "drop_bad_tid": 0,
            "drop_bad_box": 0,
            "written_objs": 0,
        }
    )

    for obj in flat_predictions:
        token = obj.get("sample_token")
        if token is None:
            raise ValueError("Prediction item is missing sample_token.")

        seq_name, frame_idx = _extract_sequence_and_frame(token)
        if seq_name not in target_set:
            raise ValueError(
                "Prediction JSON contains sequence %r that is not present in annotation file."
                % seq_name
            )

        stat = seq_stats[seq_name]
        stat["input_objs"] += 1

        tid = obj.get("tracking_id", "None")
        if tid in (None, "None"):
            stat["drop_none_tid"] += 1
            continue

        try:
            tid = int(float(tid))
        except Exception:
            stat["drop_bad_tid"] += 1
            continue

        x1y1 = obj.get("x1y1", [])
        size = obj.get("size", [])
        if len(x1y1) < 2 or len(size) < 2:
            stat["drop_bad_box"] += 1
            continue

        left = float(x1y1[0])
        top = float(x1y1[1])
        width = float(size[0])
        height = float(size[1])
        if box_format == "xywh":
            box3 = width
            box4 = height
        else:
            box3 = left + width
            box4 = top + height

        score = float(obj.get("detection_score", -1))
        if math.isnan(score):
            score = -1.0

        line = (
            f"{frame_idx:d} {tid:d} Pedestrian 0 0 0 "
            f"{left:.2f} {top:.2f} {box3:.2f} {box4:.2f} "
            f"-1 -1 -1 -1000 -1000 -1000 0 {score:.4f}\n"
        )
        seq_outputs[seq_name].append(line)
        stat["written_objs"] += 1

    return seq_outputs, seq_stats


def line_sort_key(line):
    cols = line.split()
    return int(cols[0]), int(cols[1])


def write_submission_files(output_dir, target_sequences, seq_outputs):
    os.makedirs(output_dir, exist_ok=True)
    removed = cleanup_output_dir(output_dir)

    padded_empty_count = 0
    for index, seq_name in enumerate(target_sequences):
        file_name = "%04d.txt" % index
        path = os.path.join(output_dir, file_name)
        lines = sorted(seq_outputs.get(seq_name, []), key=line_sort_key)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        if not lines:
            padded_empty_count += 1

    return removed, padded_empty_count


def create_submission_zip(zip_output, zip_source_dir):
    if zip_source_dir is None:
        raise ValueError("--zip-source-dir is required when --zip-output is set.")
    if not os.path.isdir(zip_source_dir):
        raise ValueError("Zip source directory does not exist: %s" % zip_source_dir)

    zip_parent = os.path.dirname(zip_output)
    if zip_parent:
        os.makedirs(zip_parent, exist_ok=True)

    archive_root = os.path.dirname(zip_source_dir)
    with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(zip_source_dir):
            for name in sorted(files):
                full_path = os.path.join(root, name)
                arcname = os.path.relpath(full_path, archive_root)
                zf.write(full_path, arcname)


def main():
    args = parse_args()

    target_sequences = load_target_sequences_from_ann_file(args.ann_file)
    pred_sequences = extract_sequences_from_pred_json(args.pred_json)

    target_set = set(target_sequences)
    pred_set = set(pred_sequences)
    missing_pred = sorted(target_set - pred_set)
    extra_pred = sorted(pred_set - target_set)
    if extra_pred:
        raise ValueError(
            "Prediction JSON contains sequences outside official test ann file: %s"
            % ", ".join(extra_pred)
        )

    flat_predictions = load_flat_predictions(args.pred_json)
    seq_outputs, seq_stats = build_submission_lines(
        flat_predictions,
        target_sequences,
        args.box_format,
    )
    removed_old_count, padded_empty_count = write_submission_files(
        args.output_dir,
        target_sequences,
        seq_outputs,
    )

    if args.sequence_map_output:
        write_sequence_map(args.sequence_map_output, target_sequences)

    if args.zip_output:
        create_submission_zip(args.zip_output, args.zip_source_dir)

    print(
        "[SUBMIT][SPLIT] "
        f"target_split=test "
        f"target_seq_count={len(target_sequences)} "
        f"pred_seq_count={len(pred_sequences)} "
        f"missing_pred_seq_count={len(missing_pred)} "
        f"extra_pred_seq_count={len(extra_pred)}"
    )
    print(
        "[SUBMIT][FILES] "
        f"output_dir={args.output_dir} "
        f"removed_old_txt_count={removed_old_count} "
        f"did_pad_empty_files={'yes' if padded_empty_count else 'no'} "
        f"padded_empty_count={padded_empty_count} "
        f"box_format={args.box_format}"
    )
    if target_sequences:
        print(
            "[SUBMIT][MAP] "
            f"0000={target_sequences[0]} "
            f"{len(target_sequences) - 1:04d}={target_sequences[-1]}"
        )
    for seq_name in target_sequences:
        stats = seq_stats[seq_name]
        print(
            "[SUBMIT][SEQ] "
            f"{seq_name} "
            f"input={stats['input_objs']} "
            f"drop_none_tid={stats['drop_none_tid']} "
            f"drop_bad_tid={stats['drop_bad_tid']} "
            f"drop_bad_box={stats['drop_bad_box']} "
            f"written={stats['written_objs']}"
        )
    if args.sequence_map_output:
        print(f"[SUBMIT][ARTIFACT] sequence_map={args.sequence_map_output}")
    if args.zip_output:
        print(f"[SUBMIT][ARTIFACT] submission_zip={args.zip_output}")


if __name__ == "__main__":
    main()
