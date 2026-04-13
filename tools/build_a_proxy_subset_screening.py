#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS_ROOT = (
    REPO_ROOT / "data/JRDB2019/train_dataset_with_activity/labels/labels_2d_stitched"
)
DEFAULT_OUT_DIR = REPO_ROOT / "research/seam-a/A1_proxy_subset_screening"

IMAGE_WIDTH = 3760
SEAM_BAND_PX = 400
VISIBLE_OCCLUSIONS = {"Fully_visible", "Mostly_visible"}

VALIDATION_SEQUENCES = (
    "clark-center-2019-02-28_1",
    "gates-ai-lab-2019-02-08_0",
    "huang-2-2019-01-25_0",
    "meyer-green-2019-03-16_0",
    "nvidia-aud-2019-04-18_0",
    "tressider-2019-03-16_1",
    "tressider-2019-04-26_2",
)

PROXY_TRAIN_WHITELIST = (
    "tressider-2019-03-16_0",
    "gates-to-clark-2019-02-28_1",
    "huang-basement-2019-01-25_0",
    "svl-meeting-gates-2-2019-04-08_1",
    "clark-center-2019-02-28_0",
)

PROXY_VAL_WHITELIST = (
    "clark-center-2019-02-28_1",
    "huang-2-2019-01-25_0",
    "nvidia-aud-2019-04-18_0",
    "gates-ai-lab-2019-02-08_0",
    "tressider-2019-03-16_1",
)

SEQUENCE_BUCKETS = {
    "tressider-2019-03-16_0": "seam_rich",
    "gates-to-clark-2019-02-28_1": "seam_rich",
    "huang-basement-2019-01-25_0": "seam_rich",
    "svl-meeting-gates-2-2019-04-08_1": "ordinary",
    "clark-center-2019-02-28_0": "hard",
    "clark-center-2019-02-28_1": "mandatory_probe",
    "huang-2-2019-01-25_0": "mandatory_probe",
    "nvidia-aud-2019-04-18_0": "mandatory_probe",
    "gates-ai-lab-2019-02-08_0": "ordinary_control",
    "tressider-2019-03-16_1": "ordinary_control",
}

SELECTION_REASONS = {
    "tressider-2019-03-16_0": "Selected for proxy-train as seam_rich: highest seam-band ratio among train sequences with compact seam-heavy supervision.",
    "gates-to-clark-2019-02-28_1": "Selected for proxy-train as seam_rich: elevated seam ratio with direct seam-traffic exposure.",
    "huang-basement-2019-01-25_0": "Selected for proxy-train as seam_rich: strong seam ratio plus meaningful seam-near and seam-crossing tracks.",
    "svl-meeting-gates-2-2019-04-08_1": "Selected for proxy-train as ordinary: low-seam control sequence to keep non-seam behavior visible.",
    "clark-center-2019-02-28_0": "Selected for proxy-train as hard: crowded and failure-prone sequence to keep hard-case pressure in the proxy subset.",
    "clark-center-2019-02-28_1": "Selected for proxy-val as mandatory_probe: fixed card2 smoke anchor and seam-sensitive validation probe.",
    "huang-2-2019-01-25_0": "Selected for proxy-val as mandatory_probe: fixed seam-focused validation probe.",
    "nvidia-aud-2019-04-18_0": "Selected for proxy-val as mandatory_probe: fixed failure-prone validation probe.",
    "gates-ai-lab-2019-02-08_0": "Selected for proxy-val as ordinary_control: lower-seam ordinary sequence for non-seam monitoring.",
    "tressider-2019-03-16_1": "Selected for proxy-val as ordinary_control: ordinary control with moderate crowding and lower seam emphasis than the mandatory probes.",
}

SEQUENCE_STATS_FIELDNAMES = (
    "split",
    "sequence",
    "num_frames",
    "gt_detections",
    "seam_gt_detections",
    "seam_gt_ratio",
    "seam_crossing_track_count",
    "seam_near_track_count",
    "avg_gt_per_frame",
    "bucket",
    "selected_proxy_train",
    "selected_proxy_val",
    "selection_reason",
)

EXPECTED_SEQUENCE_COUNT = 27
EXPECTED_TRAIN_SEQUENCE_COUNT = 20
EXPECTED_VAL_SEQUENCE_COUNT = 7
EXPECTED_PROXY_TRAIN_GT = 46699
EXPECTED_TRAIN_GT_TOTAL = 340184


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the fixed A-tree proxy subset whitelists and per-sequence statistics."
    )
    parser.add_argument(
        "--labels-root",
        default=str(DEFAULT_LABELS_ROOT),
        help="Root directory that contains JRDB stitched 2D label JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for the fixed proxy subset artifacts.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=IMAGE_WIDTH,
        help="JRDB stitched image width used for seam statistics.",
    )
    parser.add_argument(
        "--seam-band-px",
        type=int,
        default=SEAM_BAND_PX,
        help="Seam band width in pixels.",
    )
    return parser.parse_args()


def track_id_from_label(label_id):
    return int(str(label_id).split(":")[1]) + 1


def box_seam_flags(x, width, image_width, seam_band_px):
    x2 = x + width
    is_seam_near = (
        x < seam_band_px
        or x2 > image_width - seam_band_px
        or x < 0
        or x2 > image_width
    )
    is_seam_crossing = x < 0 or x2 > image_width
    return is_seam_near, is_seam_crossing


def iter_visible_annotations(label_payload):
    for frame_name in sorted(label_payload["labels"]):
        labels = label_payload["labels"][frame_name]
        visible = [
            item
            for item in labels
            if item["attributes"]["occlusion"] in VISIBLE_OCCLUSIONS
        ]
        yield frame_name, visible


def compute_sequence_row(sequence_path, image_width, seam_band_px):
    payload = json.loads(sequence_path.read_text())
    num_frames = 0
    gt_detections = 0
    seam_gt_detections = 0
    seam_crossing_tracks = set()
    seam_near_tracks = set()

    for _, annotations in iter_visible_annotations(payload):
        num_frames += 1
        gt_detections += len(annotations)
        for item in annotations:
            x, _, width, _ = item["box"]
            track_id = track_id_from_label(item["label_id"])
            is_seam_near, is_seam_crossing = box_seam_flags(
                x,
                width,
                image_width=image_width,
                seam_band_px=seam_band_px,
            )
            if is_seam_near:
                seam_gt_detections += 1
                seam_near_tracks.add(track_id)
            if is_seam_crossing:
                seam_crossing_tracks.add(track_id)

    seam_gt_ratio = (
        float(seam_gt_detections) / float(gt_detections) if gt_detections else 0.0
    )
    avg_gt_per_frame = float(gt_detections) / float(num_frames) if num_frames else 0.0
    sequence = sequence_path.stem
    split = "val" if sequence in VALIDATION_SEQUENCES else "train"

    return {
        "split": split,
        "sequence": sequence,
        "num_frames": num_frames,
        "gt_detections": gt_detections,
        "seam_gt_detections": seam_gt_detections,
        "seam_gt_ratio": f"{seam_gt_ratio:.6f}",
        "seam_crossing_track_count": len(seam_crossing_tracks),
        "seam_near_track_count": len(seam_near_tracks),
        "avg_gt_per_frame": f"{avg_gt_per_frame:.6f}",
        "bucket": SEQUENCE_BUCKETS.get(sequence, ""),
        "selected_proxy_train": "true" if sequence in PROXY_TRAIN_WHITELIST else "false",
        "selected_proxy_val": "true" if sequence in PROXY_VAL_WHITELIST else "false",
        "selection_reason": SELECTION_REASONS.get(sequence, ""),
    }


def compute_all_sequence_rows(labels_root, image_width, seam_band_px):
    sequence_paths = sorted(labels_root.glob("*.json"))
    return [
        compute_sequence_row(path, image_width=image_width, seam_band_px=seam_band_px)
        for path in sequence_paths
    ]


def validate_rows(rows):
    sequences = {row["sequence"] for row in rows}
    if len(rows) != EXPECTED_SEQUENCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SEQUENCE_COUNT} labeled sequences, found {len(rows)}."
        )
    if len(sequences) != EXPECTED_SEQUENCE_COUNT:
        raise RuntimeError("Duplicate sequence names detected in sequence_stats rows.")

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if len(train_rows) != EXPECTED_TRAIN_SEQUENCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_SEQUENCE_COUNT} train sequences, found {len(train_rows)}."
        )
    if len(val_rows) != EXPECTED_VAL_SEQUENCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_VAL_SEQUENCE_COUNT} val sequences, found {len(val_rows)}."
        )

    train_gt_total = sum(int(row["gt_detections"]) for row in train_rows)
    proxy_train_gt = sum(
        int(row["gt_detections"])
        for row in train_rows
        if row["sequence"] in PROXY_TRAIN_WHITELIST
    )
    if train_gt_total != EXPECTED_TRAIN_GT_TOTAL:
        raise RuntimeError(
            f"Expected train GT total {EXPECTED_TRAIN_GT_TOTAL}, found {train_gt_total}."
        )
    if proxy_train_gt != EXPECTED_PROXY_TRAIN_GT:
        raise RuntimeError(
            f"Expected proxy-train GT total {EXPECTED_PROXY_TRAIN_GT}, found {proxy_train_gt}."
        )

    proxy_train_ratio = proxy_train_gt / train_gt_total
    if not (0.10 <= proxy_train_ratio <= 0.15):
        raise RuntimeError(
            f"Proxy-train GT ratio {proxy_train_ratio:.6f} is outside the target [0.10, 0.15]."
        )

    missing_train = sorted(set(PROXY_TRAIN_WHITELIST) - sequences)
    missing_val = sorted(set(PROXY_VAL_WHITELIST) - sequences)
    if missing_train or missing_val:
        raise RuntimeError(
            f"Whitelists reference missing sequences: train={missing_train}, val={missing_val}"
        )

    return {
        "train_gt_total": train_gt_total,
        "proxy_train_gt": proxy_train_gt,
        "proxy_train_ratio": proxy_train_ratio,
    }


def write_whitelist(out_path, sequences):
    out_path.write_text("\n".join(sequences) + "\n")


def write_sequence_stats(out_path, rows):
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEQUENCE_STATS_FIELDNAMES)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["split"], item["sequence"])):
            writer.writerow(row)


def build_selection_note(summary):
    ratio_pct = summary["proxy_train_ratio"] * 100.0
    return "\n".join(
        [
            "# A1 Proxy Subset Screening",
            "",
            "A1 is a seam proxy subset screening task for the A-tree; it only fixes sequence lists and statistics, and does not launch training.",
            "",
            "## proxy-train",
            "- `tressider-2019-03-16_0` (`seam_rich`): highest seam-band ratio among the train sequences, making it the strongest compact seam-rich probe.",
            "- `gates-to-clark-2019-02-28_1` (`seam_rich`): direct seam-traffic sequence with clearly elevated seam exposure.",
            "- `huang-basement-2019-01-25_0` (`seam_rich`): strong seam ratio with meaningful seam-near and seam-crossing tracks.",
            "- `svl-meeting-gates-2-2019-04-08_1` (`ordinary`): low-seam ordinary control used to keep non-seam behavior visible.",
            "- `clark-center-2019-02-28_0` (`hard`): crowded hard sequence kept to preserve failure pressure inside the proxy subset.",
            "",
            "## proxy-val",
            "- `clark-center-2019-02-28_1` (`mandatory_probe`): fixed card2 smoke anchor and seam-sensitive validation probe.",
            "- `huang-2-2019-01-25_0` (`mandatory_probe`): fixed seam-focused validation probe.",
            "- `nvidia-aud-2019-04-18_0` (`mandatory_probe`): fixed failure-prone validation probe.",
            "- `gates-ai-lab-2019-02-08_0` (`ordinary_control`): ordinary control added to monitor non-seam behavior.",
            "- `tressider-2019-03-16_1` (`ordinary_control`): second ordinary control with moderate crowding and lower seam emphasis than the mandatory probes.",
            "",
            f"Budget summary: proxy-train covers `{summary['proxy_train_gt']} / {summary['train_gt_total']} = {ratio_pct:.1f}%` of train GT detections.",
            "",
        ]
    )


def build_proxy_subset_screening(
    labels_root=DEFAULT_LABELS_ROOT,
    out_dir=DEFAULT_OUT_DIR,
    image_width=IMAGE_WIDTH,
    seam_band_px=SEAM_BAND_PX,
):
    labels_root = Path(labels_root)
    out_dir = Path(out_dir)
    if not labels_root.exists():
        raise FileNotFoundError(f"labels_root does not exist: {labels_root}")

    rows = compute_all_sequence_rows(
        labels_root,
        image_width=image_width,
        seam_band_px=seam_band_px,
    )
    summary = validate_rows(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_whitelist_path = out_dir / "train_whitelist.txt"
    val_whitelist_path = out_dir / "val_whitelist.txt"
    sequence_stats_path = out_dir / "sequence_stats.csv"
    selection_note_path = out_dir / "selection_note.md"

    write_whitelist(train_whitelist_path, PROXY_TRAIN_WHITELIST)
    write_whitelist(val_whitelist_path, PROXY_VAL_WHITELIST)
    write_sequence_stats(sequence_stats_path, rows)
    selection_note_path.write_text(build_selection_note(summary))

    return {
        "out_dir": out_dir,
        "train_whitelist_path": train_whitelist_path,
        "val_whitelist_path": val_whitelist_path,
        "sequence_stats_path": sequence_stats_path,
        "selection_note_path": selection_note_path,
        **summary,
    }


def main():
    args = parse_args()
    summary = build_proxy_subset_screening(
        labels_root=args.labels_root,
        out_dir=args.out_dir,
        image_width=args.image_width,
        seam_band_px=args.seam_band_px,
    )
    print(
        "Built A1 proxy subset screening artifacts at "
        f"{summary['out_dir']} with proxy-train budget "
        f"{summary['proxy_train_gt']} / {summary['train_gt_total']} "
        f"({summary['proxy_train_ratio'] * 100.0:.1f}%)."
    )


if __name__ == "__main__":
    main()
