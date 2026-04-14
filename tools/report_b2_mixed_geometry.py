#!/usr/bin/env python
import argparse
import csv
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DIAG_MODULE_PATH = REPO_ROOT / "tools/diag_seam_metric_split.py"
DIAG_SPEC = importlib.util.spec_from_file_location(
    "b2_diag_seam_metric_split",
    DIAG_MODULE_PATH,
)
DIAG_MODULE = importlib.util.module_from_spec(DIAG_SPEC)
DIAG_SPEC.loader.exec_module(DIAG_MODULE)

TRACKER_SUMMARY_FLOAT_KEYS = ("HOTA", "MOTA", "IDF1", "OSPA", "Dets", "IDs")
SLICE_SUBSETS = ("full", "seam", "high_lat", "seam_high_lat")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarise B2 mixed-geometry runs into factual tables and a markdown report."
    )
    parser.add_argument(
        "--variants-csv",
        required=True,
        help="Variant manifest CSV emitted by tools/run_b2_mixed_geometry_matrix.sh.",
    )
    parser.add_argument(
        "--diag-dir",
        required=True,
        help="Diagnostic output root from tools/diag_seam_metric_split.py.",
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
        help="JRDB split tag for TrackEval.",
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
        "--frozen-bad-case-seqs",
        nargs="+",
        required=True,
        help="Frozen bad-case sequence list.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for report tables and markdown.",
    )
    return parser.parse_args()


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
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


def read_variants(variants_csv):
    rows = read_csv_rows(variants_csv)
    for row in rows:
        row["runtime_sec"] = int(row["runtime_sec"])
    return rows


def read_trackeval_combined(trackers_folder, tracker_name):
    summary_path = Path(trackers_folder) / tracker_name / "pedestrian_summary.csv"
    rows = read_csv_rows(summary_path)
    combined_rows = [row for row in rows if row["seq"] == "COMBINED"]
    if len(combined_rows) != 1:
        raise ValueError(f"Unexpected TrackEval summary format: {summary_path}")
    row = combined_rows[0]
    return {
        "HOTA": float(row["HOTA"]),
        "MOTA": float(row["MOTA"]),
        "IDF1": float(row["IDF1"]),
        "OSPA": float(row["OSPA"]),
        "Dets": int(float(row["Dets"])),
        "IDs": int(float(row["IDs"])),
    }


def read_diag_combined(diag_dir, tracker_name):
    combined_rows = read_csv_rows(Path(diag_dir) / tracker_name / "combined_metrics.csv")
    coverage_rows = read_csv_rows(Path(diag_dir) / tracker_name / "coverage.csv")

    metric_by_subset = {}
    for row in combined_rows:
        if row["seq"] != "COMBINED":
            continue
        metric_by_subset[row["subset"]] = {
            "HOTA": float(row["HOTA"]),
            "DetA": float(row["DetA"]),
            "AssA": float(row["AssA"]),
            "IDF1": float(row["IDF1"]),
            "FP": int(float(row["FP"])),
            "IDSW": int(float(row["IDSW"])),
        }

    coverage_by_subset = {}
    for row in coverage_rows:
        if row["seq"] != "COMBINED":
            continue
        coverage_by_subset[row["subset"]] = {
            "gt_dets": int(float(row["gt_dets"])),
            "tracker_dets": int(float(row["tracker_dets"])),
            "gt_ids": int(float(row["gt_ids"])),
            "tracker_ids": int(float(row["tracker_ids"])),
            "gt_fraction_of_full": float(row["gt_fraction_of_full"]),
            "tracker_fraction_of_full": float(row["tracker_fraction_of_full"]),
        }

    return metric_by_subset, coverage_by_subset


def combine_coverage_rows(coverage_rows, tracker_name):
    combined_rows = []
    full_rows = [row for row in coverage_rows if row["subset"] == "full"]
    full_gt = int(sum(row["gt_dets"] for row in full_rows))
    full_tracker = int(sum(row["tracker_dets"] for row in full_rows))
    full_ids = int(sum(row["gt_ids"] for row in full_rows))
    full_tracker_ids = int(sum(row["tracker_ids"] for row in full_rows))
    combined_rows.append(
        {
            "tracker": tracker_name,
            "subset": "full",
            "seq": "COMBINED_BAD_CASE",
            "gt_dets": full_gt,
            "tracker_dets": full_tracker,
            "gt_ids": full_ids,
            "tracker_ids": full_tracker_ids,
            "gt_fraction_of_full": 1.0,
            "tracker_fraction_of_full": 1.0,
        }
    )

    for subset_name in ("seam", "high_lat", "seam_high_lat"):
        subset_rows = [row for row in coverage_rows if row["subset"] == subset_name]
        subset_gt = int(sum(row["gt_dets"] for row in subset_rows))
        subset_tracker = int(sum(row["tracker_dets"] for row in subset_rows))
        combined_rows.append(
            {
                "tracker": tracker_name,
                "subset": subset_name,
                "seq": "COMBINED_BAD_CASE",
                "gt_dets": subset_gt,
                "tracker_dets": subset_tracker,
                "gt_ids": int(sum(row["gt_ids"] for row in subset_rows)),
                "tracker_ids": int(sum(row["tracker_ids"] for row in subset_rows)),
                "gt_fraction_of_full": (float(subset_gt) / float(full_gt)) if full_gt else 0.0,
                "tracker_fraction_of_full": (
                    float(subset_tracker) / float(full_tracker)
                )
                if full_tracker
                else 0.0,
            }
        )

    return combined_rows


class ReportDiagArgs:
    def __init__(self, args, tracker_name):
        self.gt_folder = args.gt_folder
        self.trackers_folder = args.trackers_folder
        self.split_name = args.split_name
        self.tracker_sub_folder = "data"
        self.class_name = "pedestrian"
        self.image_width = args.image_width
        self.image_height = args.image_height
        self.seam_band_px = args.seam_band_px
        self.high_lat_deg = args.high_lat_deg
        self.tracker_name = tracker_name


def compute_frozen_bad_case(diag_args, tracker_name, frozen_bad_case_seqs):
    dataset = DIAG_MODULE.make_dataset(diag_args, tracker_name)
    target_seqs = [seq for seq in dataset.seq_list if seq in frozen_bad_case_seqs]
    if len(target_seqs) != len(frozen_bad_case_seqs):
        missing = sorted(set(frozen_bad_case_seqs) - set(target_seqs))
        raise ValueError(f"Frozen bad-case sequences not found for {tracker_name}: {missing}")

    per_subset_seq_metrics = {subset_name: {} for subset_name in SLICE_SUBSETS}
    per_sequence_rows = []
    coverage_rows = []
    full_summary = None

    for seq in target_seqs:
        raw_data = dataset.get_raw_seq_data(tracker_name, seq, is_3d=False)
        preprocessed = dataset.get_preprocessed_seq_data(raw_data, diag_args.class_name)
        subset_memberships = DIAG_MODULE.build_subset_memberships(
            preprocessed,
            image_width=diag_args.image_width,
            image_height=diag_args.image_height,
            seam_band_px=diag_args.seam_band_px,
            high_lat_deg=diag_args.high_lat_deg,
        )
        full_gt = int(preprocessed["num_gt_dets"])
        full_tracker = int(preprocessed["num_tracker_dets"])

        for subset_name in SLICE_SUBSETS:
            subset_data = DIAG_MODULE.build_subset_data(
                preprocessed, subset_name, subset_memberships
            )
            metric_bundle = DIAG_MODULE.evaluate_metric_bundle(subset_data)
            per_subset_seq_metrics[subset_name][seq] = metric_bundle
            summary = DIAG_MODULE.summarise_metric_bundle(metric_bundle)
            per_sequence_rows.append(
                {
                    "tracker": tracker_name,
                    "subset": subset_name,
                    "seq": seq,
                    **summary,
                }
            )
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
                        float(subset_data["num_gt_dets"]) / float(full_gt) if full_gt else 0.0
                    ),
                    "tracker_fraction_of_full": (
                        float(subset_data["num_tracker_dets"]) / float(full_tracker)
                        if full_tracker
                        else 0.0
                    ),
                }
            )

    combined_rows = []
    for subset_name in SLICE_SUBSETS:
        combined = DIAG_MODULE.combine_metric_bundle(per_subset_seq_metrics[subset_name])
        summary = DIAG_MODULE.summarise_metric_bundle(combined)
        row = {
            "tracker": tracker_name,
            "subset": subset_name,
            "seq": "COMBINED_BAD_CASE",
            **summary,
        }
        combined_rows.append(row)
        if subset_name == "full":
            full_summary = row

    return {
        "per_sequence_metrics": per_sequence_rows,
        "combined_metrics": combined_rows,
        "coverage": [*coverage_rows, *combine_coverage_rows(coverage_rows, tracker_name)],
        "full_summary": full_summary,
    }


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def fmt_float(value, digits=3):
    return f"{float(value):.{digits}f}"


def build_report(
    variants,
    trackeval_rows,
    slice_rows,
    bad_case_rows,
    frozen_bad_case_seqs,
):
    runtime_rows = []
    trackeval_table_rows = []
    slice_table_rows = []
    bad_case_table_rows = []

    for variant in variants:
        runtime_rows.append(
            {
                "variant": variant["display_name"],
                "mode": variant["association_mode"],
                "freshness": variant["freshness"],
                "seconds": variant["runtime_sec"],
            }
        )

        trackeval = trackeval_rows[variant["tracker_name"]]
        trackeval_table_rows.append(
            {
                "variant": variant["display_name"],
                "tracker": variant["tracker_name"],
                "HOTA": fmt_float(trackeval["HOTA"]),
                "MOTA": fmt_float(trackeval["MOTA"]),
                "IDF1": fmt_float(trackeval["IDF1"]),
                "OSPA": fmt_float(trackeval["OSPA"], digits=5),
                "Dets": trackeval["Dets"],
                "IDs": trackeval["IDs"],
            }
        )

        for subset_name in SLICE_SUBSETS:
            slice_row = slice_rows[(variant["tracker_name"], subset_name)]
            slice_table_rows.append(
                {
                    "variant": variant["display_name"],
                    "subset": subset_name,
                    "HOTA": fmt_float(slice_row["HOTA"]),
                    "AssA": fmt_float(slice_row["AssA"]),
                    "IDF1": fmt_float(slice_row["IDF1"]),
                    "FP": slice_row["FP"],
                    "IDSW": slice_row["IDSW"],
                    "gt_dets": slice_row["gt_dets"],
                    "tracker_dets": slice_row["tracker_dets"],
                    "gt_ids": slice_row["gt_ids"],
                    "tracker_ids": slice_row["tracker_ids"],
                }
            )

            bad_case_row = bad_case_rows[(variant["tracker_name"], subset_name)]
            bad_case_table_rows.append(
                {
                    "variant": variant["display_name"],
                    "subset": subset_name,
                    "HOTA": fmt_float(bad_case_row["HOTA"]),
                    "AssA": fmt_float(bad_case_row["AssA"]),
                    "IDF1": fmt_float(bad_case_row["IDF1"]),
                    "FP": bad_case_row["FP"],
                    "IDSW": bad_case_row["IDSW"],
                    "gt_dets": bad_case_row["gt_dets"],
                    "tracker_dets": bad_case_row["tracker_dets"],
                    "gt_ids": bad_case_row["gt_ids"],
                    "tracker_ids": bad_case_row["tracker_ids"],
                }
            )

    lines = [
        "# 树 B2：Mixed-Geometry No-Train Probes 实验报告",
        "",
        "## 基本信息",
        f"- 仓库路径：`{REPO_ROOT}`",
        "- 评测数据：`JRDB val`",
        "- 选用 GPU：`GPU 0`",
        "- mixed-geometry 版本数：`6`",
        "",
        "## 版本定义",
        markdown_table(
            ("variant", "mode", "freshness", "tracker"),
            [
                {
                    "variant": row["display_name"],
                    "mode": row["association_mode"],
                    "freshness": row["freshness"],
                    "tracker": row["tracker_name"],
                }
                for row in variants
            ],
        ),
        "",
        "## 运行时间",
        markdown_table(("variant", "mode", "freshness", "seconds"), runtime_rows),
        "",
        "## TrackEval COMBINED 指标",
        markdown_table(
            ("variant", "tracker", "HOTA", "MOTA", "IDF1", "OSPA", "Dets", "IDs"),
            trackeval_table_rows,
        ),
        "",
        "## 分层 COMBINED 指标与覆盖",
        markdown_table(
            (
                "variant",
                "subset",
                "HOTA",
                "AssA",
                "IDF1",
                "FP",
                "IDSW",
                "gt_dets",
                "tracker_dets",
                "gt_ids",
                "tracker_ids",
            ),
            slice_table_rows,
        ),
        "",
        "## 冻结 bad-case 名单",
        *[f"- `{seq}`" for seq in frozen_bad_case_seqs],
        "",
        "## 冻结 bad-case 聚合指标与覆盖",
        markdown_table(
            (
                "variant",
                "subset",
                "HOTA",
                "AssA",
                "IDF1",
                "FP",
                "IDSW",
                "gt_dets",
                "tracker_dets",
                "gt_ids",
                "tracker_ids",
            ),
            bad_case_table_rows,
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = read_variants(REPO_ROOT / args.variants_csv)
    diag_dir = (REPO_ROOT / args.diag_dir).resolve()
    trackers_folder = (REPO_ROOT / args.trackers_folder).resolve()

    trackeval_combined_rows = []
    slice_combined_rows = []
    slice_lookup = {}
    bad_case_combined_rows = []
    bad_case_lookup = {}

    for variant in variants:
        tracker_name = variant["tracker_name"]
        trackeval_summary = read_trackeval_combined(trackers_folder, tracker_name)
        trackeval_combined_rows.append(
            {
                "variant_id": variant["variant_id"],
                "display_name": variant["display_name"],
                "tracker_name": tracker_name,
                **trackeval_summary,
            }
        )

        diag_metrics, diag_coverage = read_diag_combined(diag_dir, tracker_name)
        for subset_name in SLICE_SUBSETS:
            row = {
                "variant_id": variant["variant_id"],
                "display_name": variant["display_name"],
                "tracker_name": tracker_name,
                "subset": subset_name,
                **diag_metrics[subset_name],
                **diag_coverage[subset_name],
            }
            slice_combined_rows.append(row)
            slice_lookup[(tracker_name, subset_name)] = row

        diag_args = ReportDiagArgs(args, tracker_name)
        bad_case_data = compute_frozen_bad_case(
            diag_args=diag_args,
            tracker_name=tracker_name,
            frozen_bad_case_seqs=args.frozen_bad_case_seqs,
        )

        write_csv(
            out_dir / "frozen_bad_case" / tracker_name / "combined_metrics.csv",
            bad_case_data["combined_metrics"],
        )
        write_csv(
            out_dir / "frozen_bad_case" / tracker_name / "coverage.csv",
            bad_case_data["coverage"],
        )
        write_csv(
            out_dir / "frozen_bad_case" / tracker_name / "per_sequence_metrics.csv",
            bad_case_data["per_sequence_metrics"],
        )

        bad_case_coverage_rows = [
            row for row in bad_case_data["coverage"] if row["seq"] == "COMBINED_BAD_CASE"
        ]
        bad_case_coverage_by_subset = {row["subset"]: row for row in bad_case_coverage_rows}
        for row in bad_case_data["combined_metrics"]:
            subset_name = row["subset"]
            merged = {
                "variant_id": variant["variant_id"],
                "display_name": variant["display_name"],
                "tracker_name": tracker_name,
                "subset": subset_name,
                "HOTA": float(row["HOTA"]),
                "DetA": float(row["DetA"]),
                "AssA": float(row["AssA"]),
                "IDF1": float(row["IDF1"]),
                "FP": int(row["FP"]),
                "IDSW": int(row["IDSW"]),
                "gt_dets": int(bad_case_coverage_by_subset[subset_name]["gt_dets"]),
                "tracker_dets": int(bad_case_coverage_by_subset[subset_name]["tracker_dets"]),
                "gt_ids": int(bad_case_coverage_by_subset[subset_name]["gt_ids"]),
                "tracker_ids": int(bad_case_coverage_by_subset[subset_name]["tracker_ids"]),
            }
            bad_case_combined_rows.append(merged)
            bad_case_lookup[(tracker_name, subset_name)] = merged

    write_csv(out_dir / "variant_runs.csv", variants)
    write_csv(out_dir / "trackeval_combined.csv", trackeval_combined_rows)
    write_csv(out_dir / "slice_combined.csv", slice_combined_rows)
    write_csv(out_dir / "frozen_bad_case_combined.csv", bad_case_combined_rows)

    report_text = build_report(
        variants=variants,
        trackeval_rows={row["tracker_name"]: row for row in trackeval_combined_rows},
        slice_rows=slice_lookup,
        bad_case_rows=bad_case_lookup,
        frozen_bad_case_seqs=args.frozen_bad_case_seqs,
    )
    report_path = out_dir / "EXPERIMENT_REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Wrote B2 report to {report_path}")


if __name__ == "__main__":
    main()
