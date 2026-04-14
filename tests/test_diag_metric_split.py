import importlib.util
import pathlib

import numpy as np


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tools/diag_seam_metric_split.py"
)
SPEC = importlib.util.spec_from_file_location("diag_seam_metric_split", MODULE_PATH)
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


def test_build_subset_memberships_adds_high_lat_and_seam_high_lat():
    data = {
        "gt_dets": [
            np.array(
                [
                    [0.0, 0.0, 100.0, 60.0],
                    [1000.0, 210.0, 80.0, 60.0],
                ],
                dtype=float,
            )
        ],
        "tracker_dets": [
            np.array(
                [
                    [3700.0, 0.0, 100.0, 60.0],
                    [1200.0, 210.0, 80.0, 60.0],
                ],
                dtype=float,
            )
        ],
    }

    memberships = DIAG.build_subset_memberships(
        data,
        image_width=3760.0,
        image_height=480.0,
        seam_band_px=400.0,
        high_lat_deg=45.0,
    )

    assert memberships["seam"]["gt"][0].tolist() == [True, False]
    assert memberships["high_lat"]["gt"][0].tolist() == [True, False]
    assert memberships["seam_high_lat"]["gt"][0].tolist() == [True, False]
    assert memberships["seam"]["tracker"][0].tolist() == [True, False]


def test_rank_bad_case_sequences_uses_seam_and_high_lat_hota():
    per_sequence_metric_rows = [
        {"tracker": "proto", "subset": "full", "seq": "seq_a", "HOTA": 50.0, "AssA": 40.0, "IDF1": 45.0},
        {"tracker": "proto", "subset": "full", "seq": "seq_b", "HOTA": 55.0, "AssA": 42.0, "IDF1": 46.0},
        {"tracker": "proto", "subset": "seam", "seq": "seq_a", "HOTA": 20.0, "AssA": 15.0, "IDF1": 18.0},
        {"tracker": "proto", "subset": "seam", "seq": "seq_b", "HOTA": 40.0, "AssA": 30.0, "IDF1": 35.0},
        {"tracker": "proto", "subset": "high_lat", "seq": "seq_a", "HOTA": 25.0, "AssA": 20.0, "IDF1": 22.0},
        {"tracker": "proto", "subset": "high_lat", "seq": "seq_b", "HOTA": 35.0, "AssA": 28.0, "IDF1": 32.0},
        {"tracker": "proto", "subset": "seam_high_lat", "seq": "seq_a", "HOTA": 15.0, "AssA": 10.0, "IDF1": 12.0},
        {"tracker": "proto", "subset": "seam_high_lat", "seq": "seq_b", "HOTA": 30.0, "AssA": 25.0, "IDF1": 28.0},
    ]
    coverage_rows = [
        {"tracker": "proto", "subset": "full", "seq": "seq_a", "gt_dets": 10, "tracker_dets": 10, "gt_ids": 5, "tracker_ids": 5, "gt_fraction_of_full": 1.0, "tracker_fraction_of_full": 1.0},
        {"tracker": "proto", "subset": "full", "seq": "seq_b", "gt_dets": 10, "tracker_dets": 10, "gt_ids": 5, "tracker_ids": 5, "gt_fraction_of_full": 1.0, "tracker_fraction_of_full": 1.0},
        {"tracker": "proto", "subset": "seam", "seq": "seq_a", "gt_dets": 4, "tracker_dets": 4, "gt_ids": 2, "tracker_ids": 2, "gt_fraction_of_full": 0.4, "tracker_fraction_of_full": 0.4},
        {"tracker": "proto", "subset": "seam", "seq": "seq_b", "gt_dets": 4, "tracker_dets": 4, "gt_ids": 2, "tracker_ids": 2, "gt_fraction_of_full": 0.4, "tracker_fraction_of_full": 0.4},
        {"tracker": "proto", "subset": "high_lat", "seq": "seq_a", "gt_dets": 3, "tracker_dets": 3, "gt_ids": 2, "tracker_ids": 2, "gt_fraction_of_full": 0.3, "tracker_fraction_of_full": 0.3},
        {"tracker": "proto", "subset": "high_lat", "seq": "seq_b", "gt_dets": 3, "tracker_dets": 3, "gt_ids": 2, "tracker_ids": 2, "gt_fraction_of_full": 0.3, "tracker_fraction_of_full": 0.3},
        {"tracker": "proto", "subset": "seam_high_lat", "seq": "seq_a", "gt_dets": 2, "tracker_dets": 2, "gt_ids": 1, "tracker_ids": 1, "gt_fraction_of_full": 0.2, "tracker_fraction_of_full": 0.2},
        {"tracker": "proto", "subset": "seam_high_lat", "seq": "seq_b", "gt_dets": 2, "tracker_dets": 2, "gt_ids": 1, "tracker_ids": 1, "gt_fraction_of_full": 0.2, "tracker_fraction_of_full": 0.2},
    ]

    ranking = DIAG.rank_bad_case_sequences(
        per_sequence_metric_rows=per_sequence_metric_rows,
        coverage_rows=coverage_rows,
        tracker_name="proto",
        top_k=1,
    )

    assert ranking[0]["seq"] == "seq_a"
    assert bool(ranking[0]["selected"]) is True
    assert ranking[1]["seq"] == "seq_b"
