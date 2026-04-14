import importlib.util
import pathlib

import numpy as np


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/association_geometry.py"
)
SPEC = importlib.util.spec_from_file_location("association_geometry", MODULE_PATH)
GEOMETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEOMETRY)


def test_spherical_iou_wraps_seam_crossing_boxes():
    boxes_left = np.array([[0.0, 10.0, 100.0, 110.0]], dtype=float)
    boxes_right = np.array([[3760.0, 10.0, 3860.0, 110.0]], dtype=float)

    iou = GEOMETRY.spherical_iou_xyxy_matrix(
        boxes_left,
        boxes_right,
        image_width=3760.0,
        image_height=480.0,
    )

    assert iou.shape == (1, 1)
    assert np.isclose(iou[0, 0], 1.0)


def test_great_circle_distance_is_symmetric():
    bfovs = GEOMETRY.xyxy_to_bfov_lite(
        np.array(
            [
                [100.0, 20.0, 140.0, 80.0],
                [3500.0, 30.0, 3580.0, 120.0],
            ],
            dtype=float,
        ),
        image_width=3760.0,
        image_height=480.0,
    )

    distances = GEOMETRY.great_circle_distance_bfov_matrix(bfovs, bfovs)

    assert distances.shape == (2, 2)
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0.0)


def test_high_lat_mask_uses_bfov_center_latitude():
    boxes_xywh = np.array(
        [
            [100.0, 0.0, 80.0, 60.0],
            [200.0, 210.0, 80.0, 60.0],
            [300.0, 420.0, 80.0, 60.0],
        ],
        dtype=float,
    )

    mask = GEOMETRY.high_lat_mask_xywh(
        boxes_xywh,
        image_width=3760.0,
        image_height=480.0,
        high_lat_deg=45.0,
    )

    assert mask.tolist() == [True, False, True]


def test_seam_mask_flags_wrap_and_near_edge_boxes():
    boxes_xywh = np.array(
        [
            [0.0, 100.0, 50.0, 60.0],
            [3300.0, 100.0, 200.0, 60.0],
            [1500.0, 100.0, 100.0, 60.0],
            [-20.0, 100.0, 60.0, 60.0],
        ],
        dtype=float,
    )

    mask = GEOMETRY.seam_mask_xywh(
        boxes_xywh,
        image_width=3760.0,
        seam_band_px=400.0,
    )

    assert mask.tolist() == [True, True, False, True]


def test_selective_spherical_pair_mask_uses_seam_or_high_lat_on_either_side():
    dets = np.array(
        [
            [0.0, 10.0, 100.0, 70.0],
            [1500.0, 210.0, 1600.0, 270.0],
        ],
        dtype=float,
    )
    trks = np.array(
        [
            [2000.0, 0.0, 2080.0, 50.0],
            [1800.0, 210.0, 1880.0, 270.0],
        ],
        dtype=float,
    )

    mask = GEOMETRY.selective_spherical_pair_mask_xyxy(
        dets,
        trks,
        image_width=3760.0,
        image_height=480.0,
        seam_band_px=400.0,
        high_lat_deg=45.0,
    )

    assert mask.tolist() == [[True, True], [True, False]]


def test_normalize_association_geometry_cfg_accepts_b2_modes():
    selective = GEOMETRY.normalize_association_geometry_cfg(
        {"mode": "selective_spherical"}
    )
    rerank = GEOMETRY.normalize_association_geometry_cfg(
        {"mode": "planar_gate_spherical_rerank"}
    )

    assert selective["mode"] == "selective_spherical"
    assert rerank["mode"] == "planar_gate_spherical_rerank"
    assert selective["seam_band_px"] == 400.0
