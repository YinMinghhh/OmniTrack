import importlib.util
import pathlib
import sys
import types

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_ROOT = "tree_b_mixed_assoc_pkg.models.trackers.hybrid_sort_tracker"


def ensure_package(name):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def load_module(module_name, relative_path):
    full_name = f"{PKG_ROOT}.{module_name}"
    spec = importlib.util.spec_from_file_location(
        full_name,
        REPO_ROOT / relative_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


ensure_package("tree_b_mixed_assoc_pkg")
ensure_package("tree_b_mixed_assoc_pkg.models")
ensure_package("tree_b_mixed_assoc_pkg.models.trackers")
ensure_package(PKG_ROOT)
lap_module = types.ModuleType("lap")
lap_module.lapjv = lambda cost_matrix, extend_cost=True, cost_limit=None: (
    0.0,
    np.full(cost_matrix.shape[0], -1, dtype=int),
    np.full(cost_matrix.shape[1], -1, dtype=int),
)
sys.modules["lap"] = lap_module

load_module(
    "association_geometry",
    "projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/association_geometry.py",
)
ASSOCIATION = load_module(
    "association",
    "projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/association.py",
)


def test_associate_selective_spherical_mixes_pairwise_geometry(monkeypatch):
    monkeypatch.setattr(
        ASSOCIATION,
        "selective_spherical_pair_mask_xyxy",
        lambda *args, **kwargs: np.array([[True, False], [False, False]], dtype=bool),
    )
    monkeypatch.setattr(
        ASSOCIATION,
        "spherical_gate_and_score_matrices",
        lambda *args, **kwargs: (
            np.array([[0.8, 0.0], [0.0, 0.0]], dtype=float),
            np.array([[0.95, 0.0], [0.0, 0.0]], dtype=float),
        ),
    )

    matches, unmatched_dets, unmatched_trks = ASSOCIATION.associate_selective_spherical(
        np.zeros((2, 5), dtype=float),
        np.zeros((2, 6), dtype=float),
        planar_gate_matrix=np.array([[0.0, 0.0], [0.0, 0.7]], dtype=float),
        planar_score_matrix=np.array([[0.0, 0.0], [0.0, 0.8]], dtype=float),
        planar_overlap_threshold=0.5,
        spherical_overlap_threshold=0.5,
        association_geometry_cfg={
            "image_width": 3760.0,
            "image_height": 480.0,
            "center_distance_weight": 0.05,
            "high_lat_deg": 45.0,
            "seam_band_px": 400.0,
        },
    )

    assert matches.tolist() == [[0, 0], [1, 1]]
    assert unmatched_dets.tolist() == []
    assert unmatched_trks.tolist() == []


def test_planar_gate_spherical_rerank_cannot_revive_planar_rejected_pair(monkeypatch):
    monkeypatch.setattr(
        ASSOCIATION,
        "spherical_gate_and_score_matrices",
        lambda *args, **kwargs: (
            np.array([[0.9, 0.9]], dtype=float),
            np.array([[0.99, 0.2]], dtype=float),
        ),
    )

    matches, unmatched_dets, unmatched_trks = (
        ASSOCIATION.associate_planar_gate_spherical_rerank(
            np.zeros((1, 5), dtype=float),
            np.zeros((2, 6), dtype=float),
            planar_gate_matrix=np.array([[0.0, 0.6]], dtype=float),
            planar_overlap_threshold=0.5,
            association_geometry_cfg={
                "image_width": 3760.0,
                "image_height": 480.0,
                "center_distance_weight": 0.05,
                "high_lat_deg": 45.0,
                "seam_band_px": 400.0,
            },
        )
    )

    assert matches.tolist() == [[0, 1]]
    assert unmatched_dets.tolist() == []
    assert unmatched_trks.tolist() == [0]
