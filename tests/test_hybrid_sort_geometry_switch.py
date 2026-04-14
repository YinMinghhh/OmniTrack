import argparse
import importlib.util
import pathlib
import sys
import types

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_ROOT = "tree_b_hybrid_pkg.models.trackers.hybrid_sort_tracker"


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


ensure_package("tree_b_hybrid_pkg")
ensure_package("tree_b_hybrid_pkg.models")
ensure_package("tree_b_hybrid_pkg.models.trackers")
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
load_module(
    "association",
    "projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/association.py",
)
HYBRID_SORT_MODULE = load_module(
    "hybrid_sort",
    "projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/hybrid_sort.py",
)


class DummyKalmanBoxTracker:
    count = 0

    def __init__(self, bbox, delta_t=3, orig=False, args=None):
        self.id = DummyKalmanBoxTracker.count
        DummyKalmanBoxTracker.count += 1
        self.args = args
        self.delta_t = delta_t
        self.last_observation = np.asarray(bbox, dtype=float).copy()
        self.observations = {0: self.last_observation.copy()}
        self.age = 0
        self.time_since_update = 0
        self.hit_streak = 1
        self.velocity_lt = np.zeros((2,), dtype=float)
        self.velocity_rt = np.zeros((2,), dtype=float)
        self.velocity_lb = np.zeros((2,), dtype=float)
        self.velocity_rb = np.zeros((2,), dtype=float)

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        return (
            self.last_observation[:4].reshape(1, 4),
            np.array([self.args.track_thresh], dtype=float),
            float(self.last_observation[4]),
        )

    def update(self, bbox):
        if bbox is not None:
            self.last_observation = np.asarray(bbox, dtype=float).copy()
            self.observations[self.age] = self.last_observation.copy()
            self.time_since_update = 0
            self.hit_streak += 1

    def get_state(self):
        return self.last_observation[:4].reshape(1, 4)


HYBRID_SORT_MODULE.KalmanBoxTracker = DummyKalmanBoxTracker


def make_args():
    return argparse.Namespace(
        track_thresh=0.6,
        iou_thresh=0.15,
        asso="Height_Modulated_IoU",
        use_byte=True,
        inertia=0.05,
        deltat=3,
        TCM_first_step=True,
        TCM_byte_step=True,
        TCM_first_step_weight=1.0,
        TCM_byte_step_weight=1.0,
    )


def test_planar_legacy_keeps_legacy_matching_path(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("associate_spherical should not run in planar_legacy mode")

    monkeypatch.setattr(HYBRID_SORT_MODULE, "associate_spherical", fail_if_called)

    tracker = HYBRID_SORT_MODULE.Hybrid_Sort(
        make_args(),
        det_thresh=0.6,
        min_hits=1,
        association_geometry_cfg={"mode": "planar_legacy"},
    )
    dets = np.array([[10.0, 10.0, 50.0, 100.0, 0.9]], dtype=float)

    tracker.update(dets)
    results, _ = tracker.update(dets)

    assert results.shape[0] >= 1
    assert results[0, 4] == 1


def test_spherical_mode_switches_to_spherical_association(monkeypatch):
    call_state = {"called": False}

    def fake_associate_spherical(detections, trackers, *args, **kwargs):
        call_state["called"] = True
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.arange(len(trackers)),
        )

    monkeypatch.setattr(HYBRID_SORT_MODULE, "associate_spherical", fake_associate_spherical)

    tracker = HYBRID_SORT_MODULE.Hybrid_Sort(
        make_args(),
        det_thresh=0.6,
        min_hits=1,
        association_geometry_cfg={
            "mode": "bfov_lite_spherical",
            "image_width": 3760.0,
            "image_height": 480.0,
            "gate_threshold": 0.15,
        },
    )
    dets = np.array([[10.0, 10.0, 50.0, 100.0, 0.9]], dtype=float)

    tracker.update(dets)
    tracker.update(dets)

    assert call_state["called"] is True


def test_selective_spherical_mode_uses_mixed_geometry_path(monkeypatch):
    call_state = {"called": False}

    def fake_associate_selective_spherical(detections, trackers, *args, **kwargs):
        call_state["called"] = True
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.arange(len(trackers)),
        )

    monkeypatch.setattr(
        HYBRID_SORT_MODULE,
        "associate_selective_spherical",
        fake_associate_selective_spherical,
    )

    tracker = HYBRID_SORT_MODULE.Hybrid_Sort(
        make_args(),
        det_thresh=0.6,
        min_hits=1,
        association_geometry_cfg={
            "mode": "selective_spherical",
            "image_width": 3760.0,
            "image_height": 480.0,
            "gate_threshold": 0.15,
            "seam_band_px": 400.0,
        },
    )
    dets = np.array([[10.0, 10.0, 50.0, 100.0, 0.9]], dtype=float)

    tracker.update(dets)
    tracker.update(dets)

    assert call_state["called"] is True


def test_planar_gate_spherical_rerank_mode_uses_mixed_geometry_path(monkeypatch):
    call_state = {"called": False}

    def fake_associate_planar_gate_spherical_rerank(detections, trackers, *args, **kwargs):
        call_state["called"] = True
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.arange(len(trackers)),
        )

    monkeypatch.setattr(
        HYBRID_SORT_MODULE,
        "associate_planar_gate_spherical_rerank",
        fake_associate_planar_gate_spherical_rerank,
    )

    tracker = HYBRID_SORT_MODULE.Hybrid_Sort(
        make_args(),
        det_thresh=0.6,
        min_hits=1,
        association_geometry_cfg={
            "mode": "planar_gate_spherical_rerank",
            "image_width": 3760.0,
            "image_height": 480.0,
            "gate_threshold": 0.15,
            "seam_band_px": 400.0,
        },
    )
    dets = np.array([[10.0, 10.0, 50.0, 100.0, 0.9]], dtype=float)

    tracker.update(dets)
    tracker.update(dets)

    assert call_state["called"] is True
