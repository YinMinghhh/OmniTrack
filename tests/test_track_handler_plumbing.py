import argparse
import importlib.util
import pathlib
import sys
import types

import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_ROOT = "tree_b_trackhandler_pkg.models"


def ensure_package(name):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def register_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


ensure_package("tree_b_trackhandler_pkg")
ensure_package(PKG_ROOT)
ensure_package(f"{PKG_ROOT}.omnidetr")
ensure_package(f"{PKG_ROOT}.track")
ensure_package(f"{PKG_ROOT}.trackers")
ensure_package(f"{PKG_ROOT}.trackers.ocsort_tracker")
ensure_package(f"{PKG_ROOT}.trackers.hybrid_sort_tracker")
ensure_package(f"{PKG_ROOT}.trackers.sort_tracker")
ensure_package(f"{PKG_ROOT}.trackers.byte_tracker")

register_module(f"{PKG_ROOT}.omnidetr.ops", xywh2xyxy=lambda x: x, xyxy2xywh=lambda x: x)
register_module(
    f"{PKG_ROOT}.omnidetr.active_track_utils",
    collect_active_track_data=lambda *args, **kwargs: {"boxes": None, "retained": [], "dropped": []},
)
register_module(
    f"{PKG_ROOT}.omnidetr.seam_duplicate_resolver",
    resolve_seam_duplicates_xyxy=lambda *args, **kwargs: args[:4] + ({},),
)
register_module(f"{PKG_ROOT}.track.strack", STrack=type("DummySTrack", (), {}))
register_module(
    f"{PKG_ROOT}.track.kalman_filter",
    KalmanFilter=type("DummyKalmanFilter", (), {}),
)
register_module(
    f"{PKG_ROOT}.track.matching",
    iou_distance=lambda *args, **kwargs: None,
    iou_score=lambda *args, **kwargs: None,
)
register_module(
    f"{PKG_ROOT}.trackers.ocsort_tracker.ocsort",
    OCSort=type("DummyOCSort", (), {}),
)
register_module(
    f"{PKG_ROOT}.trackers.sort_tracker.sort",
    Sort=type("DummySort", (), {}),
)
register_module(
    f"{PKG_ROOT}.trackers.byte_tracker.byte_tracker",
    BYTETracker=type("DummyBYTETracker", (), {}),
)


class DummyHybridSort:
    calls = []
    runtime_geometry_calls = []

    def __init__(self, args, **kwargs):
        DummyHybridSort.calls.append(
            {
                "track_thresh": args.track_thresh,
                "iou_thresh": args.iou_thresh,
                "kwargs": dict(kwargs),
            }
        )

    def set_runtime_geometry(self, **kwargs):
        DummyHybridSort.runtime_geometry_calls.append(dict(kwargs))

    def update(self, dets):
        return np.empty((0, 5), dtype=float), []


def make_parser():
    class _Parser:
        def parse_args(self, _argv):
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

    return _Parser()


def build_hybrid_sort_args(args, tbd_tracker_cfg):
    args.track_thresh = tbd_tracker_cfg["track_thresh"]
    args.iou_thresh = tbd_tracker_cfg["iou_thresh"]
    return args, {
        "det_thresh": tbd_tracker_cfg["track_thresh"],
        "max_age": 30,
        "min_hits": 3,
        "association_geometry_cfg": dict(tbd_tracker_cfg["association_geometry"]),
    }


register_module(
    f"{PKG_ROOT}.trackers.hybrid_sort_tracker.hybrid_sort",
    Hybrid_Sort=DummyHybridSort,
)
register_module(
    f"{PKG_ROOT}.trackers.hybrid_sort_tracker.tracker_builder",
    build_hybrid_sort_args=build_hybrid_sort_args,
)
register_module(f"{PKG_ROOT}.trackers.args", make_parser=make_parser)

torchvision_pkg = ensure_package("torchvision")
ensure_package("torchvision.ops")
register_module("torchvision.ops", nms=lambda *args, **kwargs: None)
torchvision_pkg.ops = sys.modules["torchvision.ops"]


MODULE_PATH = (
    REPO_ROOT / "projects/mmdet3d_plugin/models/omnidetr/track_handler_module.py"
)
SPEC = importlib.util.spec_from_file_location(
    f"{PKG_ROOT}.omnidetr.track_handler_module",
    MODULE_PATH,
)
TRACK_HANDLER_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACK_HANDLER_MODULE
SPEC.loader.exec_module(TRACK_HANDLER_MODULE)


class DummyBank:
    def __init__(self):
        self.tbd_handler_cfg = {}
        self.tbd_tracker_cfg = {
            "track_thresh": 0.72,
            "iou_thresh": 0.2,
            "association_geometry": {
                "mode": "bfov_lite_spherical",
                "image_width": 3760.0,
                "image_height": 480.0,
                "gate_threshold": 0.2,
            },
        }
        self.seam_resolver_cfg = {
            "enabled": False,
            "seam_band_px": 400.0,
            "active_track_max_time_since_update": None,
            "debug_stats": False,
        }
        self.nms_thresh = 0.05
        self.track_thresh = 0.45
        self.det_thresh = 0.10
        self.init_thresh = 0.55
        self.timestamp = None


def test_track_handler_init_and_reset_use_same_tracker_builder():
    DummyHybridSort.calls.clear()
    handler = TRACK_HANDLER_MODULE.TrackHandler(DummyBank())
    handler._reset_tracker()

    assert len(DummyHybridSort.calls) == 2
    assert DummyHybridSort.calls[0] == DummyHybridSort.calls[1]
    assert DummyHybridSort.calls[0]["track_thresh"] == 0.72
    assert DummyHybridSort.calls[0]["iou_thresh"] == 0.2
    assert (
        DummyHybridSort.calls[0]["kwargs"]["association_geometry_cfg"]["mode"]
        == "bfov_lite_spherical"
    )


def test_query_handler_injects_runtime_seam_band_into_tracker(monkeypatch):
    DummyHybridSort.calls.clear()
    DummyHybridSort.runtime_geometry_calls.clear()

    class DummySTrack:
        @staticmethod
        def cxcywh_to_tlbr_to_tensor(box):
            return box

    monkeypatch.setattr(TRACK_HANDLER_MODULE, "STrack", DummySTrack)
    monkeypatch.setattr(
        TRACK_HANDLER_MODULE,
        "nms",
        lambda boxes, scores, iou_threshold: torch.arange(
            boxes.shape[0], device=boxes.device
        ),
    )

    handler = TRACK_HANDLER_MODULE.TrackHandler(DummyBank())
    bbox = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    score = torch.tensor([[0.9]], dtype=torch.float32)
    meta = {
        "image_wh": torch.tensor([[[3760.0, 480.0]]], dtype=torch.float32),
        "ori_shape": [
            torch.tensor([480.0], dtype=torch.float32),
            torch.tensor([3760.0], dtype=torch.float32),
        ],
        "timestamp": 0,
    }

    handler.query_handler(bbox, score, meta, qt=None)

    assert DummyHybridSort.runtime_geometry_calls == [
        {"image_width": 3760.0, "image_height": 480.0, "seam_band_px": 400.0}
    ]
