import argparse
import importlib.util
import pathlib
import sys
import types


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_ROOT = "tree_b_testpkg.models.trackers.hybrid_sort_tracker"


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


ensure_package("tree_b_testpkg")
ensure_package("tree_b_testpkg.models")
ensure_package("tree_b_testpkg.models.trackers")
ensure_package(PKG_ROOT)

load_module(
    "association_geometry",
    "projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/association_geometry.py",
)
TRACKER_BUILDER = load_module(
    "tracker_builder",
    "projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/tracker_builder.py",
)


def test_tbd_tracker_cfg_overrides_parser_defaults():
    args = argparse.Namespace(
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

    args, hybrid_sort_kwargs = TRACKER_BUILDER.build_hybrid_sort_args(
        args,
        {
            "track_thresh": 0.72,
            "iou_thresh": 0.22,
            "association_geometry": {
                "mode": "bfov_lite_spherical",
                "center_distance_weight": 0.2,
            },
        },
    )

    assert args.track_thresh == 0.72
    assert args.iou_thresh == 0.22
    assert hybrid_sort_kwargs["det_thresh"] == 0.72
    assert hybrid_sort_kwargs["association_geometry_cfg"]["mode"] == "bfov_lite_spherical"
    assert hybrid_sort_kwargs["association_geometry_cfg"]["center_distance_weight"] == 0.2
    assert hybrid_sort_kwargs["association_geometry_cfg"]["high_lat_deg"] == 45.0


def test_builder_accepts_explicit_hybrid_sort_constructor_overrides():
    args = argparse.Namespace(
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

    _, hybrid_sort_kwargs = TRACKER_BUILDER.build_hybrid_sort_args(
        args,
        {
            "det_thresh": 0.4,
            "max_age": 9,
            "min_hits": 2,
        },
    )

    assert hybrid_sort_kwargs["det_thresh"] == 0.4
    assert hybrid_sort_kwargs["max_age"] == 9
    assert hybrid_sort_kwargs["min_hits"] == 2
