from .association_geometry import normalize_tbd_tracker_cfg


def build_hybrid_sort_args(base_args, tbd_tracker_cfg=None):
    tracker_cfg = normalize_tbd_tracker_cfg(tbd_tracker_cfg)
    association_geometry_cfg = dict(tracker_cfg.pop("association_geometry"))

    hybrid_sort_extra = {}
    for key in ("det_thresh", "max_age", "min_hits"):
        if key in tracker_cfg:
            hybrid_sort_extra[key] = tracker_cfg.pop(key)

    args_namespace = vars(base_args)
    unknown_keys = sorted(set(tracker_cfg) - set(args_namespace))
    if unknown_keys:
        raise KeyError(
            "Unsupported tbd_tracker_cfg keys for Hybrid_Sort: "
            + ", ".join(unknown_keys)
        )

    for key, value in tracker_cfg.items():
        args_namespace[key] = value

    return base_args, {
        "det_thresh": float(hybrid_sort_extra.get("det_thresh", base_args.track_thresh)),
        "max_age": int(hybrid_sort_extra.get("max_age", 30)),
        "min_hits": int(hybrid_sort_extra.get("min_hits", 3)),
        "association_geometry_cfg": association_geometry_cfg,
    }


__all__ = ["build_hybrid_sort_args"]
