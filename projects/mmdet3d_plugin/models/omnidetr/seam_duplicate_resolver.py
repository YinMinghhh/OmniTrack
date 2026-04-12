import torch

try:
    from ..trackers.hybrid_sort_tracker.circular_utils import (
        unwrap_to_track,
        wrap_metric_batch,
    )
except ImportError:
    import importlib.util
    import pathlib

    _CIRCULAR_UTILS_PATH = (
        pathlib.Path(__file__).resolve().parents[1]
        / "trackers/hybrid_sort_tracker/circular_utils.py"
    )
    _CIRCULAR_UTILS_SPEC = importlib.util.spec_from_file_location(
        "hybrid_sort_circular_utils",
        _CIRCULAR_UTILS_PATH,
    )
    _CIRCULAR_UTILS = importlib.util.module_from_spec(_CIRCULAR_UTILS_SPEC)
    _CIRCULAR_UTILS_SPEC.loader.exec_module(_CIRCULAR_UTILS)
    unwrap_to_track = _CIRCULAR_UTILS.unwrap_to_track
    wrap_metric_batch = _CIRCULAR_UTILS.wrap_metric_batch


DEFAULT_SEAM_RESOLVER_CFG = dict(
    enabled=True,
    geometry="wrap_xyxy",
    seam_band_px=400,
    match_iou=0.5,
    track_compat_iou=0.7,
    active_track_max_time_since_update=None,
    class_strict=True,
    fuse_method="score_weighted_box_max_score",
    debug_stats=False,
)


def normalize_seam_resolver_cfg(cfg=None):
    resolved = dict(DEFAULT_SEAM_RESOLVER_CFG)
    if cfg is not None:
        resolved.update(cfg)

    if resolved["geometry"] != "wrap_xyxy":
        raise ValueError(
            f"Unsupported seam resolver geometry={resolved['geometry']!r}. "
            "Only 'wrap_xyxy' is implemented in v1."
        )
    if resolved["fuse_method"] != "score_weighted_box_max_score":
        raise ValueError(
            f"Unsupported seam resolver fuse_method={resolved['fuse_method']!r}. "
            "Only 'score_weighted_box_max_score' is implemented in v1."
        )
    max_time_since_update = resolved.get("active_track_max_time_since_update")
    if max_time_since_update is not None and max_time_since_update < 0:
        raise ValueError(
            "active_track_max_time_since_update must be None or >= 0."
        )
    return resolved


def _as_boxes_tensor(boxes, *, device=None, dtype=None):
    if boxes is None:
        return None
    if torch.is_tensor(boxes):
        tensor = boxes
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device or tensor.device, dtype=dtype or tensor.dtype)
    else:
        tensor = torch.as_tensor(boxes, device=device, dtype=dtype)
    if tensor.numel() == 0:
        return tensor.reshape(0, 4)
    return tensor.reshape(-1, 4)


def _as_vector(values, *, length, device, dtype, default=None):
    if values is None:
        if default is None:
            return None
        return torch.full((length,), default, device=device, dtype=dtype)

    if torch.is_tensor(values):
        vector = values.to(device=device)
    else:
        vector = torch.as_tensor(values, device=device)
    vector = vector.reshape(-1)
    if vector.numel() != length:
        raise ValueError(
            f"Expected vector with {length} values, got {vector.numel()}."
        )
    return vector.to(dtype=dtype)


def _box_iou_xyxy(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    union = area1[:, None] + area2[None, :] - intersection
    union = torch.where(union > 0, union, torch.ones_like(union))
    return intersection / union


def wrap_iou_matrix(boxes1, boxes2, image_width):
    boxes1 = _as_boxes_tensor(boxes1)
    boxes2 = _as_boxes_tensor(boxes2, device=boxes1.device, dtype=boxes1.dtype)
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    return wrap_metric_batch(
        _box_iou_xyxy,
        boxes1,
        boxes2,
        float(image_width),
    )


def wrap_iou(box_a, box_b, image_width):
    return wrap_iou_matrix(
        _as_boxes_tensor(box_a),
        _as_boxes_tensor(box_b),
        image_width,
    ).reshape(-1)[0]


def _canonicalize_xyxy(boxes, image_width):
    if boxes.numel() == 0:
        return boxes
    canonical = boxes.clone()
    shifts = torch.floor(canonical[:, 0] / float(image_width))
    canonical[:, [0, 2]] -= shifts[:, None] * float(image_width)
    return canonical


def _seam_candidate_mask(boxes, image_width, seam_band_px):
    return (
        (boxes[:, 0] < seam_band_px)
        | (boxes[:, 2] > image_width - seam_band_px)
        | (boxes[:, 0] < 0)
        | (boxes[:, 2] > image_width)
    )


def _build_equivalence_group_ids(pairwise_iou, threshold):
    num_items = int(pairwise_iou.shape[0])
    if num_items == 0:
        return pairwise_iou.new_empty((0,), dtype=torch.long)

    parents = list(range(num_items))

    def find(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parents[root_b] = root_a

    for i in range(num_items):
        for j in range(i + 1, num_items):
            if pairwise_iou[i, j] >= threshold:
                union(i, j)

    root_to_group = {}
    group_ids = []
    for idx in range(num_items):
        root = find(idx)
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        group_ids.append(root_to_group[root])

    return torch.as_tensor(group_ids, device=pairwise_iou.device, dtype=torch.long)


def _active_track_group_ids(active_tracks, image_width, group_match_iou):
    if (
        active_tracks is None
        or active_tracks.numel() == 0
        or group_match_iou is None
        or group_match_iou <= 0
    ):
        return None

    pairwise_iou = wrap_iou_matrix(active_tracks, active_tracks, image_width)
    return _build_equivalence_group_ids(pairwise_iou, float(group_match_iou))


def _active_track_group_iou_threshold(match_iou):
    # Active tracks lag the current detections by one frame, so allow a small
    # tolerance when deciding whether two active tracks already represent the
    # same latent seam object.
    return max(float(match_iou) - 0.03, 0.0)


def _best_track_assignments(
    boxes,
    active_tracks,
    image_width,
    track_compat_iou,
    active_track_group_iou,
    return_debug=False,
):
    if (
        active_tracks is None
        or active_tracks.numel() == 0
        or track_compat_iou is None
        or track_compat_iou <= 0
    ):
        if return_debug:
            return None, {
                "active_track_group_ids": None,
                "best_track_indices": None,
                "best_track_ious": None,
                "best_track_group_assignments": None,
            }
        return None

    active_group_ids = _active_track_group_ids(
        active_tracks,
        image_width,
        active_track_group_iou,
    )
    compat = wrap_iou_matrix(boxes, active_tracks, image_width)
    best_iou, best_idx = compat.max(dim=1)
    best_group_idx = active_group_ids[best_idx.to(dtype=torch.long)]
    invalid = best_iou < track_compat_iou
    best_group_idx[invalid] = -1
    if return_debug:
        return best_group_idx, {
            "active_track_group_ids": active_group_ids.detach().cpu().tolist(),
            "best_track_indices": best_idx.detach().cpu().tolist(),
            "best_track_ious": best_iou.detach().cpu().tolist(),
            "best_track_group_assignments": best_group_idx.detach().cpu().tolist(),
        }
    return best_group_idx


def _best_aligned_box(box, reference, image_width):
    aligned = unwrap_to_track(
        box.reshape(-1),
        reference.reshape(-1),
        float(image_width),
    )
    return aligned.reshape(-1)


def resolve_seam_duplicates_xyxy(
    boxes_xyxy,
    scores,
    labels,
    *,
    image_width,
    seam_resolver_cfg=None,
    qualities=None,
    active_tracks=None,
):
    cfg = normalize_seam_resolver_cfg(seam_resolver_cfg)
    boxes = _as_boxes_tensor(boxes_xyxy)
    if boxes.numel() == 0:
        empty_scores = _as_vector(
            scores,
            length=0,
            device=boxes.device,
            dtype=boxes.dtype,
            default=0.0,
        )
        empty_labels = _as_vector(
            labels,
            length=0,
            device=boxes.device,
            dtype=torch.long,
            default=0,
        )
        empty_quality = _as_vector(
            qualities,
            length=0,
            device=boxes.device,
            dtype=boxes.dtype,
            default=0.0,
        ) if qualities is not None else None
        return boxes, empty_scores, empty_labels, empty_quality, {
            "enabled": cfg["enabled"],
            "input_count": 0,
            "output_count": 0,
            "seam_candidate_count": 0,
            "equivalence_classes": 0,
            "merged_candidates": 0,
        }

    scores = _as_vector(
        scores,
        length=boxes.shape[0],
        device=boxes.device,
        dtype=boxes.dtype,
    )
    labels = _as_vector(
        labels,
        length=boxes.shape[0],
        device=boxes.device,
        dtype=torch.long,
        default=0,
    )
    qualities = _as_vector(
        qualities,
        length=boxes.shape[0],
        device=boxes.device,
        dtype=boxes.dtype,
    ) if qualities is not None else None
    active_tracks = _as_boxes_tensor(
        active_tracks,
        device=boxes.device,
        dtype=boxes.dtype,
    )

    if not cfg["enabled"]:
        return boxes, scores, labels, qualities, {
            "enabled": False,
            "input_count": int(boxes.shape[0]),
            "output_count": int(boxes.shape[0]),
            "seam_candidate_count": 0,
            "equivalence_classes": 0,
            "merged_candidates": 0,
        }

    seam_mask = _seam_candidate_mask(
        boxes, float(image_width), float(cfg["seam_band_px"])
    )
    seam_indices = torch.nonzero(seam_mask, as_tuple=False).reshape(-1)
    non_seam_indices = torch.nonzero(~seam_mask, as_tuple=False).reshape(-1)

    if seam_indices.numel() == 0:
        return boxes, scores, labels, qualities, {
            "enabled": True,
            "input_count": int(boxes.shape[0]),
            "output_count": int(boxes.shape[0]),
            "seam_candidate_count": 0,
            "equivalence_classes": 0,
            "merged_candidates": 0,
        }

    seam_boxes = boxes[seam_indices]
    seam_scores = scores[seam_indices]
    seam_labels = labels[seam_indices]
    seam_qualities = qualities[seam_indices] if qualities is not None else None

    pairwise_iou = wrap_iou_matrix(seam_boxes, seam_boxes, float(image_width))
    if cfg["debug_stats"]:
        track_assignments, track_debug = _best_track_assignments(
            seam_boxes,
            active_tracks,
            float(image_width),
            float(cfg["track_compat_iou"]),
            _active_track_group_iou_threshold(cfg["match_iou"]),
            return_debug=True,
        )
    else:
        track_assignments = _best_track_assignments(
            seam_boxes,
            active_tracks,
            float(image_width),
            float(cfg["track_compat_iou"]),
            _active_track_group_iou_threshold(cfg["match_iou"]),
        )
        track_debug = None

    parents = list(range(seam_boxes.shape[0]))

    def find(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parents[root_b] = root_a

    for i in range(seam_boxes.shape[0]):
        for j in range(i + 1, seam_boxes.shape[0]):
            if cfg["class_strict"] and seam_labels[i] != seam_labels[j]:
                continue
            if (
                track_assignments is not None
                and track_assignments[i] >= 0
                and track_assignments[j] >= 0
                and track_assignments[i] != track_assignments[j]
            ):
                continue
            if pairwise_iou[i, j] >= float(cfg["match_iou"]):
                union(i, j)

    groups = {}
    for local_idx in range(seam_boxes.shape[0]):
        groups.setdefault(find(local_idx), []).append(local_idx)

    fused_items = []
    for index in non_seam_indices.tolist():
        fused_items.append(
            (
                int(index),
                boxes[index],
                scores[index],
                labels[index],
                qualities[index] if qualities is not None else None,
            )
        )

    for group in groups.values():
        group_tensor = torch.as_tensor(group, device=boxes.device, dtype=torch.long)
        group_boxes = seam_boxes[group_tensor]
        group_scores = seam_scores[group_tensor]
        group_labels = seam_labels[group_tensor]
        group_qualities = seam_qualities[group_tensor] if seam_qualities is not None else None
        reference_idx = int(torch.argmax(group_scores).item())

        if group_boxes.shape[0] == 1:
            fused_box = _canonicalize_xyxy(group_boxes, float(image_width))[0]
        else:
            reference = group_boxes[reference_idx]
            aligned_boxes = torch.stack(
                [
                    _best_aligned_box(box.unsqueeze(0), reference, float(image_width))
                    for box in group_boxes
                ],
                dim=0,
            )
            weights = group_scores.clamp(min=1e-6)
            fused_box = (aligned_boxes * weights[:, None]).sum(dim=0) / weights.sum()
            fused_box = _canonicalize_xyxy(
                fused_box.unsqueeze(0), float(image_width)
            )[0]

        first_index = int(seam_indices[group_tensor].min().item())
        fused_items.append(
            (
                first_index,
                fused_box,
                group_scores[reference_idx],
                group_labels[reference_idx],
                group_qualities[reference_idx] if group_qualities is not None else None,
            )
        )

    fused_items.sort(key=lambda item: item[0])
    out_boxes = torch.stack([item[1] for item in fused_items], dim=0)
    out_scores = torch.stack([item[2] for item in fused_items], dim=0)
    out_labels = torch.stack([item[3] for item in fused_items], dim=0)
    out_qualities = (
        torch.stack([item[4] for item in fused_items], dim=0)
        if qualities is not None
        else None
    )

    stats = {
        "enabled": True,
        "input_count": int(boxes.shape[0]),
        "output_count": int(out_boxes.shape[0]),
        "seam_candidate_count": int(seam_indices.numel()),
        "equivalence_classes": int(len(groups)),
        "merged_candidates": int(seam_indices.numel() - len(groups)),
    }
    if cfg["debug_stats"]:
        stats.update(
            {
                "image_width": float(image_width),
                "seam_band_px": float(cfg["seam_band_px"]),
                "match_iou": float(cfg["match_iou"]),
                "track_compat_iou": float(cfg["track_compat_iou"]),
                "class_strict": bool(cfg["class_strict"]),
                "seam_indices": seam_indices.detach().cpu().tolist(),
                "non_seam_indices": non_seam_indices.detach().cpu().tolist(),
                "seam_boxes": seam_boxes.detach().cpu().tolist(),
                "seam_scores": seam_scores.detach().cpu().tolist(),
                "seam_labels": seam_labels.detach().cpu().tolist(),
                "pairwise_wrap_iou": pairwise_iou.detach().cpu().tolist(),
                "groups": [
                    {
                        "local_indices": [int(idx) for idx in group],
                        "global_indices": [
                            int(seam_indices[idx].item()) for idx in group
                        ],
                    }
                    for group in groups.values()
                ],
                "track_debug": track_debug,
            }
        )
    return out_boxes, out_scores, out_labels, out_qualities, stats


__all__ = [
    "DEFAULT_SEAM_RESOLVER_CFG",
    "normalize_seam_resolver_cfg",
    "resolve_seam_duplicates_xyxy",
    "wrap_iou",
    "wrap_iou_matrix",
]
