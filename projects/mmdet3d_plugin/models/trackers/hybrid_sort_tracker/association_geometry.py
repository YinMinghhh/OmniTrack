import math

import numpy as np


TAU = 2.0 * math.pi
HALF_PI = 0.5 * math.pi


DEFAULT_ASSOCIATION_GEOMETRY_CFG = {
    "mode": "planar_legacy",
    "image_width": 3760.0,
    "image_height": 480.0,
    "gate_threshold": None,
    "center_distance_weight": 0.05,
    "high_lat_deg": 45.0,
    "seam_band_px": 400.0,
}


def normalize_association_geometry_cfg(cfg=None):
    resolved = dict(DEFAULT_ASSOCIATION_GEOMETRY_CFG)
    if cfg is not None:
        resolved.update(dict(cfg))

    mode = str(resolved["mode"]).lower()
    if mode not in {
        "planar_legacy",
        "bfov_lite_spherical",
        "selective_spherical",
        "planar_gate_spherical_rerank",
    }:
        raise ValueError(
            f"Unsupported association geometry mode={resolved['mode']!r}."
        )
    resolved["mode"] = mode

    resolved["image_width"] = float(resolved["image_width"])
    resolved["image_height"] = float(resolved["image_height"])
    if resolved["image_width"] <= 0.0 or resolved["image_height"] <= 0.0:
        raise ValueError("image_width and image_height must be positive.")

    gate_threshold = resolved.get("gate_threshold")
    if gate_threshold is not None:
        gate_threshold = float(gate_threshold)
        if gate_threshold < 0.0 or gate_threshold > 1.0:
            raise ValueError("gate_threshold must be None or in [0, 1].")
    resolved["gate_threshold"] = gate_threshold

    resolved["center_distance_weight"] = float(resolved["center_distance_weight"])
    if resolved["center_distance_weight"] < 0.0:
        raise ValueError("center_distance_weight must be >= 0.")

    resolved["high_lat_deg"] = float(resolved["high_lat_deg"])
    if resolved["high_lat_deg"] < 0.0 or resolved["high_lat_deg"] > 90.0:
        raise ValueError("high_lat_deg must be in [0, 90].")

    resolved["seam_band_px"] = float(resolved["seam_band_px"])
    if resolved["seam_band_px"] < 0.0:
        raise ValueError("seam_band_px must be >= 0.")

    return resolved


def normalize_tbd_tracker_cfg(cfg=None):
    resolved = dict(cfg or {})
    resolved["association_geometry"] = normalize_association_geometry_cfg(
        resolved.get("association_geometry")
    )
    return resolved


def _as_xyxy_array(boxes):
    boxes = np.asarray(boxes, dtype=float)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    boxes = np.atleast_2d(boxes)
    if boxes.shape[1] < 4:
        raise ValueError(f"Expected boxes with at least 4 values, got {boxes.shape}.")
    return boxes[:, :4].copy()


def _as_xywh_array(boxes):
    boxes = np.asarray(boxes, dtype=float)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    boxes = np.atleast_2d(boxes)
    if boxes.shape[1] < 4:
        raise ValueError(f"Expected boxes with at least 4 values, got {boxes.shape}.")
    return boxes[:, :4].copy()


def xywh_to_xyxy(boxes_xywh):
    boxes_xywh = _as_xywh_array(boxes_xywh)
    if boxes_xywh.size == 0:
        return boxes_xywh.reshape(0, 4)
    boxes_xyxy = boxes_xywh.copy()
    boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2]
    boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3]
    return boxes_xyxy


def wrap_longitude_radians(longitude):
    longitude = np.asarray(longitude, dtype=float)
    return ((longitude + math.pi) % TAU) - math.pi


def xyxy_to_bfov_lite(boxes_xyxy, image_width, image_height):
    boxes_xyxy = _as_xyxy_array(boxes_xyxy)
    if boxes_xyxy.size == 0:
        return boxes_xyxy.reshape(0, 4)

    image_width = float(image_width)
    image_height = float(image_height)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]

    widths = np.clip(x2 - x1, a_min=0.0, a_max=image_width)
    heights = np.clip(y2 - y1, a_min=0.0, a_max=image_height)
    center_x = np.mod((x1 + x2) * 0.5, image_width)
    center_y = np.clip((y1 + y2) * 0.5, a_min=0.0, a_max=image_height)

    longitude = wrap_longitude_radians((center_x / image_width) * TAU - math.pi)
    latitude = HALF_PI - (center_y / image_height) * math.pi
    angular_width = np.clip((widths / image_width) * TAU, a_min=0.0, a_max=TAU)
    angular_height = np.clip(
        (heights / image_height) * math.pi,
        a_min=0.0,
        a_max=math.pi,
    )
    return np.stack(
        [longitude, latitude, angular_width, angular_height],
        axis=1,
    )


def xywh_to_bfov_lite(boxes_xywh, image_width, image_height):
    return xyxy_to_bfov_lite(
        xywh_to_xyxy(boxes_xywh),
        image_width=image_width,
        image_height=image_height,
    )


def _bfov_bounds(bfovs):
    bfovs = np.asarray(bfovs, dtype=float).reshape(-1, 4)
    lon = bfovs[:, 0]
    lat = bfovs[:, 1]
    half_w = bfovs[:, 2] * 0.5
    half_h = bfovs[:, 3] * 0.5
    lat_min = np.clip(lat - half_h, a_min=-HALF_PI, a_max=HALF_PI)
    lat_max = np.clip(lat + half_h, a_min=-HALF_PI, a_max=HALF_PI)
    return lon, half_w, lat_min, lat_max


def spherical_rectangle_area(bfovs):
    bfovs = np.asarray(bfovs, dtype=float).reshape(-1, 4)
    if bfovs.size == 0:
        return np.zeros((0,), dtype=float)
    _, _, lat_min, lat_max = _bfov_bounds(bfovs)
    width = np.clip(bfovs[:, 2], a_min=0.0, a_max=TAU)
    return width * np.clip(
        np.sin(lat_max) - np.sin(lat_min),
        a_min=0.0,
        a_max=None,
    )


def spherical_iou_bfov_matrix(bfovs1, bfovs2):
    bfovs1 = np.asarray(bfovs1, dtype=float).reshape(-1, 4)
    bfovs2 = np.asarray(bfovs2, dtype=float).reshape(-1, 4)
    if bfovs1.size == 0 or bfovs2.size == 0:
        return np.zeros((bfovs1.shape[0], bfovs2.shape[0]), dtype=float)

    lon1, half_w1, lat1_min, lat1_max = _bfov_bounds(bfovs1)
    lon2, half_w2, lat2_min, lat2_max = _bfov_bounds(bfovs2)
    left1 = lon1[:, None] - half_w1[:, None]
    right1 = lon1[:, None] + half_w1[:, None]
    left2 = lon2[None, :] - half_w2[None, :]
    right2 = lon2[None, :] + half_w2[None, :]

    lon_overlap = np.zeros((bfovs1.shape[0], bfovs2.shape[0]), dtype=float)
    for shift in (-TAU, 0.0, TAU):
        shifted_left2 = left2 + shift
        shifted_right2 = right2 + shift
        lon_overlap = np.maximum(
            lon_overlap,
            np.clip(
                np.minimum(right1, shifted_right2) - np.maximum(left1, shifted_left2),
                a_min=0.0,
                a_max=None,
            ),
        )

    lat_overlap_min = np.maximum(lat1_min[:, None], lat2_min[None, :])
    lat_overlap_max = np.minimum(lat1_max[:, None], lat2_max[None, :])
    lat_overlap = np.clip(
        np.sin(lat_overlap_max) - np.sin(lat_overlap_min),
        a_min=0.0,
        a_max=None,
    )
    intersection = lon_overlap * lat_overlap

    area1 = spherical_rectangle_area(bfovs1)[:, None]
    area2 = spherical_rectangle_area(bfovs2)[None, :]
    union = area1 + area2 - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def spherical_iou_xyxy_matrix(boxes1_xyxy, boxes2_xyxy, image_width, image_height):
    return spherical_iou_bfov_matrix(
        xyxy_to_bfov_lite(boxes1_xyxy, image_width=image_width, image_height=image_height),
        xyxy_to_bfov_lite(boxes2_xyxy, image_width=image_width, image_height=image_height),
    )


def great_circle_distance_bfov_matrix(bfovs1, bfovs2):
    bfovs1 = np.asarray(bfovs1, dtype=float).reshape(-1, 4)
    bfovs2 = np.asarray(bfovs2, dtype=float).reshape(-1, 4)
    if bfovs1.size == 0 or bfovs2.size == 0:
        return np.zeros((bfovs1.shape[0], bfovs2.shape[0]), dtype=float)

    lon1 = bfovs1[:, 0][:, None]
    lat1 = bfovs1[:, 1][:, None]
    lon2 = bfovs2[:, 0][None, :]
    lat2 = bfovs2[:, 1][None, :]

    cos_distance = (
        np.sin(lat1) * np.sin(lat2)
        + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    )
    cos_distance = np.clip(cos_distance, a_min=-1.0, a_max=1.0)
    return np.arccos(cos_distance)


def center_similarity_xyxy_matrix(boxes1_xyxy, boxes2_xyxy, image_width, image_height):
    distances = great_circle_distance_bfov_matrix(
        xyxy_to_bfov_lite(boxes1_xyxy, image_width=image_width, image_height=image_height),
        xyxy_to_bfov_lite(boxes2_xyxy, image_width=image_width, image_height=image_height),
    )
    return 1.0 - np.clip(distances / math.pi, a_min=0.0, a_max=1.0)


def seam_mask_xyxy(boxes_xyxy, image_width, seam_band_px):
    boxes_xyxy = _as_xyxy_array(boxes_xyxy)
    if boxes_xyxy.size == 0:
        return np.zeros((0,), dtype=bool)
    image_width = float(image_width)
    seam_band_px = float(seam_band_px)
    x1 = boxes_xyxy[:, 0]
    x2 = boxes_xyxy[:, 2]
    return (
        (x1 < seam_band_px)
        | (x2 > image_width - seam_band_px)
        | (x1 < 0.0)
        | (x2 > image_width)
    )


def seam_mask_xywh(boxes_xywh, image_width, seam_band_px):
    return seam_mask_xyxy(
        xywh_to_xyxy(boxes_xywh),
        image_width=image_width,
        seam_band_px=seam_band_px,
    )


def high_lat_mask_bfov(bfovs, high_lat_deg):
    bfovs = np.asarray(bfovs, dtype=float).reshape(-1, 4)
    if bfovs.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.abs(np.rad2deg(bfovs[:, 1])) >= float(high_lat_deg)


def high_lat_mask_xyxy(boxes_xyxy, image_width, image_height, high_lat_deg):
    return high_lat_mask_bfov(
        xyxy_to_bfov_lite(
            boxes_xyxy,
            image_width=image_width,
            image_height=image_height,
        ),
        high_lat_deg=high_lat_deg,
    )


def high_lat_mask_xywh(boxes_xywh, image_width, image_height, high_lat_deg):
    return high_lat_mask_xyxy(
        xywh_to_xyxy(boxes_xywh),
        image_width=image_width,
        image_height=image_height,
        high_lat_deg=high_lat_deg,
    )


def seam_or_high_lat_mask_xyxy(
    boxes_xyxy,
    image_width,
    image_height,
    seam_band_px,
    high_lat_deg,
):
    boxes_xyxy = _as_xyxy_array(boxes_xyxy)
    if boxes_xyxy.size == 0:
        return np.zeros((0,), dtype=bool)
    return seam_mask_xyxy(
        boxes_xyxy,
        image_width=image_width,
        seam_band_px=seam_band_px,
    ) | high_lat_mask_xyxy(
        boxes_xyxy,
        image_width=image_width,
        image_height=image_height,
        high_lat_deg=high_lat_deg,
    )


def seam_or_high_lat_mask_xywh(
    boxes_xywh,
    image_width,
    image_height,
    seam_band_px,
    high_lat_deg,
):
    return seam_or_high_lat_mask_xyxy(
        xywh_to_xyxy(boxes_xywh),
        image_width=image_width,
        image_height=image_height,
        seam_band_px=seam_band_px,
        high_lat_deg=high_lat_deg,
    )


def selective_spherical_pair_mask_xyxy(
    boxes1_xyxy,
    boxes2_xyxy,
    image_width,
    image_height,
    seam_band_px,
    high_lat_deg,
):
    boxes1_xyxy = _as_xyxy_array(boxes1_xyxy)
    boxes2_xyxy = _as_xyxy_array(boxes2_xyxy)
    if boxes1_xyxy.size == 0 or boxes2_xyxy.size == 0:
        return np.zeros((boxes1_xyxy.shape[0], boxes2_xyxy.shape[0]), dtype=bool)

    mask1 = seam_or_high_lat_mask_xyxy(
        boxes1_xyxy,
        image_width=image_width,
        image_height=image_height,
        seam_band_px=seam_band_px,
        high_lat_deg=high_lat_deg,
    )
    mask2 = seam_or_high_lat_mask_xyxy(
        boxes2_xyxy,
        image_width=image_width,
        image_height=image_height,
        seam_band_px=seam_band_px,
        high_lat_deg=high_lat_deg,
    )
    return mask1[:, None] | mask2[None, :]


__all__ = [
    "DEFAULT_ASSOCIATION_GEOMETRY_CFG",
    "center_similarity_xyxy_matrix",
    "great_circle_distance_bfov_matrix",
    "high_lat_mask_bfov",
    "high_lat_mask_xyxy",
    "high_lat_mask_xywh",
    "normalize_association_geometry_cfg",
    "normalize_tbd_tracker_cfg",
    "seam_mask_xywh",
    "seam_mask_xyxy",
    "seam_or_high_lat_mask_xywh",
    "seam_or_high_lat_mask_xyxy",
    "selective_spherical_pair_mask_xyxy",
    "spherical_iou_bfov_matrix",
    "spherical_iou_xyxy_matrix",
    "spherical_rectangle_area",
    "wrap_longitude_radians",
    "xywh_to_bfov_lite",
    "xywh_to_xyxy",
    "xyxy_to_bfov_lite",
]
