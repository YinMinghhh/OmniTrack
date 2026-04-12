import numpy as np


def circ_dx(x_a, x_b, image_width):
    if image_width is None or image_width <= 0:
        return x_a - x_b
    half = image_width / 2.0
    return ((x_a - x_b + half) % image_width) - half


def signed_circular_delta(delta, image_width):
    if image_width is None or image_width <= 0:
        return delta
    half = image_width / 2.0
    return ((delta + half) % image_width) - half


def _clone_box(box):
    if hasattr(box, "clone"):
        return box.clone()
    return np.asarray(box, dtype=np.float32).copy()


def _maximum(lhs, rhs):
    if hasattr(lhs, "detach") or hasattr(rhs, "detach"):
        import torch

        return torch.maximum(lhs, rhs)
    return np.maximum(lhs, rhs)


def shift_bbox_x(bbox, shift):
    shifted = _clone_box(bbox)
    shifted[..., 0] += shift
    shifted[..., 2] += shift
    return shifted


def bbox_center_x(bbox):
    return (bbox[..., 0] + bbox[..., 2]) / 2.0


def pairwise_iou_matrix(bboxes1, bboxes2):
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)

    bboxes1 = np.expand_dims(np.asarray(bboxes1, dtype=np.float32), 1)
    bboxes2 = np.expand_dims(np.asarray(bboxes2, dtype=np.float32), 0)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1]
    )
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1]
    )
    union = area1 + area2 - inter
    return inter / np.maximum(union, 1e-6)


def wrap_metric_batch(metric_fn, bboxes1, bboxes2, image_width):
    if image_width is None or image_width <= 0:
        return metric_fn(bboxes1, bboxes2)
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return metric_fn(bboxes1, bboxes2)

    best = None
    for shift in (-image_width, 0.0, image_width):
        shifted = shift_bbox_x(bboxes2, shift)
        current = metric_fn(bboxes1, shifted)
        best = current if best is None else _maximum(best, current)
    return best


def wrap_iou_batch(bboxes1, bboxes2, image_width):
    return wrap_metric_batch(pairwise_iou_matrix, bboxes1, bboxes2, image_width)


def wrap_iou_pair(bbox1, bbox2, image_width):
    return float(wrap_iou_batch([bbox1], [bbox2], image_width)[0, 0])


def unwrap_to_track(bbox_xyxy, ref_box_xyxy, image_width):
    if image_width is None or image_width <= 0:
        return _clone_box(bbox_xyxy)

    bbox = _clone_box(bbox_xyxy)
    reference_center_x = bbox_center_x(ref_box_xyxy)
    center_x = bbox_center_x(bbox)
    delta_x = circ_dx(center_x, reference_center_x, image_width)
    target_center_x = reference_center_x + delta_x
    shift_x = target_center_x - center_x
    return shift_bbox_x(bbox, shift_x)


__all__ = [
    "bbox_center_x",
    "circ_dx",
    "pairwise_iou_matrix",
    "shift_bbox_x",
    "signed_circular_delta",
    "unwrap_to_track",
    "wrap_iou_batch",
    "wrap_iou_pair",
    "wrap_metric_batch",
]
