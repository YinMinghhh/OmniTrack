import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious

def _as_tlbr_array(tlbrs):
    return np.ascontiguousarray(np.asarray(tlbrs, dtype=np.float))


def _shift_x(tlbrs, dx):
    shifted = tlbrs.copy()
    shifted[:, [0, 2]] += dx
    return shifted


def ious(atlbrs, btlbrs, wrap_w=None):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    atlbrs = _as_tlbr_array(atlbrs)
    btlbrs = _as_tlbr_array(btlbrs)
    ious = bbox_ious(atlbrs, btlbrs)
    if wrap_w is None:
        return ious

    wrap_w = float(wrap_w)
    if wrap_w <= 0:
        return ious

    ious_pos = bbox_ious(atlbrs, _shift_x(btlbrs, wrap_w))
    ious_neg = bbox_ious(atlbrs, _shift_x(btlbrs, -wrap_w))
    ious = np.maximum(ious, np.maximum(ious_pos, ious_neg))

    return ious


def iou_distance(atracks, btracks, wrap_w=None):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs, wrap_w=wrap_w)
    cost_matrix = 1 - _ious

    return cost_matrix

def iou_score(atracks, btracks, wrap_w=None):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    ret = []
    _ious = ious(atracks, btracks, wrap_w=wrap_w)


    return _ious.diagonal()
