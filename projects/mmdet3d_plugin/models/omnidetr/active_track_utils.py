import numpy as np


def _box_from_array(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape[0] < 4:
        return None
    return array[:4].copy()


def _is_valid_xyxy(box):
    return (
        box is not None
        and np.isfinite(box).all()
        and box[2] > box[0]
        and box[3] > box[1]
    )


def _track_identifier(track):
    for attr in ("id", "track_id"):
        if hasattr(track, attr):
            value = getattr(track, attr)
            if value is not None:
                return int(value)
    return None


def collect_active_track_data(trackers, *, max_time_since_update=None):
    retained_boxes = []
    retained_meta = []
    dropped_meta = []

    for track_index, track in enumerate(trackers or []):
        last_observation = _box_from_array(getattr(track, "last_observation", None))
        state_box = _box_from_array(track.get_state())
        time_since_update = getattr(track, "time_since_update", None)

        meta = {
            "track_index": int(track_index),
            "track_id": _track_identifier(track),
            "time_since_update": None
            if time_since_update is None
            else int(time_since_update),
            "last_observation": None
            if last_observation is None
            else last_observation.tolist(),
            "state_box": None if state_box is None else state_box.tolist(),
        }

        if max_time_since_update is not None:
            if time_since_update is None or time_since_update > max_time_since_update:
                dropped_meta.append(
                    {
                        **meta,
                        "selected_box": None,
                        "selected_box_source": None,
                        "drop_reason": "stale_time_since_update",
                    }
                )
                continue

        selected_box = None
        selected_source = None
        if _is_valid_xyxy(last_observation):
            selected_box = last_observation
            selected_source = "last_observation"
        elif _is_valid_xyxy(state_box):
            selected_box = state_box
            selected_source = "state_box"

        if selected_box is None:
            dropped_meta.append(
                {
                    **meta,
                    "selected_box": None,
                    "selected_box_source": None,
                    "drop_reason": "invalid_box_geometry",
                }
            )
            continue

        retained_boxes.append(selected_box)
        retained_meta.append(
            {
                **meta,
                "selected_box": selected_box.tolist(),
                "selected_box_source": selected_source,
            }
        )

    boxes = None
    if retained_boxes:
        boxes = np.asarray(retained_boxes, dtype=np.float32)

    return {
        "boxes": boxes,
        "retained": retained_meta,
        "dropped": dropped_meta,
    }


__all__ = ["collect_active_track_data"]
