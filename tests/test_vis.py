import importlib.util
import pathlib

import numpy as np


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools/vis.py"
SPEC = importlib.util.spec_from_file_location("repo_vis", MODULE_PATH)
VIS_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIS_MODULE)
Visualizer = VIS_MODULE.Visualizer


def _capture_rectangle_calls(monkeypatch):
    calls = []

    def fake_rectangle(img, pt1, pt2, color, thickness):
        calls.append((pt1, pt2, color, thickness))
        return img

    monkeypatch.setattr(VIS_MODULE.cv2, "rectangle", fake_rectangle)
    monkeypatch.setattr(VIS_MODULE.cv2, "putText", lambda img, *args, **kwargs: img)
    return calls


def _draw_and_get_call(monkeypatch, bboxes, *, ltwh):
    calls = _capture_rectangle_calls(monkeypatch)
    vis = Visualizer.__new__(Visualizer)
    image = np.zeros((220, 3900, 3), dtype=np.uint8)
    vis.draw_bbox(image, bboxes, ltwh=ltwh, track_id=[1], score=[0.9])
    assert len(calls) == 1
    return calls[0]


def test_ltwh_matches_xyxy_for_regular_box(monkeypatch):
    xyxy_call = _draw_and_get_call(monkeypatch, [[100, 20, 130, 60]], ltwh=False)
    ltwh_call = _draw_and_get_call(monkeypatch, [[100, 20, 30, 40]], ltwh=True)

    assert xyxy_call[:2] == ((100, 20), (130, 60))
    assert ltwh_call[:2] == xyxy_call[:2]


def test_ltwh_keeps_seam_crossing_geometry(monkeypatch):
    seam_call = _draw_and_get_call(monkeypatch, [[3730, 10, 100, 100]], ltwh=True)

    assert seam_call[:2] == ((3730, 10), (3830, 110))


def test_ltwh_control_case_geometry(monkeypatch):
    control_call = _draw_and_get_call(monkeypatch, [[100, 20, 30, 40]], ltwh=True)

    assert control_call[:2] == ((100, 20), (130, 60))
