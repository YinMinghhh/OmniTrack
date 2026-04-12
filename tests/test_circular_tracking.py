import pathlib
import sys
from types import SimpleNamespace

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACKER_ROOT = ROOT / "projects" / "mmdet3d_plugin" / "models" / "trackers"
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from hybrid_sort_tracker.circular_utils import circ_dx, unwrap_to_track, wrap_iou_pair, wrap_metric_batch
from hybrid_sort_tracker import hybrid_sort as hybrid_sort_module
from hybrid_sort_tracker.hybrid_sort import Hybrid_Sort


class StubKalmanBoxTracker:
    count = 0

    def __init__(self, bbox, delta_t=3, args=None, image_width=None, orig=False):
        self.bbox = np.asarray(bbox, dtype=np.float32).copy()
        self.id = StubKalmanBoxTracker.count
        StubKalmanBoxTracker.count += 1
        self.delta_t = delta_t
        self.args = args
        self.time_since_update = 0
        self.hit_streak = 1
        self.age = 0
        self.age_recover_for_cbiou = 0
        self.history = []
        self.observations = {}
        self.history_observations = []
        self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=np.float32)
        self.last_observation_save = np.array([-1, -1, -1, -1, -1], dtype=np.float32)
        self.velocity_lt = np.zeros((2,), dtype=np.float32)
        self.velocity_rt = np.zeros((2,), dtype=np.float32)
        self.velocity_lb = np.zeros((2,), dtype=np.float32)
        self.velocity_rb = np.zeros((2,), dtype=np.float32)
        self.confidence = float(self.bbox[-1])
        self.confidence_pre = None
        self.image_width = float(image_width) if image_width is not None else None
        self.visible_reference_box = self.bbox.copy()
        self._internal_box = hybrid_sort_module.principalize_bbox_to_interval(
            self.bbox,
            self.image_width,
        )

    def set_image_width(self, image_width):
        self.image_width = float(image_width)

    def get_internal_state(self):
        return self._internal_box.reshape(1, -1)

    def get_visible_state(self):
        return np.asarray(
            unwrap_to_track(self._internal_box, self.visible_reference_box, self.image_width),
            dtype=np.float32,
        ).reshape(1, -1)

    def get_state(self):
        return self.get_visible_state()

    def update(self, bbox, visible_bbox=None):
        self.age += 1
        if bbox is None:
            self.time_since_update += 1
            self.hit_streak = 0
            return

        visible_bbox = np.asarray(
            visible_bbox if visible_bbox is not None else bbox,
            dtype=np.float32,
        ).copy()
        self._internal_box = np.asarray(bbox, dtype=np.float32).copy()
        self.visible_reference_box = visible_bbox.copy()
        self.last_observation = visible_bbox.copy()
        self.last_observation_save = visible_bbox.copy()
        self.observations[self.age] = visible_bbox.copy()
        self.history_observations.append(visible_bbox.copy())
        self.time_since_update = 0
        self.hit_streak += 1
        self.confidence_pre = self.confidence
        self.confidence = float(visible_bbox[-1])

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        self.history.append(self.get_internal_state())
        score = np.array([self._internal_box[-1]], dtype=np.float32)
        return self.get_internal_state()[:, :4], score, score[0]


hybrid_sort_module.KalmanBoxTracker = StubKalmanBoxTracker


def make_args(**overrides):
    args = SimpleNamespace(
        track_thresh=0.6,
        iou_thresh=0.15,
        min_hits=1,
        inertia=0.05,
        deltat=3,
        track_buffer=30,
        match_thresh=0.9,
        asso="iou",
        use_byte=False,
        TCM_first_step=False,
        TCM_byte_step=False,
        TCM_first_step_weight=1.0,
        TCM_byte_step_weight=1.0,
        use_circular_track=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_tracker(**overrides):
    args = make_args(**overrides)
    return Hybrid_Sort(
        args,
        det_thresh=args.track_thresh,
        iou_threshold=args.iou_thresh,
        delta_t=args.deltat,
        inertia=args.inertia,
        use_byte=args.use_byte,
        asso_func="iou",
    )


def test_circ_dx_prefers_shortest_wrap():
    image_width = 3760.0
    assert np.isclose(circ_dx(20.0, 3780.0, image_width), 0.0)
    assert np.isclose(circ_dx(50.0, 3780.0, image_width), 30.0)
    assert np.isclose(circ_dx(3780.0, 50.0, image_width), -30.0)


def test_unwrap_to_track_preserves_reference_branch():
    image_width = 3760.0
    reference_bbox = np.array([3730.0, 10.0, 3830.0, 110.0], dtype=np.float32)
    seam_bbox = np.array([0.0, 10.0, 100.0, 110.0, 0.9], dtype=np.float32)

    aligned = unwrap_to_track(seam_bbox, reference_bbox, image_width)

    assert aligned[0] >= image_width
    assert aligned[2] > image_width


def test_wrap_metric_batch_prefers_shifted_overlap():
    image_width = 3760.0
    left = np.array([[0.0, 10.0, 60.0, 110.0]], dtype=np.float32)
    right = np.array([[3760.0, 10.0, 3820.0, 110.0]], dtype=np.float32)

    wrapped_iou = wrap_metric_batch(
        lambda a, b: np.array([[wrap_iou_pair(a[0], b[0], image_width)]], dtype=np.float32),
        left,
        right,
        image_width,
    )

    assert np.isclose(float(wrapped_iou[0, 0]), 1.0)


def test_hybrid_sort_exposes_visible_state_but_keeps_internal_principal_state():
    tracker = make_tracker()
    frame_context = {"image_width": 3760.0}
    results, tracklets = tracker.update(
        np.array([[3730.0, 10.0, 3830.0, 110.0, 0.95]], dtype=np.float32),
        frame_context=frame_context,
    )

    assert results.shape[0] == 1
    track = tracklets[0]
    visible_state = track.get_visible_state()[0]
    internal_state = track.get_internal_state()[0]

    assert visible_state[2] > frame_context["image_width"]
    assert internal_state[0] < 0.0
    assert internal_state[2] < frame_context["image_width"]
    assert np.allclose(track.get_state()[0], visible_state)


def test_hybrid_sort_keeps_same_id_across_seam_and_visible_branch():
    tracker = make_tracker()
    frame_context = {"image_width": 3760.0}

    results1, _ = tracker.update(
        np.array([[3730.0, 10.0, 3830.0, 110.0, 0.95]], dtype=np.float32),
        frame_context=frame_context,
    )
    results2, tracklets = tracker.update(
        np.array([[0.0, 10.0, 100.0, 110.0, 0.95]], dtype=np.float32),
        frame_context=frame_context,
    )

    assert results1.shape[0] == 1
    assert results2.shape[0] == 1
    assert int(results1[0, 4]) == int(results2[0, 4])
    assert tracklets[0].last_observation[2] > frame_context["image_width"]


def test_non_circular_tracker_does_not_adopt_frame_width():
    tracker = make_tracker(use_circular_track=False)
    frame_context = {"image_width": 3760.0}

    results, tracklets = tracker.update(
        np.array([[3730.0, 10.0, 3830.0, 110.0, 0.95]], dtype=np.float32),
        frame_context=frame_context,
    )

    assert results.shape[0] == 1
    track = tracklets[0]
    assert track.image_width is None
    assert np.allclose(track.get_state()[0][:4], np.array([3730.0, 10.0, 3830.0, 110.0], dtype=np.float32))
