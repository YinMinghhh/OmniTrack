"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np

from .association import *
from .circular_utils import bbox_center_x, shift_bbox_x, unwrap_to_track, wrap_metric_batch


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    score = bbox[4]
    if score:
        return np.array([x, y, s, score, r]).reshape((5, 1))
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[4])
    h = x[2] / w
    score = x[3]
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_lt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_rt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[3]
    cx2, cy2 = bbox2[0], bbox2[3]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_lb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[1]
    cx2, cy2 = bbox2[2], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_rb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[3]
    cx2, cy2 = bbox2[2], bbox2[3]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def wrap_center(center_x, image_width):
    if image_width is None or image_width <= 0:
        return center_x
    return center_x % image_width


def principalize_bbox_to_interval(bbox, image_width):
    bbox = np.asarray(bbox, dtype=np.float32).copy()
    if image_width is None or image_width <= 0:
        return bbox
    center_x = float(bbox_center_x(bbox))
    wrapped_center_x = float(wrap_center(center_x, image_width))
    return shift_bbox_x(bbox, wrapped_center_x - center_x)


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, delta_t=3, orig=False, args=None, image_width=None):
        if not orig:
            from .kalmanfilter_score_new import KalmanFilterNew_score_new as KalmanFilter_score_new

            self.kf = KalmanFilter_score_new(dim_x=9, dim_z=5)
        else:
            from filterpy.kalman import KalmanFilter

            self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[5:, 5:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[-2, -2] *= 0.01
        self.kf.Q[5:, 5:] *= 0.01

        self.image_width = None
        self.visible_reference_box = np.asarray(bbox, dtype=np.float32).copy()
        self.set_image_width(image_width)
        self.kf.x[:5] = convert_bbox_to_z(self._principalize_visible_bbox(bbox))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.age_recover_for_cbiou = 0
        self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=np.float32)
        self.last_observation_save = np.array([-1, -1, -1, -1, -1], dtype=np.float32)
        self.observations = dict()
        self.history_observations = []
        self.velocity_lt = None
        self.velocity_rt = None
        self.velocity_lb = None
        self.velocity_rb = None
        self.delta_t = delta_t
        self.confidence_pre = None
        self.confidence = bbox[-1]
        self.args = args
        self.kf.args = args

    def set_image_width(self, image_width):
        if image_width is None:
            return
        image_width = float(image_width)
        if image_width > 0:
            self.image_width = image_width
            self._normalize_internal_state()

    def _principalize_visible_bbox(self, bbox):
        return principalize_bbox_to_interval(bbox, self.image_width)

    def _normalize_internal_state(self):
        if self.image_width is None or self.image_width <= 0:
            return
        self.kf.x[0, 0] = wrap_center(float(self.kf.x[0, 0]), self.image_width)

    def get_internal_state(self):
        return convert_x_to_bbox(self.kf.x)

    def get_visible_state(self):
        internal_box = self.get_internal_state()[0]
        if self.image_width is None or self.image_width <= 0:
            return internal_box.reshape(1, -1)
        visible_reference = self.visible_reference_box
        visible_box = unwrap_to_track(internal_box, visible_reference, self.image_width)
        return np.asarray(visible_box, dtype=np.float32).reshape(1, -1)

    def get_state(self):
        return self.get_visible_state()

    def update(self, bbox, visible_bbox=None):
        velocity_lt = None
        velocity_rt = None
        velocity_lb = None
        velocity_rb = None

        if bbox is not None:
            visible_bbox = np.asarray(
                visible_bbox if visible_bbox is not None else bbox,
                dtype=np.float32,
            ).copy()
            internal_bbox = np.asarray(bbox, dtype=np.float32).copy()

            if self.last_observation.sum() >= 0:
                previous_box = None
                for i in range(self.delta_t):
                    if self.age - i - 1 in self.observations:
                        previous_box = self.observations[self.age - i - 1]
                        if velocity_lt is not None:
                            velocity_lt += speed_direction_lt(previous_box, visible_bbox)
                            velocity_rt += speed_direction_rt(previous_box, visible_bbox)
                            velocity_lb += speed_direction_lb(previous_box, visible_bbox)
                            velocity_rb += speed_direction_rb(previous_box, visible_bbox)
                        else:
                            velocity_lt = speed_direction_lt(previous_box, visible_bbox)
                            velocity_rt = speed_direction_rt(previous_box, visible_bbox)
                            velocity_lb = speed_direction_lb(previous_box, visible_bbox)
                            velocity_rb = speed_direction_rb(previous_box, visible_bbox)
                if previous_box is None:
                    previous_box = self.last_observation
                    self.velocity_lt = speed_direction_lt(previous_box, visible_bbox)
                    self.velocity_rt = speed_direction_rt(previous_box, visible_bbox)
                    self.velocity_lb = speed_direction_lb(previous_box, visible_bbox)
                    self.velocity_rb = speed_direction_rb(previous_box, visible_bbox)
                else:
                    self.velocity_lt = velocity_lt
                    self.velocity_rt = velocity_rt
                    self.velocity_lb = velocity_lb
                    self.velocity_rb = velocity_rb

            self.last_observation = visible_bbox
            self.last_observation_save = visible_bbox.copy()
            self.observations[self.age] = visible_bbox.copy()
            self.history_observations.append(visible_bbox.copy())
            self.visible_reference_box = visible_bbox.copy()

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(internal_bbox))
            self._normalize_internal_state()
            self.confidence_pre = self.confidence
            self.confidence = visible_bbox[-1]
            self.age_recover_for_cbiou = self.age
        else:
            self.kf.update(bbox)
            self._normalize_internal_state()
            self.confidence_pre = None

    def predict(self):
        if (self.kf.x[7] + self.kf.x[2]) <= 0:
            self.kf.x[7] *= 0.0

        self.kf.predict()
        self._normalize_internal_state()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.get_internal_state())
        if not self.confidence_pre:
            return (
                self.history[-1],
                np.clip(self.kf.x[3], self.args.track_thresh, 1.0),
                np.clip(self.confidence, 0.1, self.args.track_thresh),
            )
        return (
            self.history[-1],
            np.clip(self.kf.x[3], self.args.track_thresh, 1.0),
            np.clip(self.confidence - (self.confidence_pre - self.confidence), 0.1, self.args.track_thresh),
        )


ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
    "Height_Modulated_IoU": hmiou,
}


class Hybrid_Sort(object):
    def __init__(
        self,
        args,
        det_thresh,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_byte=False,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        self.args = args
        self.use_circular_track = bool(getattr(args, "use_circular_track", False))
        self.latest_circular_stats = {}
        KalmanBoxTracker.count = 0

    def _normalize_frame_context(self, frame_context):
        frame_context = dict(frame_context or {})
        frame_context["image_width"] = float(frame_context.get("image_width", 0.0) or 0.0)
        return frame_context

    def _metric_matrix(self, detections, trackers, frame_context):
        if self.use_circular_track and frame_context["image_width"] > 0:
            return wrap_metric_batch(
                self.asso_func,
                detections,
                trackers,
                frame_context["image_width"],
            )
        return np.array(self.asso_func(detections, trackers))

    def _set_tracker_image_width(self, image_width):
        if not self.use_circular_track or image_width <= 0:
            return
        for tracker in self.trackers:
            tracker.set_image_width(image_width)

    def _align_detection_for_track(self, detection, track, frame_context):
        detection = np.asarray(detection, dtype=np.float32).copy()
        if not self.use_circular_track or frame_context["image_width"] <= 0:
            return detection, detection.copy()

        reference_bbox = track.get_visible_state()[0]
        visible_bbox = np.asarray(
            unwrap_to_track(detection, reference_bbox, frame_context["image_width"]),
            dtype=np.float32,
        )
        internal_bbox = principalize_bbox_to_interval(
            visible_bbox,
            frame_context["image_width"],
        )
        return visible_bbox, internal_bbox

    def update(self, output_results, frame_context=None):
        if output_results is None:
            return np.empty((0, 5)), self.trackers

        frame_context = self._normalize_frame_context(frame_context)
        self.latest_circular_stats = {
            "use_circular_track": self.use_circular_track,
            "image_width": frame_context["image_width"],
        }
        self._set_tracker_image_width(frame_context["image_width"])

        self.frame_count += 1
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos, kalman_score, simple_score = self.trackers[t].predict()
            try:
                trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score[0], simple_score[0]]
            except Exception:
                trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score[0], simple_score]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities_lt = np.array(
            [trk.velocity_lt if trk.velocity_lt is not None else np.array((0, 0)) for trk in self.trackers]
        )
        velocities_rt = np.array(
            [trk.velocity_rt if trk.velocity_rt is not None else np.array((0, 0)) for trk in self.trackers]
        )
        velocities_lb = np.array(
            [trk.velocity_lb if trk.velocity_lb is not None else np.array((0, 0)) for trk in self.trackers]
        )
        velocities_rb = np.array(
            [trk.velocity_rb if trk.velocity_rb is not None else np.array((0, 0)) for trk in self.trackers]
        )
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers]
        )

        if self.args.TCM_first_step:
            if self.use_circular_track and frame_context["image_width"] > 0:
                matched, unmatched_dets, unmatched_trks = associate_4_points_with_score_circular(
                    dets,
                    trks,
                    self.iou_threshold,
                    velocities_lt,
                    velocities_rt,
                    velocities_lb,
                    velocities_rb,
                    k_observations,
                    self.inertia,
                    frame_context["image_width"],
                    lambda det_arr, trk_arr: self._metric_matrix(det_arr, trk_arr, frame_context),
                    self.args,
                )
            else:
                matched, unmatched_dets, unmatched_trks = associate_4_points_with_score(
                    dets,
                    trks,
                    self.iou_threshold,
                    velocities_lt,
                    velocities_rt,
                    velocities_lb,
                    velocities_rb,
                    k_observations,
                    self.inertia,
                    self.asso_func,
                    self.args,
                )
        else:
            if self.use_circular_track and frame_context["image_width"] > 0:
                matched, unmatched_dets, unmatched_trks = associate_4_points_circular(
                    dets,
                    trks,
                    self.iou_threshold,
                    velocities_lt,
                    velocities_rt,
                    velocities_lb,
                    velocities_rb,
                    k_observations,
                    self.inertia,
                    frame_context["image_width"],
                    lambda det_arr, trk_arr: self._metric_matrix(det_arr, trk_arr, frame_context),
                    self.args,
                )
            else:
                matched, unmatched_dets, unmatched_trks = associate_4_points(
                    dets,
                    trks,
                    self.iou_threshold,
                    velocities_lt,
                    velocities_rt,
                    velocities_lb,
                    velocities_rb,
                    k_observations,
                    self.inertia,
                    self.asso_func,
                    self.args,
                )

        for m in matched:
            visible_bbox, internal_bbox = self._align_detection_for_track(
                dets[m[0], :],
                self.trackers[m[1]],
                frame_context,
            )
            self.trackers[m[1]].update(internal_bbox, visible_bbox=visible_bbox)

        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self._metric_matrix(dets_second, u_trks, frame_context)
            if iou_left.max() > self.iou_threshold:
                if self.args.TCM_byte_step:
                    iou_left -= np.array(cal_score_dif_batch_two_score(dets_second, u_trks) * self.args.TCM_byte_step_weight)
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    visible_bbox, internal_bbox = self._align_detection_for_track(
                        dets_second[det_ind, :],
                        self.trackers[trk_ind],
                        frame_context,
                    )
                    self.trackers[trk_ind].update(internal_bbox, visible_bbox=visible_bbox)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self._metric_matrix(left_dets, left_trks, frame_context)

            if iou_left.max() > self.iou_threshold:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    visible_bbox, internal_bbox = self._align_detection_for_track(
                        dets[det_ind, :],
                        self.trackers[trk_ind],
                        frame_context,
                    )
                    self.trackers[trk_ind].update(internal_bbox, visible_bbox=visible_bbox)
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :],
                delta_t=self.delta_t,
                args=self.args,
                image_width=frame_context["image_width"] if self.use_circular_track else None,
            )
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_visible_state()[0][:4]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret), self.trackers
        return np.empty((0, 5)), self.trackers

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh

        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti(
            dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia
        )

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:, 4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                        cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if trk.time_since_update < 1:
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    ret.append(np.concatenate((d, [trk.id + 1], [trk.cate], [0])).reshape(1, -1))
                if trk.hit_streak == self.min_hits:
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i + 2)]
                        ret.append(
                            (
                                np.concatenate(
                                    (
                                        prev_observation[:4],
                                        [trk.id + 1],
                                        [trk.cate],
                                        [-(prev_i + 1)],
                                    )
                                )
                            ).reshape(1, -1)
                        )
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))
