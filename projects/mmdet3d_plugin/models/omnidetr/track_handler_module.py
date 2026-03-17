# track_handler_module.py
import copy

import numpy as np
import torch
from torchvision.ops import nms

from ..track.strack import STrack
from ..trackers.args import make_parser
from ..trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort



class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


def _timestamp_to_scalar(ts):
    if ts is None:
        return None
    if hasattr(ts, "numel"):
        if ts.numel() > 1:
            ts = ts[0]
        if hasattr(ts, "item"):
            ts = ts.item()
        return ts
    if isinstance(ts, (list, tuple)):
        if len(ts) == 0:
            return None
        return _timestamp_to_scalar(ts[0])
    return ts


class TrackHandler:
    def __init__(self, instance_bank):
        self.instance_bank = instance_bank
        self.tbd_backend = getattr(instance_bank, "tbd_backend", "hybridsort")
        if self.tbd_backend != "hybridsort":
            raise NotImplementedError(
                f"Unsupported TBD backend {self.tbd_backend!r}. "
                "Only 'hybridsort' is supported."
            )

        cfg = dict(getattr(instance_bank, "tbd_handler_cfg", {}))
        self.det_thresh = cfg.get("det_thresh", 0.10)
        self.nms_iou = cfg.get("nms_iou", 0.35)
        self.reset_time_gap = cfg.get("reset_time_gap", 100)

        self.tracker_args = self._build_tracker_args(
            getattr(instance_bank, "tbd_tracker_cfg", {})
        )
        self.tracker = self._build_tracker()
        self.save_data = {}

    def __getattr__(self, name):
        return getattr(self.instance_bank, name)

    def _build_tracker_args(self, override_cfg):
        args = make_parser().parse_args([])
        override_cfg = dict(override_cfg or {})
        unknown_keys = sorted(set(override_cfg) - set(vars(args)))
        if unknown_keys:
            raise KeyError(
                "Unknown HybridSORT tracker config keys: %s"
                % ", ".join(unknown_keys)
            )
        for key, value in override_cfg.items():
            setattr(args, key, value)
        return args

    def _build_tracker(self):
        args = copy.deepcopy(self.tracker_args)
        return Hybrid_Sort(
            args,
            det_thresh=args.track_thresh,
            iou_threshold=args.iou_thresh,
            asso_func=args.asso,
            delta_t=args.deltat,
            inertia=args.inertia,
            use_byte=args.use_byte,
        )

    def query_handler(self, bbox, score, meta, qt):
        self.img_wh = meta['image_wh'][0][0].cpu().numpy()
        self.ori_shape = np.array([meta['ori_shape'][1][0].cpu().numpy(), meta['ori_shape'][0][0].cpu().numpy()])
        curr_ts = _timestamp_to_scalar(meta['timestamp'])
        prev_ts = _timestamp_to_scalar(self.instance_bank.timestamp)
        if prev_ts is None or abs(prev_ts - curr_ts) > self.reset_time_gap:
            self.tracker = self._build_tracker()
            self.instance_bank.frame_id = 0

        mask = score > self.det_thresh
        bbox = bbox[mask.squeeze(-1)]
        score = score[mask]
        if bbox.numel() > 0:
            bbox = STrack.cxcywh_to_tlbr_to_tensor(bbox)
            bbox[:, [0, 2]] *= self.img_wh[0]
            bbox[:, [1, 3]] *= self.img_wh[1]
            bbox = bbox.to(dtype=torch.int, device=bbox.device)
            bbox = bbox.to(dtype=torch.float32, device=bbox.device)
            if meta['image_wh'][0][0][0] - meta['ori_shape'][1] > 0:
                cx = (bbox[:, 0] + bbox[:, 2]) / 2
                mask = cx < meta['ori_shape'][1]
                bbox = bbox[mask]
                score = score[mask]

            if bbox.shape[0] > 0:
                keep_indices = nms(bbox, score, iou_threshold=self.nms_iou)
                bbox = bbox[keep_indices]
                score = score[keep_indices]
            dets = torch.cat([bbox, score.unsqueeze(1)], dim=1)
        else:
            dets = bbox.new_zeros((0, 5))

        results, tracklets = self.tracker.update(dets.cpu().numpy())

        # self.save_data.update(
        #     {
        #         f'{self.frame_id}_input': dets,
        #         f'{self.frame_id}_otput': results,
        #     }
        # )

        self.instance_bank.timestamp = curr_ts
        self.instance_bank.frame_id += 1
        self.instance_bank.starcks = tracklets

        return results, []
        
