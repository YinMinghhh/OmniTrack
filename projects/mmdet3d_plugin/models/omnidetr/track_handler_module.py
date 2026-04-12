# query_handler_module.py
import torch
import pickle
from torch import nn
import numpy as np
from .ops import xywh2xyxy,xyxy2xywh
from .active_track_utils import collect_active_track_data
from .seam_duplicate_resolver import resolve_seam_duplicates_xyxy
from ..track.strack import STrack
from ..track.kalman_filter import KalmanFilter
from ..track.matching import iou_distance, iou_score
from ..trackers.ocsort_tracker.ocsort import OCSort
from ..trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort
from ..trackers.sort_tracker.sort import Sort
from ..trackers.byte_tracker.byte_tracker import BYTETracker
from ..trackers.args import make_parser
from torchvision.ops import nms



class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class TrackHandler:
    def __init__(self, instance_bank):
        self.instance_bank = instance_bank
        self.instance_bank.nms_thresh = 0.05
        self.instance_bank.track_thresh = 0.45
        self.instance_bank.det_thresh = 0.10
        self.instance_bank.init_thresh = 0.55

        self.pre_tracker_quantize_boxes = bool(
            self.tbd_handler_cfg.get("pre_tracker_quantize_boxes", False)
        )
        args = self._build_tracker_args()
        self.tracker = self._build_tracker(args)
        # args.track_thresh = 0.2
        # self.tracker = Sort(args, det_thresh=args.track_thresh)
        self.save_data = {}

    def __getattr__(self, name):
        # use instance_bank attr as normal
        if name in self.instance_bank.__dict__:
            return getattr(self.instance_bank, name)
        else:
            return getattr(self, name)
        
    def __setattr__(self, name, value):
        # 
        if name != 'instance_bank':
            setattr(self.instance_bank, name, value)
        else:
            super().__setattr__(name, value)

    def _build_tracker_args(self):
        args = make_parser().parse_args([])
        if not hasattr(args, "use_circular_track"):
            args.use_circular_track = False
        for key, value in (self.tbd_tracker_cfg or {}).items():
            setattr(args, key, value)
        return args

    def _build_tracker(self, args):
        return Hybrid_Sort(
            args,
            det_thresh=args.track_thresh,
            iou_threshold=args.iou_thresh,
            asso_func=args.asso,
            delta_t=args.deltat,
            inertia=args.inertia,
            use_byte=args.use_byte,
        )

    def _get_active_track_boxes(self, device, dtype):
        if not hasattr(self.tracker, "trackers"):
            return None, None

        seam_cfg = getattr(self.instance_bank, "seam_resolver_cfg", {})
        active_track_data = collect_active_track_data(
            self.tracker.trackers,
            max_time_since_update=seam_cfg.get(
                "active_track_max_time_since_update"
            ),
        )

        active_boxes = active_track_data["boxes"]
        active_track_boxes = None
        if active_boxes is not None:
            active_track_boxes = torch.as_tensor(
                active_boxes,
                device=device,
                dtype=dtype,
            )

        debug_payload = None
        if seam_cfg.get("debug_stats", False):
            debug_payload = {
                "active_track_max_time_since_update": seam_cfg.get(
                    "active_track_max_time_since_update"
                ),
                "retained_count": int(len(active_track_data["retained"])),
                "dropped_count": int(len(active_track_data["dropped"])),
                "retained": active_track_data["retained"],
                "dropped": active_track_data["dropped"],
            }

        return active_track_boxes, debug_payload

    def query_handler(self, bbox, score, meta, qt):
        self.img_wh = meta['image_wh'][0][0].cpu().numpy()
        self.ori_shape = np.array([meta['ori_shape'][1][0].cpu().numpy(), meta['ori_shape'][0][0].cpu().numpy()])
        scale_w = self.ori_shape[0] / self.img_wh[0]
        mask = score > self.det_thresh
        bbox = bbox[mask.squeeze(-1)]
        qt = qt[mask.squeeze(-1)] if qt is not None else None
        bbox = STrack.cxcywh_to_tlbr_to_tensor(bbox)
        bbox[:,[0,2]] *= self.img_wh[0]
        bbox[:,[1,3]] *= self.img_wh[1]
        score = score[mask]
        if self.pre_tracker_quantize_boxes:
            bbox = bbox.to(dtype=torch.int, device=bbox.device)
            bbox = bbox.to(dtype=torch.float32, device=bbox.device)
        else:
            bbox = bbox.to(dtype=torch.float32, device=bbox.device)
        if meta['image_wh'][0][0][0]-meta['ori_shape'][1] > 0 :
            seam_cfg = getattr(self.instance_bank, "seam_resolver_cfg", {})
            if seam_cfg.get("enabled", False):
                labels = torch.zeros(
                    bbox.shape[0], device=bbox.device, dtype=torch.long
                )
                active_tracks, active_track_debug = self._get_active_track_boxes(
                    device=bbox.device,
                    dtype=bbox.dtype,
                )
                bbox, score, _, qt, self.seam_resolver_last_stats = resolve_seam_duplicates_xyxy(
                    bbox,
                    score,
                    labels,
                    image_width=float(meta['ori_shape'][1]),
                    seam_resolver_cfg=seam_cfg,
                    qualities=qt,
                    active_tracks=active_tracks,
                )
                if active_track_debug is not None:
                    self.seam_resolver_last_stats = dict(
                        self.seam_resolver_last_stats
                    )
                    self.seam_resolver_last_stats[
                        "active_track_debug"
                    ] = active_track_debug
            else:
                cx = (bbox[:,0] + bbox[:,2])/2
                mask = cx < meta['ori_shape'][1]
                bbox = bbox[mask]
                score = score[mask]
                qt = qt[mask] if qt is not None else None
                self.seam_resolver_last_stats = {
                    "enabled": False,
                    "legacy_hard_crop": True,
                    "input_count": int(mask.shape[0]),
                    "output_count": int(mask.sum().item()),
                }

        keep_indices = nms(bbox, score, iou_threshold=0.35)
        bbox = bbox[keep_indices]
        score = score[keep_indices]

        dets = torch.cat([bbox, score.unsqueeze(1)], dim=1)

        count = 0
        if self.timestamp is None or abs(self.timestamp - meta['timestamp']) >100:
            args = self._build_tracker_args()
            self.tracker = self._build_tracker(args)
            # args.track_thresh = 0.2
            # self.tracker = Sort(args, det_thresh=args.track_thresh)
            self.frame_id = 0
            # with open("/root/autodl-tmp/sparse4D_track/huang-2-2019-01-25_0.pkl", 'wb') as f:
            #     pickle.dump(self.save_data, f)
            # print("save!!!")
        frame_context = {
            "image_width": float(self.ori_shape[0]),
        }
        results, tracklets = self.tracker.update(
            dets.cpu().numpy(),
            frame_context=frame_context,
        )

        # self.save_data.update(
        #     {
        #         f'{self.frame_id}_input': dets,
        #         f'{self.frame_id}_otput': results,
        #     }
        # )

        self.timestamp = meta['timestamp']
        self.frame_id += 1
        self.starcks = tracklets

        return results, []
        
