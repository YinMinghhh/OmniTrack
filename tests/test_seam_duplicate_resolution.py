import importlib.util
import pathlib
import unittest

import torch


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "projects/mmdet3d_plugin/models/omnidetr/seam_duplicate_resolver.py"
)
SPEC = importlib.util.spec_from_file_location("seam_duplicate_resolver", MODULE_PATH)
SEAM_RESOLVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEAM_RESOLVER)


class SeamDuplicateResolverTest(unittest.TestCase):
    def setUp(self):
        self.image_width = 3760.0
        self.cfg = dict(
            enabled=True,
            geometry="wrap_xyxy",
            seam_band_px=400,
            match_iou=0.5,
            track_compat_iou=0.7,
            active_track_max_time_since_update=None,
            class_strict=True,
            fuse_method="score_weighted_box_max_score",
            debug_stats=True,
        )

    def test_wrap_iou_prefers_shifted_overlap(self):
        box_a = torch.tensor([[0.0, 10.0, 60.0, 110.0]])
        box_b = torch.tensor([[3760.0, 10.0, 3820.0, 110.0]])
        iou = SEAM_RESOLVER.wrap_iou(box_a, box_b, self.image_width)
        self.assertAlmostEqual(float(iou), 1.0, places=6)

    def test_merges_seam_duplicates_and_keeps_canonical_wrap_box(self):
        boxes = torch.tensor(
            [
                [0.0, 10.0, 100.0, 110.0],
                [3730.0, 10.0, 3830.0, 110.0],
            ]
        )
        scores = torch.tensor([0.9, 0.8])
        labels = torch.zeros(2, dtype=torch.long)
        merged_boxes, merged_scores, merged_labels, merged_quality, stats = (
            SEAM_RESOLVER.resolve_seam_duplicates_xyxy(
                boxes,
                scores,
                labels,
                image_width=self.image_width,
                seam_resolver_cfg=self.cfg,
            )
        )

        self.assertEqual(merged_boxes.shape[0], 1)
        self.assertGreater(float(merged_boxes[0, 2]), self.image_width)
        self.assertGreaterEqual(float(merged_boxes[0, 0]), 0.0)
        self.assertLess(float(merged_boxes[0, 0]), self.image_width)
        self.assertAlmostEqual(float(merged_scores[0]), 0.9, places=6)
        self.assertEqual(int(merged_labels[0]), 0)
        self.assertIsNone(merged_quality)
        self.assertEqual(stats["merged_candidates"], 1)

    def test_does_not_merge_distinct_objects_near_seam(self):
        boxes = torch.tensor(
            [
                [0.0, 10.0, 60.0, 110.0],
                [3730.0, 10.0, 3790.0, 110.0],
            ]
        )
        scores = torch.tensor([0.9, 0.8])
        labels = torch.zeros(2, dtype=torch.long)
        merged_boxes, _, _, _, stats = SEAM_RESOLVER.resolve_seam_duplicates_xyxy(
            boxes,
            scores,
            labels,
            image_width=self.image_width,
            seam_resolver_cfg=self.cfg,
        )

        self.assertEqual(merged_boxes.shape[0], 2)
        self.assertEqual(stats["merged_candidates"], 0)

    def test_track_compatibility_blocks_wrong_merge(self):
        cfg = dict(self.cfg)
        cfg["match_iou"] = 0.55
        cfg["track_compat_iou"] = 0.5
        boxes = torch.tensor(
            [
                [0.0, 10.0, 40.0, 110.0],
                [3730.0, 10.0, 3800.0, 110.0],
            ]
        )
        scores = torch.tensor([0.9, 0.85])
        labels = torch.zeros(2, dtype=torch.long)
        active_tracks = torch.tensor(
            [
                [0.0, 10.0, 20.0, 110.0],
                [3720.0, 10.0, 3770.0, 110.0],
            ]
        )

        merged_boxes, _, _, _, stats = SEAM_RESOLVER.resolve_seam_duplicates_xyxy(
            boxes,
            scores,
            labels,
            image_width=self.image_width,
            seam_resolver_cfg=cfg,
            active_tracks=active_tracks,
        )

        self.assertEqual(merged_boxes.shape[0], 2)
        self.assertEqual(stats["merged_candidates"], 0)

    def test_equivalent_active_tracks_do_not_deadlock_merge(self):
        boxes = torch.tensor(
            [
                [0.0, 10.0, 80.0, 110.0],
                [3740.0, 10.0, 3820.0, 110.0],
            ]
        )
        scores = torch.tensor([0.9, 0.85])
        labels = torch.zeros(2, dtype=torch.long)
        active_tracks = torch.tensor(
            [
                [0.0, 10.0, 80.0, 110.0],
                [3740.0, 10.0, 3820.0, 110.0],
            ]
        )
        cfg = dict(self.cfg)
        cfg["track_compat_iou"] = 0.5

        merged_boxes, _, _, _, stats = SEAM_RESOLVER.resolve_seam_duplicates_xyxy(
            boxes,
            scores,
            labels,
            image_width=self.image_width,
            seam_resolver_cfg=cfg,
            active_tracks=active_tracks,
        )

        self.assertEqual(merged_boxes.shape[0], 1)
        self.assertEqual(stats["merged_candidates"], 1)

    def test_near_threshold_active_tracks_still_merge_seam_duplicates(self):
        boxes = torch.tensor(
            [
                [3703.0, 40.0, 3826.0, 386.0],
                [0.0, 39.0, 66.0, 390.0],
            ]
        )
        scores = torch.tensor([0.95, 0.9])
        labels = torch.zeros(2, dtype=torch.long)
        active_tracks = torch.tensor(
            [
                [3696.0, 37.0, 3826.0, 390.0],
                [0.0, 36.0, 64.0, 394.0],
            ]
        )

        merged_boxes, _, _, _, stats = SEAM_RESOLVER.resolve_seam_duplicates_xyxy(
            boxes,
            scores,
            labels,
            image_width=self.image_width,
            seam_resolver_cfg=self.cfg,
            active_tracks=active_tracks,
        )

        self.assertEqual(merged_boxes.shape[0], 1)
        self.assertEqual(stats["merged_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
