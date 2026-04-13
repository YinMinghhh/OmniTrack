import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_module(relative_path, module_name):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_package(package_name, relative_path):
    if package_name in sys.modules:
        return sys.modules[package_name]
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO_ROOT / relative_path)]
    sys.modules[package_name] = package
    return package


def load_dataset_module():
    ensure_package("projects", "projects")
    ensure_package("projects.mmdet3d_plugin", "projects/mmdet3d_plugin")
    ensure_package("projects.mmdet3d_plugin.core", "projects/mmdet3d_plugin/core")
    ensure_package(
        "projects.mmdet3d_plugin.datasets", "projects/mmdet3d_plugin/datasets"
    )
    load_module(
        "projects/mmdet3d_plugin/core/box3d.py",
        "projects.mmdet3d_plugin.core.box3d",
    )
    load_module(
        "projects/mmdet3d_plugin/datasets/utils.py",
        "projects.mmdet3d_plugin.datasets.utils",
    )
    return load_module(
        "projects/mmdet3d_plugin/datasets/JRDB_2d_det_track_dataset.py",
        "projects.mmdet3d_plugin.datasets.JRDB_2d_det_track_dataset",
    )


CONV = load_module(
    "projects/mmdet3d_plugin/models/omnidetr/conv.py",
    "omnitrack_conv",
)
AUGMENT = load_module(
    "projects/mmdet3d_plugin/datasets/pipelines/augment.py",
    "omnitrack_augment",
)
DATASET = load_dataset_module()
EXPORT = load_module(
    "tools/export_jrdb_trackeval_2d.py",
    "omnitrack_export_trackeval",
)


class CircularWidthPaddingTest(unittest.TestCase):
    def _freeze_bn_identity(self, module):
        module.eval()
        module.bn.weight.data.fill_(1.0)
        module.bn.bias.data.zero_()
        module.bn.running_mean.zero_()
        module.bn.running_var.fill_(1.0)

    def test_apply_width_circular_padding_keeps_height_non_periodic(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])
        padded = CONV.apply_width_circular_padding(x, (1, 1))
        expected = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [3.0, 1.0, 2.0, 3.0, 1.0],
                        [6.0, 4.0, 5.0, 6.0, 4.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.assertTrue(torch.equal(padded, expected))

    def test_circular_width_conv_wraps_left_and_right_edges(self):
        conv = CONV.CircularWidthConv(1, 1, k=3, s=1, p=1, act=False)
        self._freeze_bn_identity(conv)
        conv.conv.weight.data.fill_(1.0)

        x = torch.tensor([[[[1.0, 2.0, 3.0]]]])
        output = conv(x)
        expected = torch.tensor([[[[6.0, 6.0, 6.0]]]])
        self.assertTrue(torch.allclose(output, expected, atol=1e-4))

    def test_circular_width_repconv_only_wraps_3x3_branch(self):
        repconv = CONV.CircularWidthRepConv(1, 1, act=False, bn=False)
        self._freeze_bn_identity(repconv.conv1)
        self._freeze_bn_identity(repconv.conv2)
        repconv.conv1.conv.weight.data.fill_(1.0)
        repconv.conv2.conv.weight.data.zero_()

        x = torch.tensor([[[[1.0, 2.0, 3.0]]]])
        output = repconv(x)
        expected = torch.tensor([[[[6.0, 6.0, 6.0]]]])
        self.assertTrue(torch.allclose(output, expected, atol=1e-4))


class RollAugmentationTest(unittest.TestCase):
    def test_roll_stitched_image_moves_pixels_and_bbox_centers(self):
        results = {
            "aug_config": {"roll_px": 2, "roll_image_width": 8},
            "img": [np.arange(8, dtype=np.float32).reshape(1, 8, 1)],
            "gt_bboxes_2d": np.array([[7.0, 0.5, 2.0, 1.0]], dtype=np.float32),
        }
        rolled = AUGMENT.RollStitchedImageJRDB2D()(results)
        self.assertEqual(int(rolled["roll_px"]), 2)
        self.assertTrue(
            np.array_equal(
                rolled["img"][0][0, :, 0],
                np.array([6, 7, 0, 1, 2, 3, 4, 5], dtype=np.float32),
            )
        )
        self.assertAlmostEqual(float(rolled["gt_bboxes_2d"][0, 0]), 1.0)

    def test_roll_then_extend_preserves_seam_object_via_duplicate_box(self):
        results = {
            "aug_config": {
                "roll_px": 2,
                "roll_image_width": 8,
                "crop": (0, 0, 10, 2),
            },
            "img": [np.zeros((2, 8, 3), dtype=np.uint8)],
            "pad_shape": (2, 8, 3),
            "gt_bboxes_2d": np.array([[7.0, 1.0, 2.0, 1.0]], dtype=np.float32),
            "gt_labels_2d": np.array([0], dtype=np.int64),
            "gt_names": np.array(["pedestrian"], dtype=object),
            "instance_inds": np.array([1], dtype=np.int64),
        }
        roll = AUGMENT.RollStitchedImageJRDB2D()
        extend = AUGMENT.ExtendStitchedImageJRDB2D()
        bbox_extend = AUGMENT.BBoxExtendJRDB2DDETR()

        results = roll(results)
        results = extend(results)
        results = bbox_extend(results)

        centers = sorted(float(center) for center in results["gt_bboxes_2d"][:, 0])
        self.assertEqual(centers, [1.0, 9.0])


class SequenceAndExportHelperTest(unittest.TestCase):
    def test_filter_data_infos_by_sequence(self):
        infos = [
            {"token": "clark-center-2019-02-28_1_000001"},
            {"token": "nvidia-aud-2019-04-18_0_000001"},
        ]
        filtered = DATASET.filter_data_infos_by_sequence(
            infos, {"clark-center-2019-02-28_1"}
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["token"], "clark-center-2019-02-28_1_000001")

    def test_build_split_index_and_inverse_roll(self):
        infos = [
            {"token": "clark-center-2019-02-28_1_000001"},
            {"token": "clark-center-2019-02-28_1_000003"},
            {"token": "nvidia-aud-2019-04-18_0_000002"},
        ]
        seq_to_frames = EXPORT.build_split_index(
            infos, sequence_names=["clark-center-2019-02-28_1"]
        )
        self.assertEqual(seq_to_frames, {"clark-center-2019-02-28_1": [1, 3]})
        self.assertAlmostEqual(
            EXPORT.inverse_roll_x(100.0, inverse_roll_px=940.0, image_width=3760.0),
            2920.0,
        )


if __name__ == "__main__":
    unittest.main()
