import importlib.util
import pathlib
import unittest


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "projects/mmdet3d_plugin/models/omnidetr/active_track_utils.py"
)
SPEC = importlib.util.spec_from_file_location("active_track_utils", MODULE_PATH)
ACTIVE_TRACK_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACTIVE_TRACK_UTILS)


class DummyTrack:
    def __init__(
        self,
        *,
        track_id,
        time_since_update,
        last_observation,
        state_box,
    ):
        self.track_id = track_id
        self.time_since_update = time_since_update
        self.last_observation = last_observation
        self._state_box = state_box

    def get_state(self):
        return self._state_box


class ActiveTrackUtilsTest(unittest.TestCase):
    def test_filters_stale_tracks_before_box_collection(self):
        tracks = [
            DummyTrack(
                track_id=10,
                time_since_update=0,
                last_observation=[0.0, 10.0, 50.0, 90.0],
                state_box=[0.0, 10.0, 50.0, 90.0],
            ),
            DummyTrack(
                track_id=11,
                time_since_update=2,
                last_observation=[3700.0, 10.0, 3820.0, 90.0],
                state_box=[3700.0, 10.0, 3820.0, 90.0],
            ),
        ]

        data = ACTIVE_TRACK_UTILS.collect_active_track_data(
            tracks,
            max_time_since_update=0,
        )

        self.assertEqual(data["boxes"].shape[0], 1)
        self.assertEqual(data["retained"][0]["track_id"], 10)
        self.assertEqual(data["dropped"][0]["track_id"], 11)
        self.assertEqual(data["dropped"][0]["drop_reason"], "stale_time_since_update")

    def test_retains_valid_wrap_boxes_and_debug_metadata(self):
        tracks = [
            DummyTrack(
                track_id=21,
                time_since_update=0,
                last_observation=[-15.0, 10.0, 25.0, 90.0],
                state_box=[-20.0, 8.0, 30.0, 92.0],
            ),
            DummyTrack(
                track_id=22,
                time_since_update=0,
                last_observation=[1.0, 1.0, 1.0, 10.0],
                state_box=[3730.0, 12.0, 3830.0, 108.0],
            ),
        ]

        data = ACTIVE_TRACK_UTILS.collect_active_track_data(
            tracks,
            max_time_since_update=None,
        )

        self.assertEqual(data["boxes"].shape[0], 2)
        self.assertEqual(data["retained"][0]["selected_box_source"], "last_observation")
        self.assertEqual(data["retained"][0]["selected_box"], [-15.0, 10.0, 25.0, 90.0])
        self.assertEqual(data["retained"][1]["selected_box_source"], "state_box")
        self.assertEqual(data["retained"][1]["state_box"], [3730.0, 12.0, 3830.0, 108.0])


if __name__ == "__main__":
    unittest.main()
