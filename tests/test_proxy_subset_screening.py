import csv
import importlib.util
import pathlib
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools/build_a_proxy_subset_screening.py"
SPEC = importlib.util.spec_from_file_location("proxy_subset_screening", MODULE_PATH)
PROXY_SCREENING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROXY_SCREENING)


class ProxySubsetScreeningUnitTest(unittest.TestCase):
    def test_box_seam_flags_boundary_cases(self):
        self.assertEqual(
            PROXY_SCREENING.box_seam_flags(
                x=500,
                width=100,
                image_width=PROXY_SCREENING.IMAGE_WIDTH,
                seam_band_px=PROXY_SCREENING.SEAM_BAND_PX,
            ),
            (False, False),
        )
        self.assertEqual(
            PROXY_SCREENING.box_seam_flags(
                x=399,
                width=20,
                image_width=PROXY_SCREENING.IMAGE_WIDTH,
                seam_band_px=PROXY_SCREENING.SEAM_BAND_PX,
            ),
            (True, False),
        )
        self.assertEqual(
            PROXY_SCREENING.box_seam_flags(
                x=3300,
                width=61,
                image_width=PROXY_SCREENING.IMAGE_WIDTH,
                seam_band_px=PROXY_SCREENING.SEAM_BAND_PX,
            ),
            (True, False),
        )
        self.assertEqual(
            PROXY_SCREENING.box_seam_flags(
                x=-1,
                width=20,
                image_width=PROXY_SCREENING.IMAGE_WIDTH,
                seam_band_px=PROXY_SCREENING.SEAM_BAND_PX,
            ),
            (True, True),
        )
        self.assertEqual(
            PROXY_SCREENING.box_seam_flags(
                x=3750,
                width=20,
                image_width=PROXY_SCREENING.IMAGE_WIDTH,
                seam_band_px=PROXY_SCREENING.SEAM_BAND_PX,
            ),
            (True, True),
        )


@unittest.skipUnless(
    PROXY_SCREENING.DEFAULT_LABELS_ROOT.exists(),
    "JRDB stitched label JSONs are required for the integration test.",
)
class ProxySubsetScreeningIntegrationTest(unittest.TestCase):
    def test_build_outputs_match_fixed_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = PROXY_SCREENING.build_proxy_subset_screening(out_dir=tmp_dir)
            out_dir = pathlib.Path(tmp_dir)

            train_whitelist = (out_dir / "train_whitelist.txt").read_text().splitlines()
            val_whitelist = (out_dir / "val_whitelist.txt").read_text().splitlines()
            note = (out_dir / "selection_note.md").read_text()

            self.assertEqual(
                train_whitelist, list(PROXY_SCREENING.PROXY_TRAIN_WHITELIST)
            )
            self.assertEqual(val_whitelist, list(PROXY_SCREENING.PROXY_VAL_WHITELIST))

            with (out_dir / "sequence_stats.csv").open() as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(
                    tuple(reader.fieldnames),
                    PROXY_SCREENING.SEQUENCE_STATS_FIELDNAMES,
                )

            self.assertEqual(len(rows), PROXY_SCREENING.EXPECTED_SEQUENCE_COUNT)
            self.assertEqual(
                sum(row["split"] == "train" for row in rows),
                PROXY_SCREENING.EXPECTED_TRAIN_SEQUENCE_COUNT,
            )
            self.assertEqual(
                sum(row["split"] == "val" for row in rows),
                PROXY_SCREENING.EXPECTED_VAL_SEQUENCE_COUNT,
            )

            label_sequences = sorted(
                path.stem
                for path in PROXY_SCREENING.DEFAULT_LABELS_ROOT.glob("*.json")
            )
            self.assertEqual(sorted(row["sequence"] for row in rows), label_sequences)

            self.assertGreaterEqual(summary["proxy_train_ratio"], 0.10)
            self.assertLessEqual(summary["proxy_train_ratio"], 0.15)
            self.assertEqual(
                summary["proxy_train_gt"], PROXY_SCREENING.EXPECTED_PROXY_TRAIN_GT
            )
            self.assertEqual(
                summary["train_gt_total"], PROXY_SCREENING.EXPECTED_TRAIN_GT_TOTAL
            )

            self.assertIn("13.7%", note)
            for sequence in (
                *PROXY_SCREENING.PROXY_TRAIN_WHITELIST,
                *PROXY_SCREENING.PROXY_VAL_WHITELIST,
            ):
                self.assertIn(sequence, note)


if __name__ == "__main__":
    unittest.main()
