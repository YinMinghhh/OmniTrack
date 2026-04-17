#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault(
        "OMNITRACK_A2_ROOT",
        str(repo_root / "research/seam-a/A2_rolllite_clean"),
    )
    env.setdefault(
        "OMNITRACK_A2_A_CONFIG",
        str(repo_root / "projects/configs/JRDB_OmniTrack_wt_a_circular_padding_rollaug_rolllite.py"),
    )
    env.setdefault(
        "OMNITRACK_A2_SUMMARY_TITLE",
        "A2 Roll-Lite Clean Stage Gate Summary",
    )
    runner = repo_root / "tools/run_a2_two_stage_training.py"
    os.execvpe(sys.executable, [sys.executable, str(runner), *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
