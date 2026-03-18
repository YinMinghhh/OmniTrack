#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export TRACKING_MODE=${TRACKING_MODE:-tbd}
export TBD_BACKEND=${TBD_BACKEND:-hybridsort}
export PKL_OUT=${PKL_OUT:-work_dirs/jrdb2019_4g_bs2_tbd_hybridsort/results_test.pkl}
export SUBMISSION_ROOT=${SUBMISSION_ROOT:-results/test_submission_tbd_hybridsort}

bash "$ROOT_DIR/scripts/run_test_submission.sh"
