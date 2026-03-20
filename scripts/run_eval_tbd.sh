#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export TRACKING_MODE=${TRACKING_MODE:-tbd}
export TBD_BACKEND=${TBD_BACKEND:-hybridsort}
export PKL_OUT=${PKL_OUT:-work_dirs/jrdb2019_4g_bs2_tbd_hybridsort/results.pkl}
export JSON_OUT=${JSON_OUT:-results/submission_tbd_hybridsort/results_jrdb2d.json}
export WORKSPACE=${WORKSPACE:-"$ROOT_DIR/results/eval/jrdb_tbd_hybridsort"}

bash "$ROOT_DIR/scripts/run_eval_e2e.sh"
