#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

GPU_ID=${GPU_ID:-1}
CONFIG=${CONFIG:-projects/configs/JRDB_OmniTrack.py}
CHECKPOINT=${CHECKPOINT:-work_dirs/jrdb2019_4g_bs2/iter_47565.pth}
PKL_OUT=${PKL_OUT:-work_dirs/jrdb2019_4g_bs2/results.pkl}
JSON_OUT=${JSON_OUT:-results/submission/results_jrdb2d.json}
WORKSPACE=${WORKSPACE:-"$ROOT_DIR/evaluation_workspace"}

cd "$ROOT_DIR"

echo "=== 1/6 Inference (model -> JSON/PKL) ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" tools/test.py \
  "$CONFIG" \
  "$CHECKPOINT" \
  --out "$PKL_OUT" \
  --eval track

echo "=== 2/6 Sanity Check (JSON structure) ==="
"$PYTHON_BIN" tools/eval_sanity_check.py --pred-json "$JSON_OUT"

echo "=== 3/6 Format Conversion (JSON -> TrackEval TXT) ==="
"$PYTHON_BIN" tools/convert_pred_json2kitti.py \
  --pred_json "$JSON_OUT" \
  --box_format xywh \
  --output_dir "$WORKSPACE/pred/JRDB-train"

echo "=== 4/6 Environment Setup (GT + seqmap + missing seq padding) ==="
"$PYTHON_BIN" tools/prepare_eval_env.py --workspace "$WORKSPACE" --box_format xywh

echo "=== 5/6 Sanity Check (workspace contract) ==="
"$PYTHON_BIN" tools/eval_sanity_check.py --workspace "$WORKSPACE"

echo "=== 6/6 HOTA Evaluation (TrackEval) ==="
pushd jrdb_toolkit/tracking_eval/TrackEval >/dev/null
"$PYTHON_BIN" scripts/run_jrdb.py \
  --GT_FOLDER "$WORKSPACE/gt" \
  --TRACKERS_FOLDER "$WORKSPACE/pred" \
  --SPLIT_TO_EVAL train \
  --TRACKERS_TO_EVAL JRDB-train \
  --CLASSES_TO_EVAL pedestrian \
  --PRINT_RESULTS True \
  --TRACKER_SUB_FOLDER ""
popd >/dev/null

echo "=== Done ==="
