#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

GPU_ID=${GPU_ID:-1}
CONFIG=${CONFIG:-projects/configs/JRDB_OmniTrack.py}
CHECKPOINT=${CHECKPOINT:-work_dirs/jrdb2019_4g_bs2/iter_135900.pth}
PKL_OUT=${PKL_OUT:-work_dirs/jrdb2019_4g_bs2/results.pkl}
JSON_OUT=${JSON_OUT:-results/submission/results_jrdb2d.json}
WORKSPACE=${WORKSPACE:-"$ROOT_DIR/evaluation_workspace"}
INFER_SPLIT=${INFER_SPLIT:-val}
ANN_ROOT=data/JRDB2019_2d_stitched_anno_pkls
PRED_DIR="$WORKSPACE/pred/JRDB-train"

case "$INFER_SPLIT" in
  train)
    TEST_ANN_FILE="$ANN_ROOT/JRDB_infos_train_v1.2.pkl"
    ;;
  val)
    TEST_ANN_FILE="$ANN_ROOT/JRDB_infos_val_v1.2.pkl"
    ;;
  *)
    echo "Unsupported INFER_SPLIT: $INFER_SPLIT (expected train or val)" >&2
    exit 1
    ;;
esac

cd "$ROOT_DIR"

echo "=== Eval Config ==="
echo "Inference split: $INFER_SPLIT"
echo "Inference ann_file: $TEST_ANN_FILE"

echo "=== 1/7 Inference (model -> JSON/PKL) ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" tools/test.py \
  "$CONFIG" \
  "$CHECKPOINT" \
  --out "$PKL_OUT" \
  --eval track \
  --cfg-options "data.test.ann_file=$TEST_ANN_FILE"

echo "=== 2/7 Sanity Check (JSON structure) ==="
"$PYTHON_BIN" tools/eval_sanity_check.py --pred-json "$JSON_OUT"

echo "=== 3/7 Format Conversion (JSON -> TrackEval TXT) ==="
mkdir -p "$PRED_DIR"
shopt -s nullglob
stale_pred_files=("$PRED_DIR"/*.txt "$PRED_DIR"/*.csv)
if ((${#stale_pred_files[@]})); then
  rm -f "${stale_pred_files[@]}"
fi
shopt -u nullglob
"$PYTHON_BIN" tools/convert_pred_json2kitti.py \
  --pred_json "$JSON_OUT" \
  --box_format xywh \
  --output_dir "$PRED_DIR"

echo "=== 4/7 Environment Setup (GT + seqmap + missing seq padding) ==="
"$PYTHON_BIN" tools/prepare_eval_env.py \
  --workspace "$WORKSPACE" \
  --box_format xywh \
  --split auto \
  --pred-json "$JSON_OUT"

echo "=== 5/7 Sanity Check (workspace contract) ==="
"$PYTHON_BIN" tools/eval_sanity_check.py --workspace "$WORKSPACE"

EVAL_SPLIT=$(<"$WORKSPACE/gt/eval_split.txt")
echo "=== 6/7 HOTA Evaluation (TrackEval, split=$EVAL_SPLIT) ==="
pushd jrdb_toolkit/tracking_eval/TrackEval >/dev/null
"$PYTHON_BIN" scripts/run_jrdb.py \
  --GT_FOLDER "$WORKSPACE/gt" \
  --TRACKERS_FOLDER "$WORKSPACE/pred" \
  --SPLIT_TO_EVAL "$EVAL_SPLIT" \
  --TRACKERS_TO_EVAL JRDB-train \
  --CLASSES_TO_EVAL pedestrian \
  --PRINT_RESULTS True \
  --TRACKER_SUB_FOLDER ""
popd >/dev/null

echo "=== 7/7 Post-Eval Summary ==="
"$PYTHON_BIN" tools/eval_sanity_check.py --workspace "$WORKSPACE"

echo "=== Done ==="
