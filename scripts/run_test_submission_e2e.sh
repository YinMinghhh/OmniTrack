#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

GPU_ID=${GPU_ID:-1}
CONFIG=${CONFIG:-projects/configs/JRDB_OmniTrack.py}
CHECKPOINT=${CHECKPOINT:-work_dirs/jrdb2019_4g_bs2/iter_135900.pth}
INFER_SPLIT=${INFER_SPLIT:-test}
TRACKING_MODE=${TRACKING_MODE:-e2e}
TBD_BACKEND=${TBD_BACKEND:-hybridsort}
EXTRA_CFG_OPTIONS=${EXTRA_CFG_OPTIONS:-}
ANN_ROOT=data/JRDB2019_2d_stitched_anno_pkls
TEST_ANN_FILE=${TEST_ANN_FILE:-$ANN_ROOT/JRDB_infos_test_v1.2.pkl}
PKL_OUT=${PKL_OUT:-work_dirs/jrdb2019_4g_bs2/results_test.pkl}
SUBMISSION_ROOT=${SUBMISSION_ROOT:-results/test_submission}
JSON_DIR=${JSON_DIR:-$SUBMISSION_ROOT/raw_json}
JSON_OUT=${JSON_OUT:-$JSON_DIR/results_jrdb2d.json}
JSON_DIR=$(dirname "$JSON_OUT")
SUBMISSION_DATA_DIR=${SUBMISSION_DATA_DIR:-$SUBMISSION_ROOT/CIWT/data}
SUBMISSION_ZIP=${SUBMISSION_ZIP:-$SUBMISSION_ROOT/jrdb_2dt_submission.zip}
SEQUENCE_MAP_OUT=${SEQUENCE_MAP_OUT:-$SUBMISSION_ROOT/sequence_index_map.csv}
SUBMISSION_BOX_FORMAT=${SUBMISSION_BOX_FORMAT:-xywh}

if [[ "$INFER_SPLIT" != "test" ]]; then
  echo "Unsupported INFER_SPLIT: $INFER_SPLIT (expected test for this script)" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$(dirname "$PKL_OUT")" "$JSON_DIR" "$SUBMISSION_DATA_DIR"

CFG_OPTIONS=(
  "data.test.ann_file=$TEST_ANN_FILE"
  "model.head.instance_bank.tracking_mode=$TRACKING_MODE"
  "model.head.instance_bank.tbd_backend=$TBD_BACKEND"
)
if [[ -n "$EXTRA_CFG_OPTIONS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_CFG_OPTIONS_ARR=($EXTRA_CFG_OPTIONS)
  CFG_OPTIONS+=("${EXTRA_CFG_OPTIONS_ARR[@]}")
fi

if [[ ! -f "$TEST_ANN_FILE" ]]; then
  echo "Missing test annotation file: $TEST_ANN_FILE" >&2
  exit 1
fi

echo "=== Test Submission Config ==="
echo "Inference split: $INFER_SPLIT"
echo "Inference ann_file: $TEST_ANN_FILE"
echo "Submission root: $SUBMISSION_ROOT"
echo "Submission box format: $SUBMISSION_BOX_FORMAT"
echo "Tracking mode: $TRACKING_MODE"
echo "TBD backend: $TBD_BACKEND"

echo "=== 1/3 Inference (model -> JSON/PKL, official test) ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" tools/test.py \
  "$CONFIG" \
  "$CHECKPOINT" \
  --out "$PKL_OUT" \
  --format-only \
  --eval-options "jsonfile_prefix=$JSON_DIR" \
  --cfg-options "${CFG_OPTIONS[@]}"

if [[ ! -f "$JSON_OUT" ]]; then
  echo "Expected JSON output not found: $JSON_OUT" >&2
  exit 1
fi

echo "=== 2/3 Format Conversion (JSON -> JRDB official submission) ==="
"$PYTHON_BIN" tools/convert_pred_json_to_jrdb_submission.py \
  --pred-json "$JSON_OUT" \
  --ann-file "$TEST_ANN_FILE" \
  --output-dir "$SUBMISSION_DATA_DIR" \
  --zip-output "$SUBMISSION_ZIP" \
  --zip-source-dir "$SUBMISSION_ROOT/CIWT" \
  --sequence-map-output "$SEQUENCE_MAP_OUT" \
  --box-format "$SUBMISSION_BOX_FORMAT"

echo "=== 3/3 Done ==="
echo "JSON output: $JSON_OUT"
echo "Submission txt dir: $SUBMISSION_DATA_DIR"
echo "Sequence map: $SEQUENCE_MAP_OUT"
echo "Submission zip: $SUBMISSION_ZIP"
