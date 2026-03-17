#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PORT=${PORT:-29532}

CONFIG=${CONFIG:-projects/configs/JRDB_OmniTrack.py}
WORK_DIR=${WORK_DIR:-work_dirs/jrdb2019_4g_bs2}
TRACKING_MODE=${TRACKING_MODE:-e2e}
TBD_BACKEND=${TBD_BACKEND:-hybridsort}
EXTRA_CFG_OPTIONS=${EXTRA_CFG_OPTIONS:-}

CFG_OPTIONS=(
  "find_unused_parameters=True"
  "data.samples_per_gpu=4"
  "runner.max_iters=135900"
  "checkpoint_config.interval=6795"
  "model.img_backbone.with_cp=False"
  "model.head.instance_bank.tracking_mode=$TRACKING_MODE"
  "model.head.instance_bank.tbd_backend=$TBD_BACKEND"
)
if [[ -n "$EXTRA_CFG_OPTIONS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_CFG_OPTIONS_ARR=($EXTRA_CFG_OPTIONS)
  CFG_OPTIONS+=("${EXTRA_CFG_OPTIONS_ARR[@]}")
fi

bash tools/dist_train.sh "${CONFIG}" 4 \
  --work-dir "${WORK_DIR}" \
  --cfg-options "${CFG_OPTIONS[@]}"
