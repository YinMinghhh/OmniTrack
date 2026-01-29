#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PORT=${PORT:-29532}

CONFIG="projects/configs/JRDB_OmniTrack.py"
WORK_DIR="work_dirs/jrdb2019_4g_bs2"

bash tools/dist_train.sh "${CONFIG}" 4 \
  --work-dir "${WORK_DIR}" \
  --cfg-options \
    "find_unused_parameters=True" \
    "data.samples_per_gpu=4" \
    "runner.max_iters=135900" \
    "checkpoint_config.interval=6795" \
    "model.img_backbone.with_cp=False"
