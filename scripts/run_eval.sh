#!/bin/bash
set -e # 遇到错误立即停止

# ==================== 配置区 ====================
GPU_ID=1
CONFIG="projects/configs/JRDB_OmniTrack.py"
CHECKPOINT="work_dirs/jrdb2019_4g_bs2/iter_47565.pth"
PKL_OUT="work_dirs/jrdb2019_4g_bs2/results.pkl"
JSON_OUT="results/submission/results_jrdb2d.json" # test.py 默认输出路径
WORKSPACE="$(pwd)/evaluation_workspace"
# ===============================================

echo "=== 1. Inference (Model Prediction) ==="
# 注意：OmniTrack 的 test.py 会自动生成 JSON 到固定目录，我们主要依赖那个 JSON
CUDA_VISIBLE_DEVICES=$GPU_ID python tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --out $PKL_OUT \
    --eval track

echo "=== 2. Format Conversion (Pred -> KITTI) ==="
python tools/convert_pred_json2kitti.py \
    --pred_json $JSON_OUT \
    --output_dir "$WORKSPACE/pred/JRDB-train"

echo "=== 3. Environment Setup (GT & Seqmap & Padding) ==="
# 这一步负责生成 GT、生成 Seqmap 并补全缺失文件
python tools/prepare_eval_env.py --workspace "$WORKSPACE"

echo "=== 4. HOTA Evaluation ==="
cd jrdb_toolkit/tracking_eval/TrackEval

# 关键参数说明：
# TRACKER_SUB_FOLDER "": 告诉代码直接在 Pred 根目录找，别去 data/ 子目录找
# GT_FOLDER: 指向 workspace/gt，代码会自动在下面找 label_02 和 seqmap
python scripts/run_jrdb.py \
    --GT_FOLDER "$WORKSPACE/gt" \
    --TRACKERS_FOLDER "$WORKSPACE/pred" \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL JRDB-train \
    --CLASSES_TO_EVAL pedestrian \
    --PRINT_RESULTS True \
    --TRACKER_SUB_FOLDER ""

cd -

echo "=== Done! ==="