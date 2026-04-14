#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

source /home/SNN/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/sdb/ym/envs/OmniTrack-clean

GPU=${GPU:-0}
PORT_BASE=${PORT_BASE:-29670}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}

ANN_FILE="data/JRDB2019_2d_stitched_anno_pkls/JRDB_infos_val_v1.2.pkl"
CHECKPOINT="ckpt/jrdb2019_baseline_iter_135900.pth"
CONFIG="projects/configs/JRDB_OmniTrack.py"
GT_DIR="jrdb_toolkit/tracking_eval/TrackEval/data/gt/jrdb/jrdb_2d_box_val"
TRACKERS_DIR="jrdb_toolkit/tracking_eval/TrackEval/data/trackers/jrdb/jrdb_2d_box_val"
BATCH_ROOT="work_dirs/b2_mixed_${TS}"
DIAG_DIR="work_dirs/diag_b2_mixed_${TS}"
REPORT_DIR="${BATCH_ROOT}/report"
VARIANTS_CSV="${BATCH_ROOT}/variants.csv"

mkdir -p "$BATCH_ROOT"
printf 'variant_id,display_name,association_mode,freshness,tracker_name,run_dir,runtime_sec\n' > "$VARIANTS_CSV"

VARIANT_IDS=(
  legacy
  legacy_fresh0
  selective_spherical
  selective_spherical_fresh0
  planar_gate_spherical_rerank
  planar_gate_spherical_rerank_fresh0
)
DISPLAY_NAMES=(
  "legacy"
  "legacy + fresh0"
  "selective spherical"
  "selective spherical + fresh0"
  "planar gate + spherical rerank"
  "planar gate + spherical rerank + fresh0"
)
ASSOCIATION_MODES=(
  planar_legacy
  planar_legacy
  selective_spherical
  selective_spherical
  planar_gate_spherical_rerank
  planar_gate_spherical_rerank
)
FRESHNESS_LABELS=(
  legacy
  fresh0
  legacy
  fresh0
  legacy
  fresh0
)
TRACKER_NAMES=(
  "OmniTrackB2Legacy_${TS}"
  "OmniTrackB2LegacyFresh0_${TS}"
  "OmniTrackB2Selective_${TS}"
  "OmniTrackB2SelectiveFresh0_${TS}"
  "OmniTrackB2PlanarGateRerank_${TS}"
  "OmniTrackB2PlanarGateRerankFresh0_${TS}"
)

run_variant() {
  local idx=$1
  local variant_id=${VARIANT_IDS[$idx]}
  local display_name=${DISPLAY_NAMES[$idx]}
  local association_mode=${ASSOCIATION_MODES[$idx]}
  local freshness=${FRESHNESS_LABELS[$idx]}
  local tracker_name=${TRACKER_NAMES[$idx]}
  local port=$((PORT_BASE + idx))
  local run_dir="work_dirs/eval_jrdb_val_b2_${variant_id}_${TS}"
  local json_prefix="${run_dir}/raw_json/results"
  local export_log="${run_dir}/export.log"
  local trackeval_log="${run_dir}/trackeval.log"
  local val_log="${run_dir}/val.log"
  local start_ts
  local end_ts
  local runtime_sec

  mkdir -p "$run_dir"
  start_ts=$(date +%s)

  local cfg_opts=(
    data.workers_per_gpu=0
    model.head.instance_bank.tracking_mode=tbd
    model.head.instance_bank.tbd_backend=hybridsort
    model.head.instance_bank.tbd_tracker_cfg.association_geometry.mode="${association_mode}"
  )
  if [[ "$freshness" == "fresh0" ]]; then
    cfg_opts+=(
      model.head.instance_bank.seam_resolver_cfg.active_track_max_time_since_update=0
    )
  fi

  echo "[$variant_id] inference -> $run_dir"
  CUDA_VISIBLE_DEVICES="$GPU" PORT="$port" \
    bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" 1 \
    --out "${run_dir}/results_val.pkl" \
    --format-only \
    --eval-options "jsonfile_prefix=${json_prefix}" \
    --cfg-options "${cfg_opts[@]}" \
    > "$val_log" 2>&1

  local export_args=(
    python tools/export_jrdb_trackeval_2d.py
    --ann-file "$ANN_FILE"
    --pred-json "${run_dir}/raw_json/results/results_jrdb2d.json"
    --split-name val
    --tracker-name "$tracker_name"
    --gt-out-dir "$GT_DIR"
    --trackers-out-dir "$TRACKERS_DIR"
  )
  if [[ "$idx" -gt 0 ]]; then
    export_args+=(--skip-gt)
  fi
  echo "[$variant_id] export -> $tracker_name"
  "${export_args[@]}" > "$export_log" 2>&1

  echo "[$variant_id] trackeval -> $tracker_name"
  (
    cd jrdb_toolkit/tracking_eval/TrackEval
    PYTHONPATH=. python scripts/run_jrdb.py \
      --USE_PARALLEL False \
      --GT_FOLDER data/gt/jrdb/jrdb_2d_box_val \
      --TRACKERS_FOLDER data/trackers/jrdb/jrdb_2d_box_val \
      --SPLIT_TO_EVAL val \
      --TRACKERS_TO_EVAL "$tracker_name" \
      --METRICS HOTA CLEAR Identity OSPA \
      --PRINT_ONLY_COMBINED True \
      --PLOT_CURVES False
  ) > "$trackeval_log" 2>&1

  end_ts=$(date +%s)
  runtime_sec=$((end_ts - start_ts))
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$variant_id" \
    "$display_name" \
    "$association_mode" \
    "$freshness" \
    "$tracker_name" \
    "$run_dir" \
    "$runtime_sec" >> "$VARIANTS_CSV"
  echo "[$variant_id] done in ${runtime_sec}s"
}

for idx in "${!VARIANT_IDS[@]}"; do
  run_variant "$idx"
done

python tools/diag_seam_metric_split.py \
  --tracker-names "${TRACKER_NAMES[@]}" \
  --gt-folder "$GT_DIR" \
  --trackers-folder "$TRACKERS_DIR" \
  --split-name val \
  --image-width 3760 \
  --image-height 480 \
  --seam-band-px 400 \
  --high-lat-deg 45 \
  --bad-case-top-k 5 \
  --out-dir "$DIAG_DIR"

python tools/report_b2_mixed_geometry.py \
  --variants-csv "$VARIANTS_CSV" \
  --diag-dir "$DIAG_DIR" \
  --gt-folder "$GT_DIR" \
  --trackers-folder "$TRACKERS_DIR" \
  --split-name val \
  --image-width 3760 \
  --image-height 480 \
  --seam-band-px 400 \
  --high-lat-deg 45 \
  --frozen-bad-case-seqs \
    tressider-2019-04-26_2 \
    clark-center-2019-02-28_1 \
    gates-ai-lab-2019-02-08_0 \
    huang-2-2019-01-25_0 \
    nvidia-aud-2019-04-18_0 \
  --out-dir "$REPORT_DIR"

echo "B2 matrix finished."
echo "Batch root: $BATCH_ROOT"
echo "Diag root: $DIAG_DIR"
echo "Report dir: $REPORT_DIR"
