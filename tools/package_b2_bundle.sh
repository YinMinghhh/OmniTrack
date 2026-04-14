#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <batch_root> <diag_dir> <report_dir> [bundle_root]" >&2
  exit 1
fi

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

BATCH_ROOT=$1
DIAG_DIR=$2
REPORT_DIR=$3
BUNDLE_ROOT=${4:-"work_dirs/review_bundle_b2_$(date +%Y%m%d_%H%M%S)"}

mkdir -p "$BUNDLE_ROOT"/{code,tests,results,report,patches}

copy_file_into() {
  local src=$1
  local dst_root=$2
  mkdir -p "$dst_root/$(dirname "$src")"
  cp -a "$src" "$dst_root/$src"
}

copy_path_into() {
  local src=$1
  local dst=$2
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
}

copy_dir_contents_into() {
  local src=$1
  local dst=$2
  mkdir -p "$dst"
  cp -a "$src"/. "$dst"/
}

CODE_FILES=(
  projects/configs/JRDB_OmniTrack.py
  projects/mmdet3d_plugin/models/omnidetr/instance_back_omnidetr.py
  projects/mmdet3d_plugin/models/omnidetr/track_handler_module.py
  projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/association.py
  projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/hybrid_sort.py
  projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/association_geometry.py
  projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/tracker_builder.py
  tools/diag_seam_metric_split.py
  tools/export_jrdb_trackeval_2d.py
  tools/report_b2_mixed_geometry.py
  tools/run_b2_mixed_geometry_matrix.sh
  tools/package_b2_bundle.sh
  jrdb_toolkit/tracking_eval/TrackEval/trackeval/__init__.py
)

TEST_FILES=(
  tests/test_association_geometry.py
  tests/test_diag_metric_split.py
  tests/test_hybrid_sort_geometry_switch.py
  tests/test_mixed_geometry_association.py
  tests/test_track_handler_plumbing.py
  tests/test_tracker_builder.py
  tests/test_active_track_utils.py
  tests/test_seam_duplicate_resolution.py
)

for file in "${CODE_FILES[@]}"; do
  copy_file_into "$file" "$BUNDLE_ROOT/code"
done

for file in "${TEST_FILES[@]}"; do
  copy_file_into "$file" "$BUNDLE_ROOT"
done

copy_dir_contents_into "$REPORT_DIR" "$BUNDLE_ROOT/report"
copy_path_into "$BATCH_ROOT" "$BUNDLE_ROOT/results/batch"
copy_path_into "$DIAG_DIR" "$BUNDLE_ROOT/results/diag"

python - <<'PY' "$REPO_ROOT/$BATCH_ROOT/variants.csv" "$BUNDLE_ROOT"
import csv
import shutil
import sys
from pathlib import Path

variants_csv = Path(sys.argv[1])
bundle_root = Path(sys.argv[2])
repo_root = variants_csv.parents[2]
trackers_root = repo_root / "jrdb_toolkit/tracking_eval/TrackEval/data/trackers/jrdb/jrdb_2d_box_val"

with variants_csv.open("r", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

for row in rows:
    tracker_name = row["tracker_name"]
    src = trackers_root / tracker_name
    dst = bundle_root / "results" / "trackers" / tracker_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
PY

git diff -- \
  projects/configs/JRDB_OmniTrack.py \
  projects/mmdet3d_plugin/models/omnidetr/instance_back_omnidetr.py \
  projects/mmdet3d_plugin/models/omnidetr/track_handler_module.py \
  projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/association.py \
  projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/hybrid_sort.py \
  projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/association_geometry.py \
  projects/mmdet3d_plugin/models/trackers/hybrid_sort_tracker/tracker_builder.py \
  tools/diag_seam_metric_split.py \
  tools/export_jrdb_trackeval_2d.py \
  tools/report_b2_mixed_geometry.py \
  tools/run_b2_mixed_geometry_matrix.sh \
  tools/package_b2_bundle.sh \
  jrdb_toolkit/tracking_eval/TrackEval/trackeval/__init__.py \
  tests/test_association_geometry.py \
  tests/test_diag_metric_split.py \
  tests/test_hybrid_sort_geometry_switch.py \
  tests/test_mixed_geometry_association.py \
  tests/test_track_handler_plumbing.py \
  tests/test_tracker_builder.py \
  > "$BUNDLE_ROOT/patches/b2_code_changes.diff"

{
  echo "# Bundle root: $(basename "$BUNDLE_ROOT")"
  echo "# Generated: $(date -Iseconds)"
  echo "# Paths are relative to bundle root"
  find "$BUNDLE_ROOT" -type f ! -name FILE_MANIFEST.txt -print0 \
    | sort -z \
    | while IFS= read -r -d '' file; do
        rel=${file#"$BUNDLE_ROOT"/}
        size=$(stat -c '%s' "$file")
        sha=$(sha256sum "$file" | awk '{print $1}')
        printf '%s  %s  %s\n' "$sha" "$size" "$rel"
      done
} > "$BUNDLE_ROOT/FILE_MANIFEST.txt"

tar -czf "${BUNDLE_ROOT}.tar.gz" -C "$(dirname "$BUNDLE_ROOT")" "$(basename "$BUNDLE_ROOT")"

echo "Bundle root: $BUNDLE_ROOT"
echo "Archive: ${BUNDLE_ROOT}.tar.gz"
