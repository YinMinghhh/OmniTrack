# JRDB2019 Evaluation Pipeline Audit (Current Repo State)

This note documents what the current scripts and evaluator actually do, and where metric-impacting risks exist.

## 1. Scope

Files covered:

- `scripts/train_jrdb2019_4g_bs2.sh`
- `scripts/eval_jrdb2019.sh`
- `scripts/run_eval.sh`
- `tools/convert_pred_json2kitti.py`
- `tools/convert_json2kitti_2d.py`
- `tools/prepare_eval_env.py`
- `jrdb_toolkit/tracking_eval/TrackEval/scripts/run_jrdb.py`
- `jrdb_toolkit/tracking_eval/TrackEval/trackeval/datasets/jrdb_2d_box.py`

## 2. End-to-End Contract (What "evaluation pipeline" means)

The evaluation number (HOTA/CLEAR/ID) is only meaningful if all these interfaces are aligned:

1. Model inference output format
2. Prediction conversion format
3. GT + seqmap workspace layout
4. TrackEval parser and metric assumptions

## 3. Canonical I/O Expectations of TrackEval (JRDB2DBox)

Given these runtime args:

- `--GT_FOLDER <gt_root>`
- `--TRACKERS_FOLDER <pred_root>`
- `--TRACKERS_TO_EVAL JRDB-train`
- `--TRACKER_SUB_FOLDER ""`
- `--SPLIT_TO_EVAL train`

TrackEval expects:

- GT file: `<gt_root>/evaluate_tracking.seqmap.train`
- GT labels: `<gt_root>/label_02/<seq>.txt`
- Pred labels: `<pred_root>/JRDB-train/<seq>.txt`

Important parser behavior:

- Sequence list and length are read from seqmap.
- `seq_lengths[seq] = row[3]`, then `num_timesteps = seq_lengths[seq]`.
- Timesteps evaluated are `0 .. num_timesteps-1`.

## 4. What each script currently does

### 4.1 `scripts/train_jrdb2019_4g_bs2.sh`

Role:

- Launch 4-GPU distributed training on `projects/configs/JRDB_OmniTrack.py`.
- Write checkpoints under `work_dirs/jrdb2019_4g_bs2`.

Impact on eval:

- This script only produces checkpoints used by eval scripts.
- It does not affect evaluator logic directly.

### 4.2 `scripts/eval_jrdb2019.sh`

Role:

- Run inference.
- Convert JSON predictions with `tools/convert_json2kitti_2d.py`.
- Call TrackEval.

Observed issues:

- No call to `tools/prepare_eval_env.py` (GT/seqmap prep is assumed pre-existing).
- Does not pass `--TRACKER_SUB_FOLDER ""`, so dataset default `TRACKER_SUB_FOLDER='data'` is used.
- Converter writes to `evaluation_workspace/pred/JRDB-train/<seq>.txt`, but default parser expects `.../JRDB-train/data/<seq>.txt`.

Conclusion:

- As written, this script is structurally fragile and likely inconsistent with current output layout.

### 4.3 `scripts/run_eval.sh`

Role:

- Run inference.
- Convert JSON predictions with `tools/convert_pred_json2kitti.py`.
- Run TrackEval with `--TRACKER_SUB_FOLDER ""`.

Observed issues:

- Step 3 is announced (`Environment Setup`) but `prepare_eval_env.py` is currently commented out.
- This means GT/seqmap/padding may be stale from older runs.

Conclusion:

- Better aligned than `eval_jrdb2019.sh` on tracker folder mapping, but still depends on external stale state unless Step 3 is executed.

## 5. Converter + evaluator semantic contract

TrackEval JRDB2D parser loads columns 6:10 as box coordinates and computes similarity with:

- `_calculate_box_ious(..., box_format='xywh')`

Current converters write:

- `left, top, right, bottom` (x1, y1, x2, y2)

This is a high-risk semantic mismatch:

- Evaluator expects `(x, y, w, h)` but input appears `(x1, y1, x2, y2)`.
- This can severely distort IoU and produce abnormal HOTA behavior.

## 6. Other evaluator changes that affect comparability

In `jrdb_2d_box.py`, several filtering behaviors are force-disabled:

- `is_occluded_or_truncated=False`
- `is_distractor_class=False`
- `is_within_crowd_ignore_region=False`

Effect:

- Metric behavior is no longer equivalent to standard TrackEval preprocessing assumptions.
- Even if your pipeline is internally consistent, scores may not be comparable to paper-reported numbers that use different preprocessing rules.

## 7. Risk register (priority ordered)

### P0 (must resolve before trusting any score)

1. Box format mismatch risk (xywh vs x1y1x2y2).
2. `run_eval.sh` step-3 prep currently not executed (commented).
3. `eval_jrdb2019.sh` tracker subfolder mismatch with produced prediction layout.

### P1 (resolve for reproducibility and fairness)

1. Seqmap end index may be off-by-one depending on expected convention.
2. GT fallback `hash(label_id)` is nondeterministic across runs/processes if triggered.
3. Workspace is not cleaned between runs, so stale files can leak into evaluation.

### P2 (quality and maintainability)

1. Two eval scripts with overlapping logic increase confusion and drift.
2. Duplicate converters make it hard to know which contract is authoritative.

## 8. Minimal "Trust Gate" before using as baseline

Before any model improvement work, confirm:

1. A single canonical eval entrypoint is selected.
2. Step-3 environment prep is deterministic and always executed.
3. Converter box semantics are explicitly documented and match evaluator expectation.
4. Pred/GT sequence sets and frame coverage are explicitly checked each run.
5. Evaluator modifications are documented as "official" vs "local patched".

If any item above is unknown, the resulting HOTA should be treated as non-trustworthy for comparison.
