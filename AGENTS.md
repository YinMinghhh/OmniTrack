# OmniTrack Repo Guide

## Working Environment

- Canonical baseline host and shared-asset root: `/mnt/sdb/ym/OmniTrack`.
- This worktree may live at a different path, but it is expected to expose the same repo-local entry points:
  - `data/JRDB2019`
  - `data/JRDB2019_2d_stitched_anno_pkls`
  - `ckpt/resnet50-19c8e357.pth`
  - `ckpt/jrdb2019_baseline_iter_135900.pth`
  - `work_dirs/jrdb2019_4g_bs16_20260404_001118`
- The real shared files live under `/mnt/sdb/ym/OmniTrack/_shared_assets/{data,ckpt,work_dirs}`.
- Default environment:
  - `source /home/SNN/anaconda3/etc/profile.d/conda.sh`
  - `conda activate /mnt/sdb/ym/envs/OmniTrack-clean`
- Clean reproduction is validated in `/mnt/sdb/ym/envs/OmniTrack-clean` rather than the older `/mnt/sdb/ym/envs/OmniTrack`.
- Target stack from repo docs: Python 3.10, PyTorch 2.1.1, CUDA 11.8.
- The user states the server has sufficient hardware. If the assistant-side shell reports `torch.cuda.is_available()==False`, do not immediately conclude the machine has no GPU; that can be a sandbox visibility issue.

## Shared Asset Contract

- Prefer the repo-local entry points above over hard-coded `_shared_assets` paths in scripts, docs, and commands.
- `data/JRDB2019` is the canonical JRDB2019 dataset entry.
- `data/JRDB2019_2d_stitched_anno_pkls` is the canonical generated annotation entry.
- `ckpt/resnet50-19c8e357.pth` is the canonical ImageNet backbone checkpoint entry.
- `ckpt/jrdb2019_baseline_iter_135900.pth` is the stable baseline checkpoint alias and should be preferred over hard-coding a specific `work_dirs/.../iter_135900.pth` path.
- `work_dirs/jrdb2019_4g_bs16_20260404_001118` is the canonical full baseline training artifact directory.
- When creating new worktrees, preserve these symlinks instead of copying the large assets into each worktree.

## Repro Memo

- Required repo-local assets are `data/JRDB2019`, `data/JRDB2019_2d_stitched_anno_pkls`, `ckpt/resnet50-19c8e357.pth`, and `ckpt/jrdb2019_baseline_iter_135900.pth`.
- For the 4 x RTX 3090 setup, the stable training command is `TS=$(date +%Y%m%d_%H%M%S); bash tools/dist_train.sh projects/configs/JRDB_OmniTrack.py 4 --work-dir=work_dirs/jrdb2019_4g_bs16_$TS --cfg-options num_gpus=4 batch_size=4 data.samples_per_gpu=4 find_unused_parameters=True model.img_backbone.with_cp=False`.
- The validated full run directory is `work_dirs/jrdb2019_4g_bs16_20260404_001118`, and its final baseline checkpoint is exposed through `ckpt/jrdb2019_baseline_iter_135900.pth`.
- On the raw `baseline/jrdb-repro-ready` branch, the default config can load `ckpt/jrdb2019_baseline_iter_135900.pth` directly for baseline evaluation.
- For the strongest paper-aligned line, use `TBD + HybridSORT` on local `val`, and run it through the distributed test path even on one GPU: `TS=$(date +%Y%m%d_%H%M%S); bash tools/dist_test.sh projects/configs/JRDB_OmniTrack.py ckpt/jrdb2019_baseline_iter_135900.pth 1 --out work_dirs/eval_jrdb_val_tbd_dist1_$TS/results_val.pkl --cfg-options model.tracking_mode=tbd model.tbd_backend=hybridsort`.
- That `TBD + HybridSORT` command depends on the later configurable tracking interface from `research/seam-a` / `research/seam-b`; do not expect the raw `5737901` baseline branch to accept `model.tracking_mode` or `model.tbd_backend` overrides.
- Then export and score with `python tools/export_jrdb_trackeval_2d.py --ann-file data/JRDB2019_2d_stitched_anno_pkls/JRDB_infos_val_v1.2.pkl --pred-json work_dirs/eval_jrdb_val_tbd_dist1_$TS/raw_json/results_jrdb2d.json --split-name val --tracker-name OmniTrackTBDDist1 --gt-out-dir jrdb_toolkit/tracking_eval/TrackEval/data/gt/jrdb/jrdb_2d_box_val --trackers-out-dir jrdb_toolkit/tracking_eval/TrackEval/data/trackers/jrdb/jrdb_2d_box_val` and `cd jrdb_toolkit/tracking_eval/TrackEval && PYTHONPATH=. python scripts/run_jrdb.py --USE_PARALLEL False --GT_FOLDER data/gt/jrdb/jrdb_2d_box_val --TRACKERS_FOLDER data/trackers/jrdb/jrdb_2d_box_val --SPLIT_TO_EVAL val --TRACKERS_TO_EVAL OmniTrackTBDDist1 --METRICS HOTA CLEAR Identity OSPA --PRINT_ONLY_COMBINED True --PLOT_CURVES False`.
- The validated local `val` strongest-line metrics are `HOTA 27.582 / MOTA 29.827 / IDF1 33.010 / OSPA 0.86098`.

## Git Repo

- commit 采用 `type(scope): 中文摘要` 的格式
