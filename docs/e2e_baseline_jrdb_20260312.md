# OmniTrack E2E Baseline Lock

Model: OmniTrack E2E
Commit: 94dd465
Checkpoint: work_dirs/jrdb2019_4g_bs2/iter_135900.pth

Local protocol:
- Split: JRDB val7
- Script: scripts/run_eval_e2e.sh
- Split sanity: passed
- Combined: HOTA=..., MOTA=..., IDF1=...

Official benchmark:
- Task: JRDB 2D Tracking
- Input: Stitched Images
- Detection Type: Private Detections
- Tracking Type: Online Tracking
- Runtime: ... s/frame
- Test HOTA: 20.501%
- Submission accepted: yes

Gap to paper:
- Paper HOTA: 21.56%
- Reproduced HOTA: 20.501%
- Absolute gap: -1.059

Decision:
- Freeze this E2E baseline as the project baseline.
- Do not continue debugging E2E for now.
- Start a separate TBD/DA reproduction line from this baseline branchpoint.

```
INFER_SPLIT=train bash scripts/run_eval_e2e.sh
INFER_SPLIT=val bash scripts/run_eval_e2e.sh
INFER_SPLIT=test bash scripts/run_test_submission_e2e.sh
```
