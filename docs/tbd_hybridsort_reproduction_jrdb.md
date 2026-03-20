# JRDB HybridSORT-TBD Reproduction Guide

这份文档给出当前仓库中 JRDB stitched-image 2D tracking 的 TBD 复现方式。

默认前提：

- E2E 已冻结为当前项目 baseline。
- 这版 TBD 的目标是把 `HybridSORT` 路径做成可切换、可评估、可提交。
- 训练仍复用 OmniTrack 主干训练线，不额外引入新的 loss、数据管线或专门的 TBD 训练算法。

## 1. 先说结论

当前仓库里的模式开关已经显式化到 config：

- `model.head.instance_bank.tracking_mode = "e2e" | "tbd"`
- `model.head.instance_bank.tbd_backend = "hybridsort"`

默认值仍然是：

```python
tracking_mode = "e2e"
tbd_backend = "hybridsort"
```

也就是说：

- 不传任何新参数时，行为仍然是当前 E2E baseline。
- 切到 TBD 时，当前唯一正式支持的 backend 是 `HybridSORT`。
- 如果把 backend 设成 `ocsort` 或 `bytetrack`，代码会直接抛 `NotImplementedError`，避免“看起来能选、实际没接通”的误导。

## 2. 这版 TBD 到底是什么意思

这版 TBD 采用“共享训练、TBD 体现在推理后处理”的定义。

具体来说：

1. 模型训练仍然使用当前 OmniTrack 主干与 decoder 学习。
2. 推理时通过 `tracking_mode=tbd` 把 `post_process()` 后的实例管理切到 `TrackHandler -> HybridSORT`。
3. 评估、JSON 导出、TrackEval 转换、官方 test submission 打包继续沿用当前 canonical script 链路。

因此：

- 你可以直接拿现有训练 checkpoint 跑 TBD 评估。
- 如果想让工作目录命名更清楚，也可以在训练脚本里显式传 `TRACKING_MODE=tbd`，但推荐理解成“共享训练线”，而不是“单独的 TBD 训练算法”。

## 3. 一键切换命令

### 3.1 保持默认 E2E

```bash
INFER_SPLIT=val bash scripts/run_eval_e2e.sh
```

### 3.2 直接跑 TBD-HybridSORT

```bash
INFER_SPLIT=val bash scripts/run_eval_tbd.sh
```

这个 wrapper 等价于：

```bash
TRACKING_MODE=tbd \
TBD_BACKEND=hybridsort \
INFER_SPLIT=val \
bash scripts/run_eval_e2e.sh
```

TBD wrapper 默认把结果写到独立目录，避免覆盖 E2E baseline：

- `work_dirs/jrdb2019_4g_bs2_tbd_hybridsort/results.pkl`
- `results/submission_tbd_hybridsort/results_jrdb2d.json`
- `results/eval/jrdb_tbd_hybridsort/`

## 4. 训练、评估、提交的完整流程

### 4.1 训练

推荐理解：

- 训练仍然是共享训练线。
- 默认直接用现有训练脚本即可。

标准训练命令：

```bash
bash scripts/train_jrdb2019_4g_bs2.sh
```

如果你只是想在日志或目录命名上保留“这是 TBD 复现线”的信息，可以显式传入模式参数和单独 work dir：

```bash
TRACKING_MODE=tbd \
TBD_BACKEND=hybridsort \
WORK_DIR=work_dirs/jrdb2019_4g_bs2_tbd_hybridsort \
bash scripts/train_jrdb2019_4g_bs2.sh
```

但请注意：

- 这不会把训练算法切成另一套逻辑。
- 它的主要作用是让训练配置和后续 TBD 评估命名保持一致。

### 4.2 在 `val` 上评估 TBD

```bash
CHECKPOINT=work_dirs/jrdb2019_4g_bs2/iter_135900.pth \
INFER_SPLIT=val \
bash scripts/run_eval_tbd.sh
```

### 4.3 在 `train` 上评估 TBD

```bash
CHECKPOINT=work_dirs/jrdb2019_4g_bs2/iter_135900.pth \
INFER_SPLIT=train \
bash scripts/run_eval_tbd.sh
```

### 4.4 生成官方 `test` 提交包

```bash
CHECKPOINT=work_dirs/jrdb2019_4g_bs2/iter_135900.pth \
INFER_SPLIT=test \
bash scripts/run_test_submission_tbd.sh
```

默认输出目录：

- `results/test_submission_tbd_hybridsort/raw_json/results_jrdb2d.json`
- `results/test_submission_tbd_hybridsort/CIWT/data/`
- `results/test_submission_tbd_hybridsort/jrdb_2dt_submission.zip`

## 5. 如果你想用单脚本而不是 wrapper

所有 canonical script 都支持以下环境变量：

- `TRACKING_MODE`
- `TBD_BACKEND`
- `EXTRA_CFG_OPTIONS`

例如：

```bash
TRACKING_MODE=tbd \
TBD_BACKEND=hybridsort \
EXTRA_CFG_OPTIONS="model.head.instance_bank.tbd_tracker_cfg.track_thresh=0.55" \
INFER_SPLIT=val \
bash scripts/run_eval_e2e.sh
```

同样也可以用于 test submission：

```bash
TRACKING_MODE=tbd \
TBD_BACKEND=hybridsort \
INFER_SPLIT=test \
bash scripts/run_test_submission_e2e.sh
```

## 6. 当前这版 TBD 的配置边界

`InstanceBackOMNIDETR` 当前支持以下显式配置：

```python
instance_bank = dict(
    tracking_mode="e2e",
    tbd_backend="hybridsort",
    e2e_handler_cfg=dict(
        nms_thresh=0.05,
        det_thresh=0.10,
        track_thresh=0.39,
        init_thresh=0.315,
        max_time_lost=10,
    ),
    tbd_handler_cfg=dict(
        det_thresh=0.10,
        nms_iou=0.35,
        reset_time_gap=100,
    ),
    tbd_tracker_cfg=dict(),
)
```

说明：

- `e2e_handler_cfg` 只是把当前 E2E baseline 阈值显式化。
- `tbd_handler_cfg` 控制 TrackHandler 自己的检测过滤、NMS 和跨序列 reset。
- `tbd_tracker_cfg` 透传给 HybridSORT 的参数模板，代码内部会先读 `make_parser().parse_args([])` 的默认值，再用这里的字段覆盖，不再依赖全局命令行参数。

## 7. 常见误区

### 7.1 `temp_query=True` 不是 E2E/TBD 开关

不是。

它只表示：

- 是否把上一帧轨迹回灌成 temp queries。

### 7.2 `is_track=True` 不是 E2E/TBD 开关

不是。

它只表示：

- `post_process()` 是否输出 tracking 结果。

### 7.3 这版 TBD 不是“完整三后端切换”

不是。

当前正式支持的是：

- `HybridSORT`

当前不会继续伪装成可用的是：

- `OCSort`
- `ByteTrack`

如果选这两个 backend，代码会直接报错。

### 7.4 评估入口应该用哪个

如果目标是可信的 JRDB baseline / reproduction 数字，继续使用：

- `scripts/run_eval_e2e.sh`
- `scripts/run_eval_tbd.sh`
- `scripts/run_test_submission_e2e.sh`
- `scripts/run_test_submission_tbd.sh`

不要绕开这条脚本链自己拼评估流程，否则很容易再次踩到 JSON / TrackEval / split 对齐的问题。

## 8. 推荐最小复现路径

如果你现在已经有一个可用 checkpoint，最短路径就是：

```bash
CHECKPOINT=work_dirs/jrdb2019_4g_bs2/iter_135900.pth \
INFER_SPLIT=val \
bash scripts/run_eval_tbd.sh
```

如果要生成官方 test 提交包：

```bash
CHECKPOINT=work_dirs/jrdb2019_4g_bs2/iter_135900.pth \
INFER_SPLIT=test \
bash scripts/run_test_submission_tbd.sh
```

这样就能在不改源码的前提下，把同一套 OmniTrack checkpoint 切到 TBD-HybridSORT 路径，并沿用当前仓库的 canonical eval / submission chain。
