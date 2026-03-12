# OmniTrack None-ID 链路排查说明（git branch: ）

这篇文档记录本次 “`tracking_id=None` 大量出现” 的排查过程、根因、最小修复，以及修复后还剩下的独立问题。

目标读者是第一次接触本项目的人，所以会先解释这个项目里“预测结果是怎么一路流到评估指标里的”，再解释这次为什么会出现大量 `None-ID`。

## 1. 这次在排查什么

现象来自评估日志：

- Prediction JSON 中 `tracking_id=None` 数量很大
- 转换成 TrackEval 文本后，只有少量序列真正有预测
- 评估指标很低，尤其是 HOTA / CLEAR / IDF1

本次任务的边界很明确：

- 先定位 `None-ID` 的来源
- 画清楚 ID 生成链路
- 在关键环节加最小诊断统计
- 只做最小修复，不做重构
- 不改训练脚本
- 不改评估指标定义

## 2. 先看整体评估链路

当前标准入口是 [scripts/run_eval.sh](../scripts/run_eval.sh)。

它做的事情可以概括成 6 步：

1. 模型推理，生成 `PKL` 和 `results_jrdb2d.json`
2. 检查 JSON 结构是否合理
3. 把 JSON 转成 TrackEval 需要的 `txt`
4. 准备 GT、seqmap、缺失序列的空预测文件
5. 检查评估工作区结构是否合理
6. 调用 TrackEval，计算 HOTA / CLEAR / IDF1

如果只关心这次 `None-ID` 问题，可以把链路理解成下面这条主线：

```text
模型输出
  -> QueryHandler 生成轨迹/候选检测
  -> Head 整理为 boxes_2d + instance_ids
  -> Dataset 写 results_jrdb2d.json
  -> convert_pred_json2kitti.py 过滤并转成 txt
  -> TrackEval 读 txt 计算指标
```

## 3. 排查范围

这次排查主要涉及 6 个文件：

- [projects/mmdet3d_plugin/models/omnidetr/query_handler_module.py](../projects/mmdet3d_plugin/models/omnidetr/query_handler_module.py)
  - 负责把当前帧结果组织成 “在线轨迹” 和 “检测候选”
- [projects/mmdet3d_plugin/models/omnidetr/omnidetr_head.py](../projects/mmdet3d_plugin/models/omnidetr/omnidetr_head.py)
  - 负责把 QueryHandler 的输出整理成最终 result dict
- [projects/mmdet3d_plugin/datasets/JRDB_2d_det_track_dataset.py](../projects/mmdet3d_plugin/datasets/JRDB_2d_det_track_dataset.py)
  - 负责把 result dict 写成 `results_jrdb2d.json`
- [tools/convert_pred_json2kitti.py](../tools/convert_pred_json2kitti.py)
  - 负责把 JSON 转成 TrackEval 读取的 `txt`
- [tools/eval_sanity_check.py](../tools/eval_sanity_check.py)
  - 负责打印结构检查和按序列统计
- [tools/prepare_eval_env.py](../tools/prepare_eval_env.py)
  - 负责准备 GT、seqmap 和缺失预测文件

## 4. ID 生成链路到底是什么

本次最重要的事情，就是把下面这条链真正讲清楚：

```text
query_handler
  -> instance_ids
  -> tracking_id
  -> convert filter
```

### 4.1 `query_handler`：先分成“正式轨迹”和“候选检测”

入口在 [query_handler_module.py](../projects/mmdet3d_plugin/models/omnidetr/query_handler_module.py)。

它最终返回两个集合：

- `output_stracks`
  - 已经被激活、处于跟踪状态的轨迹
  - 这些对象应该带有 `track_id`
- `detections`
  - 当前帧的检测候选
  - 它们不一定已经被激活，因此不一定有 `track_id`

关键点是：

- 第一帧里，新轨迹会调用 `activate()`，从而拿到 `track_id`
- 后续帧里，会先生成一批 `detections`
- 只有满足初始化阈值的检测候选，才会调用 `activate()` 变成正式 track

所以从语义上说：

- `output_stracks` 是 “可追踪对象”
- `detections` 是 “候选框”，并不保证有 ID

这一步本身没有错。问题出在后面有没有把这两类东西混在一起。

### 4.2 `instance_ids`：只从正式轨迹里取 ID

这一步在 [omnidetr_head.py](../projects/mmdet3d_plugin/models/omnidetr/omnidetr_head.py)。

代码做了两件不同的事：

1. 遍历 `online_targets`，把轨迹的 `track_id` 收集到 `instance_ids`
2. 把 `dets` 单独存到 `det_boxes_2d` / `det_scores_2d`

也就是说，结果字典里同时会有：

- `boxes_2d` + `instance_ids`
  - 这是 tracking 主输出
- `det_boxes_2d`
  - 这是额外的检测候选

这次排查加的一个关键统计，就是这里的：

- `query_handler_tracks`
- `query_handler_dets`
- `query_handler_ids_none`

它们的作用是确认：

- 正式轨迹里是不是已经出现 `None`
- 还是只是检测候选很多

本次日志证明了一个很重要的事实：

- `instance_ids_none = 0`

这说明真正的在线轨迹并没有在这里丢 ID。

### 4.3 `tracking_id`：`instance_ids` 被写到 `Box2D.token`，最后变成 JSON 字段

这一步在 [JRDB_2d_det_track_dataset.py](../projects/mmdet3d_plugin/datasets/JRDB_2d_det_track_dataset.py) 里完成。

先看 `output_to_jrdb_box()`：

- 对 `boxes_2d` 这一路，会把 `instance_ids[i]` 写到 `box.token`
- 对 `det_boxes_2d` 这一路，只创建 box，不会给它们写 token

这非常关键，因为它意味着：

- 正式轨迹的 box 有 `token`
- 检测候选的 box 默认没有 `token`

接下来 `_format_bbox()` 会把每个 box 写成 JSON 对象，其中：

- `box.token` 会被写成 `tracking_id`

问题就出在这里。

在修复前，代码会把所有 box 都写入 JSON，包括：

- 有 token 的 tracking box
- 没有 token 的 detection candidate box

于是后一类 box 被写出去时，就会形成：

```text
tracking_id = None
```

这就是本次 `None-ID` 的真正来源。

### 4.4 `convert filter`：转换脚本最后再做一层兜底过滤

入口在 [tools/convert_pred_json2kitti.py](../tools/convert_pred_json2kitti.py)。

它会读取 JSON 里的每个对象，然后：

- 如果 `tracking_id` 是 `None` 或 `"None"`，直接丢弃
- 如果 `tracking_id` 不能转成整数，也丢弃
- 只有合法 ID 才会写进最终 `txt`

所以 convert 的角色是：

- 最后一层保险
- 不是根因修复点

如果前面的 JSON 已经混入很多无 ID 对象，convert 虽然会把它们丢掉，但：

- JSON 会被污染
- 统计会失真
- 问题根因仍然存在

## 5. 这次到底是怎么定位到根因的

这次没有先大改，而是先在每个关键环节加了最小统计。

### 5.1 在 Head 里看 QueryHandler 到底返回了什么

新增的统计项：

- `query_handler_tracks`
- `query_handler_dets`
- `query_handler_ids_none`

它回答的问题是：

- 正式轨迹有多少
- 候选检测有多少
- 正式轨迹里有没有 `None` ID

实际结果是：

- `query_handler_ids_none=0`
- 但 `query_handler_dets` 很大

这已经说明 `None` 不是从正式轨迹里来的。

### 5.2 在 Dataset 格式化阶段看 `tracking_id` 是在哪里变成 `None`

新增的统计项：

- `instance_ids_total`
- `instance_ids_none`
- `tracking_total`
- `tracking_none`
- `written`

实际结果非常关键：

- `instance_ids_none=0`
- 但 `tracking_none` 很高
- 同时 `tracking_total` 约等于 `qh_tracks + qh_dets`

这说明：

- 上游的 `instance_ids` 没问题
- 但是写 JSON 时，把本来不该算 tracking 输出的 `det_boxes_2d` 也混了进去

### 5.3 在 JSON Sanity Check 里确认最终结果

新增的统计项：

- `JSON frames`
- `objects`
- `none_id`
- 每个序列的 `none_ratio`

修复后结果变成：

- `none_id=0`
- 每个序列的 `none_ratio=0.00%`

这证明 `None-ID` 已经在 JSON 层面被消掉了。

### 5.4 在 Converter 里确认还有没有漏网之鱼

新增的统计项：

- `drop_none_tid`
- `drop_bad_tid`
- `drop_bad_box`
- `written`

修复后结果是：

- `drop_none_tid=0`
- `drop_bad_tid=0`

这说明 convert 这一层已经不需要再替上游“收尸”。

## 6. 根因结论

这次 `None-ID` 的根因可以用一句话概括：

> 无 ID 的检测候选被错误地写进了 tracking JSON，于是 `box.token=None` 最终变成了 `tracking_id=None`。

更精确一点说：

1. `query_handler` 会同时返回正式轨迹和检测候选
2. 正式轨迹有 `track_id`
3. 检测候选不保证有 `track_id`
4. `output_to_jrdb_box()` 只给正式轨迹写 `box.token`
5. 旧逻辑却把两类 box 都写进了 tracking JSON
6. 没有 token 的 box 被写出时，`tracking_id` 就成了 `None`

本次日志已经证明：

- 不是 track 本身丢了 ID
- 而是 det candidate 被混入了 tracking 输出

## 7. 本次做了哪些最小改动

本次只改了必要文件，没有动训练脚本，也没有改评估定义。

### 7.1 [omnidetr_head.py](../projects/mmdet3d_plugin/models/omnidetr/omnidetr_head.py)

改动内容：

- 增加 QueryHandler 级别的最小统计
- 把 `query_handler_tracks` / `query_handler_dets` / `query_handler_ids_none` 放进结果字典

作用：

- 定位 `None` 是否在模型输出阶段产生

### 7.2 [JRDB_2d_det_track_dataset.py](../projects/mmdet3d_plugin/datasets/JRDB_2d_det_track_dataset.py)

改动内容：

- 增加按序列统计
- 在 tracking 模式下，如果 `box.token` 为 `None`，则不写入 JSON

这是本次真正的最小修复点。

作用：

- 让 `results_jrdb2d.json` 只包含真正可追踪的对象

### 7.3 [convert_pred_json2kitti.py](../tools/convert_pred_json2kitti.py)

改动内容：

- 增加按序列统计
- 统计 `drop_none_tid`、`drop_bad_tid`、`written`

作用：

- 明确 convert 到底丢掉了哪些对象

### 7.4 [eval_sanity_check.py](../tools/eval_sanity_check.py)

改动内容：

- 增加 Prediction JSON 的按序列 `none_id` 统计
- 增加 workspace 结构检查输出

作用：

- 让修复结果能直接在日志里看出来

## 8. 修复前后可以怎么理解

### 8.1 修复前的状态

旧日志 [logs/MobaXterm_20260305_135406.txt](../logs/MobaXterm_20260305_135406.txt) 显示：

- JSON: `frames=6203, objects=342261, none_id=237371`
- Converted sequences: `7`
- workspace: `seqmap=27, gt=27, pred=27`
- empty prediction files: `20`
- HOTA COMBINED: `10.964`
- CLEAR MOTA COMBINED: `7.0743`
- IDF1 COMBINED: `7.5342`

从这个状态单看现象，会很容易误以为：

- 模型没有产生 ID

但这次定位证明并不是这样。

### 8.2 修复后的 `val` 结果

修复后日志显示：

- `instance_ids_none=0`
- `JSON frames=6203, objects=94508, none_id=0`
- `Converted 7 sequences`
- 每个序列 `drop_none_tid=0`

这说明：

- `None-ID` 已经从 JSON 和 convert 链路中消失
- JSON 对象数明显下降，是因为无 ID 的候选框不再被写出

### 8.3 修复后的 `train probe` 结果

后面又跑了一次 train probe，日志显示：

- `JSON frames=21744, objects=338061, none_id=0`
- 每个序列 `instance_ids_none=0`
- 每个序列 `none_ratio=0.00%`

这进一步说明：

- 不管跑 `val` 那 7 个序列，还是跑 `train` 那 20 个序列
- `None-ID` 这条链都已经修住了

## 9. 这次修复没有解决什么

这很重要。

`None-ID` 修好之后，评估结果仍然不一定会立刻变好，因为还有一个独立问题：

- 数据切分和评估工作区没有对齐

### 9.1 当前 `pkl` 切分实际上是 `20(train) + 7(val)`

在 [tools/JRDB2019_2d_stitched_converter.py](../tools/JRDB2019_2d_stitched_converter.py) 里：

- `train` 会主动排除那 7 个 validation 序列
- `val` 只保留那 7 个 validation 序列

所以：

- `JRDB_infos_train_v1.2.pkl` 对应 20 个序列
- `JRDB_infos_val_v1.2.pkl` 对应 7 个序列

### 9.2 但评估工作区目前按原始 27 个序列全量构造

在 [tools/prepare_eval_env.py](../tools/prepare_eval_env.py) 里：

- GT 来源直接写死为 `train_dataset_with_activity/labels/labels_2d_stitched/*.json`
- 这会把原始 train 标签目录下的所有序列都转成 GT 和 seqmap

所以现在会出现：

- 跑 `val` 时，只有 7 个序列有预测，剩下 20 个被补成空文件
- 跑 `train` 时，只有 20 个序列有预测，另外 7 个仍然不在当前预测集合里

这就是为什么：

- `None-ID` 修好了
- 但 `Converted 序列数 / 空预测序列数 / HOTA/CLEAR/IDF1` 仍然可能被 split mismatch 主导

换句话说：

- `None-ID` 是本次已经解决的一个问题
- `20/7/27` 的 split 对齐，是另外一个问题

## 10. 用一句话总结这次工作

这次工作的本质不是“让模型学会产生 ID”，而是：

> 证明模型生成的正式轨迹 ID 本来就在，真正的问题是无 ID 的检测候选被误写进了 tracking JSON；修复方式是在 JSON 输出边界把这类对象拦掉，并用按序列统计把链路跑通、证据补齐。

## 12. 后续阅读需牢记这三件事

1. `query_handler` 返回的是两类对象，不要默认它们都应该进入 tracking 评估
2. `instance_ids_none=0` 说明 track ID 本身是正常的
3. `none_id=0` 之后如果指标还是怪，优先去看 split / seqmap / workspace 对齐问题，而不是继续怀疑 ID 生成链路
