import json, os, glob, shutil, argparse

# 配置部分
GT_SOURCE_DIR = "data/JRDB2019/train_dataset_with_activity/labels/labels_2d_stitched"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    return parser.parse_args()


def convert_gt(workspace_gt):
    print("[1/3] Converting GT...")
    out_dir = os.path.join(workspace_gt, "label_02")  # 强制使用 label_02
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # 这里简化了之前 convert_gt_stitched_to_kitti.py 的逻辑
    for json_f in glob.glob(os.path.join(GT_SOURCE_DIR, "*.json")):
        seq = os.path.splitext(os.path.basename(json_f))[0]
        with open(json_f) as f:
            data = json.load(f)
        lines = []
        for frame, objs in data.get('labels', {}).items():
            fid = int(frame.split('.')[0])
            for obj in objs:
                try:
                    tid = int(obj['label_id'].split(':')[-1]) if ':' in obj['label_id'] else int(obj['label_id'])
                except:
                    tid = abs(hash(obj['label_id'])) % 100000
                b = obj['box']
                lines.append(
                    f"{fid} {tid} Pedestrian 0 0 0 {b[0]} {b[1]} {b[0] + b[2]} {b[1] + b[3]} -1 -1 -1 -1000 -1000 -1000 0 1.0\n")

        lines.sort(key=lambda x: int(x.split(' ')[0]))
        with open(os.path.join(out_dir, f"{seq}.txt"), 'w') as f:
            f.writelines(lines)


def generate_seqmap(workspace_gt):
    print("[2/3] Generating Seqmap...")
    gt_dir = os.path.join(workspace_gt, "label_02")
    # 直接放在 GT 根目录，满足 jrdb_2d_box.py 的读取逻辑
    seqmap_path = os.path.join(workspace_gt, "evaluate_tracking.seqmap.train")

    lines = []
    for f in glob.glob(os.path.join(gt_dir, "*.txt")):
        seq = os.path.splitext(os.path.basename(f))[0]
        with open(f) as txt:
            content = txt.readlines()
            frames = [int(l.split(' ')[0]) for l in content] if content else [0]
            # 强制修正 Max Frame，避免 1727 越界错误
            lines.append(f"{seq} empty {min(frames)} {max(frames)}\n")

    lines.sort()
    with open(seqmap_path, 'w') as f: f.writelines(lines)


def fill_missing(workspace_pred, workspace_gt):
    print("[3/3] Filling missing predictions...")
    gt_seqs = {os.path.basename(f) for f in glob.glob(os.path.join(workspace_gt, "label_02/*.txt"))}
    pred_seqs = {os.path.basename(f) for f in glob.glob(os.path.join(workspace_pred, "*.txt"))}

    for seq in (gt_seqs - pred_seqs):
        with open(os.path.join(workspace_pred, seq), 'w') as f: pass
        print(f"  -> Created empty file for missing seq: {seq}")


def main():
    args = parse_args()
    gt_ws = os.path.join(args.workspace, "gt")
    pred_ws = os.path.join(args.workspace, "pred/JRDB-train")

    convert_gt(gt_ws)
    generate_seqmap(gt_ws)
    fill_missing(pred_ws, gt_ws)
    print("[SUCCESS] Evaluation environment ready.")


if __name__ == '__main__': main()