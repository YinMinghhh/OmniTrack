import json, os, argparse, math
from tqdm import tqdm
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Convert OmniTrack JSON to KITTI txt')
    parser.add_argument('--pred_json', required=True, help='Path to results_jrdb2d.json')
    parser.add_argument('--output_dir', required=True, help='Directory to save KITTI txt files')
    parser.add_argument(
        '--box_format',
        default='xywh',
        choices=['xywh', 'xyxy'],
        help='Output bbox format written to columns 6:10.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    print(f"[INFO] Loading {args.pred_json}...")
    print(f"[INFO] Writing boxes as: {args.box_format}")
    with open(args.pred_json, 'r') as f:
        data = json.load(f)

    # 智能解包逻辑
    if isinstance(data, dict) and 'results' in data:
        content = data['results']
    elif isinstance(data, list):
        content = data
    else:
        content = data  # 假设是字典直接包含数据

    # 扁平化数据
    flat_list = []
    if isinstance(content, dict):
        for k, v in content.items():
            for obj in v:
                if 'sample_token' not in obj: obj['sample_token'] = k
                flat_list.append(obj)
    elif isinstance(content, list):
        flat_list = content

    seq_outputs = defaultdict(list)
    for obj in tqdm(flat_list, desc="Converting"):
        token = obj.get('sample_token', '')
        last_us = token.rfind('_')
        if last_us == -1: continue

        seq_name = token[:last_us]
        try:
            frame_idx = int(token[last_us + 1:])
        except:
            continue

        # 过滤无效ID
        tid = obj.get('tracking_id', 'None')
        if tid == 'None' or tid is None: continue
        try:
            tid = int(float(tid))
        except:
            continue

        # 坐标提取
        x1y1, size = obj.get('x1y1', []), obj.get('size', [])
        if len(x1y1) < 2 or len(size) < 2: continue

        score = float(obj.get('detection_score', -1))
        if math.isnan(score): score = -1.0

        left, top = float(x1y1[0]), float(x1y1[1])
        width, height = float(size[0]), float(size[1])
        if args.box_format == 'xywh':
            x3, x4 = width, height
        else:
            x3, x4 = left + width, top + height

        line = (
            f"{frame_idx} {tid} Pedestrian 0 0 0 "
            f"{left:.2f} {top:.2f} {x3:.2f} {x4:.2f} "
            f"-1 -1 -1 -1000 -1000 -1000 0 {score:.4f}\n"
        )
        seq_outputs[seq_name].append(line)

    for seq, lines in seq_outputs.items():
        lines.sort(key=lambda x: int(x.split(' ')[0]))
        with open(os.path.join(args.output_dir, f"{seq}.txt"), 'w') as f:
            f.writelines(lines)
    print(f"[SUCCESS] Converted {len(seq_outputs)} sequences to {args.output_dir}")


if __name__ == '__main__': main()
