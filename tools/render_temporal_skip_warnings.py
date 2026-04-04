import argparse
import ast
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WARNING_RE = re.compile(
    r"Temporal group skipped \| "
    r"rank=(?P<rank>\d+) "
    r"count=(?P<count>\d+) "
    r"reason=(?P<reason>\S+) "
    r"sample_idx=(?P<sample_idx>\[.*?\]) "
    r"cur_timestamp=(?P<cur_timestamp>\[.*?\]|None) "
    r"prev_timestamp=(?P<prev_timestamp>\[.*?\]|None) "
    r"inter_time=(?P<inter_time>\[.*?\]|None) "
    r"time_mask=(?P<time_mask>\[.*?\]|None) "
    r"gt_groups=(?P<gt_groups>\[.*?\]|None) "
    r"prev_gt_groups=(?P<prev_gt_groups>\[.*?\]|None)"
)


@dataclass
class TemporalSkipEvent:
    line_no: int
    rank: int
    count: int
    reason: str
    sample_idx: List[str]
    cur_timestamp: Optional[list]
    prev_timestamp: Optional[list]
    inter_time: Optional[list]
    time_mask: Optional[list]
    gt_groups: Optional[list]
    prev_gt_groups: Optional[list]
    raw_line: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render JRDB images referenced by temporal-skip warnings."
    )
    parser.add_argument("--log-file", required=True, type=Path, help="Training console log.")
    parser.add_argument(
        "--line-number",
        dest="line_numbers",
        action="append",
        type=int,
        help="Specific warning line number to render. Repeatable.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render every temporal-skip warning in the log.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to work_dirs/temporal_skip_render_<log_stem>.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repo root used to resolve relative JRDB paths.",
    )
    parser.add_argument(
        "--pkl-glob",
        default="data/JRDB2019_2d_stitched_anno_pkls/JRDB_infos_*_v1.2.pkl",
        help="Glob pattern for JRDB info pkls relative to repo root.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=1600,
        help="Preview width used in event contact sheets.",
    )
    return parser.parse_args()


def literal_or_none(value):
    if value == "None":
        return None
    return ast.literal_eval(value)


def parse_events(log_file):
    events = []
    with log_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            match = WARNING_RE.search(line)
            if not match:
                continue
            groups = match.groupdict()
            events.append(
                TemporalSkipEvent(
                    line_no=line_no,
                    rank=int(groups["rank"]),
                    count=int(groups["count"]),
                    reason=groups["reason"],
                    sample_idx=literal_or_none(groups["sample_idx"]),
                    cur_timestamp=literal_or_none(groups["cur_timestamp"]),
                    prev_timestamp=literal_or_none(groups["prev_timestamp"]),
                    inter_time=literal_or_none(groups["inter_time"]),
                    time_mask=literal_or_none(groups["time_mask"]),
                    gt_groups=literal_or_none(groups["gt_groups"]),
                    prev_gt_groups=literal_or_none(groups["prev_gt_groups"]),
                    raw_line=line.rstrip("\n"),
                )
            )
    return events


def load_info_index(repo_root, pkl_glob):
    info_index = {}
    for pkl_path in sorted(repo_root.glob(pkl_glob)):
        with pkl_path.open("rb") as f:
            obj = pickle.load(f)
        infos = obj["infos"] if isinstance(obj, dict) and "infos" in obj else obj
        for info in infos:
            info_index[info["token"]] = info
    return info_index


def resolve_image_path(repo_root, info):
    rel_path = info["cams"]["image_stitched"]["data_path"]
    rel_path = rel_path[2:] if rel_path.startswith("./") else rel_path
    return (repo_root / rel_path).resolve()


def xywh_to_xyxy(boxes):
    boxes = np.asarray(boxes)
    xyxy = np.zeros_like(boxes, dtype=np.float32)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def get_font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def annotate_image(image_path, info, token, event):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    boxes = xywh_to_xyxy(info["gt_boxes"])
    for x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=2)

    font = get_font()
    overlay_lines = [
        f"token: {token}",
        f"reason: {event.reason} | rank: {event.rank} | count: {event.count}",
        f"gt_boxes: {len(boxes)} | timestamp: {info['timestamp']}",
    ]
    if event.inter_time is not None:
        sample_pos = event.sample_idx.index(token)
        overlay_lines.append(
            f"inter_time: {event.inter_time[sample_pos]:.6f} | time_mask: {event.time_mask[sample_pos]}"
        )

    padding = 10
    line_height = 24
    box_height = padding * 2 + line_height * len(overlay_lines)
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0))
    for idx, text in enumerate(overlay_lines):
        draw.text((padding, padding + idx * line_height), text, fill=(255, 255, 255), font=font)
    return image


def write_contact_sheet(images, output_path, tile_width):
    if not images:
        return

    resized = []
    for image in images:
        scale = tile_width / image.width
        tile_height = max(1, int(image.height * scale))
        resized.append(image.resize((tile_width, tile_height)))

    padding = 12
    canvas_width = tile_width + padding * 2
    canvas_height = sum(image.height for image in resized) + padding * (len(resized) + 1)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(24, 24, 24))

    y = padding
    for image in resized:
        canvas.paste(image, (padding, y))
        y += image.height + padding
    canvas.save(output_path)


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def render_event(event, info_index, repo_root, out_dir, tile_width):
    event_dir = out_dir / f"L{event.line_no}_{sanitize_name(event.reason)}_rank{event.rank}_count{event.count}"
    event_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    token_records = []
    for token in event.sample_idx:
        info = info_index.get(token)
        if info is None:
            token_records.append({"token": token, "error": "token not found in pkls"})
            continue

        image_path = resolve_image_path(repo_root, info)
        token_record = {
            "token": token,
            "image_path": str(image_path),
            "timestamp": info["timestamp"],
            "num_gt_boxes": int(len(info["gt_boxes"])),
        }

        if not image_path.exists():
            token_record["error"] = "image file not found"
            token_records.append(token_record)
            continue

        rendered_image = annotate_image(image_path, info, token, event)
        image_out = event_dir / f"{token}.jpg"
        rendered_image.save(image_out, quality=95)
        token_record["rendered_path"] = str(image_out)
        rendered.append(rendered_image)
        token_records.append(token_record)

    summary = asdict(event)
    summary["tokens"] = token_records
    with (event_dir / "event.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_contact_sheet(rendered, event_dir / "contact_sheet.jpg", tile_width)
    return event_dir


def main():
    args = parse_args()
    events = parse_events(args.log_file)
    if not events:
        raise SystemExit(f"No temporal skip warnings found in {args.log_file}")

    selected = []
    if args.all:
        selected = events
    elif args.line_numbers:
        selected = [event for event in events if event.line_no in set(args.line_numbers)]
        missing = sorted(set(args.line_numbers) - {event.line_no for event in selected})
        if missing:
            raise SystemExit(f"Warning lines not found: {missing}")
    else:
        selected = [events[-1]]

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir or (
        repo_root / "work_dirs" / f"temporal_skip_render_{sanitize_name(args.log_file.stem)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    info_index = load_info_index(repo_root, args.pkl_glob)
    if not info_index:
        raise SystemExit("No JRDB info pkls found; check --repo-root or --pkl-glob.")

    index_summary = []
    for event in selected:
        event_dir = render_event(event, info_index, repo_root, out_dir, args.tile_width)
        index_summary.append(
            {
                "line_no": event.line_no,
                "reason": event.reason,
                "rank": event.rank,
                "count": event.count,
                "event_dir": str(event_dir),
            }
        )

    with (out_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(index_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
