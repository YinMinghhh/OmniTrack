import glob
import json
import os


DEFAULT_GT_SOURCE_DIR = os.path.join(
    "data",
    "JRDB2019",
    "train_dataset_with_activity",
    "labels",
    "labels_2d_stitched",
)

VALIDATION_SEQUENCES = (
    "clark-center-2019-02-28_1",
    "gates-ai-lab-2019-02-08_0",
    "huang-2-2019-01-25_0",
    "meyer-green-2019-03-16_0",
    "nvidia-aud-2019-04-18_0",
    "tressider-2019-03-16_1",
    "tressider-2019-04-26_2",
)


def get_validation_sequences():
    return list(VALIDATION_SEQUENCES)


def get_all_sequences(gt_source_dir=DEFAULT_GT_SOURCE_DIR):
    pattern = os.path.join(gt_source_dir, "*.json")
    sequences = [
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(pattern)
    ]
    sequences.sort()
    return sequences


def get_split_sequences(split, gt_source_dir=DEFAULT_GT_SOURCE_DIR):
    split = split.lower()
    all_sequences = get_all_sequences(gt_source_dir)
    all_sequence_set = set(all_sequences)
    validation_set = set(VALIDATION_SEQUENCES)
    missing_validation = sorted(validation_set - all_sequence_set)
    if missing_validation:
        raise ValueError(
            "Validation sequences are missing from GT source dir %s: %s"
            % (gt_source_dir, ", ".join(missing_validation))
        )

    if split == "all":
        return all_sequences
    if split == "val":
        return list(VALIDATION_SEQUENCES)
    if split == "train":
        return [seq for seq in all_sequences if seq not in validation_set]
    raise ValueError("Unsupported JRDB eval split: %s" % split)


def _extract_sequence_from_token(token):
    if not isinstance(token, str) or "_" not in token:
        raise ValueError("Invalid JRDB token: %r" % (token,))
    seq_name, frame = token.rsplit("_", 1)
    if not frame.isdigit():
        raise ValueError("Invalid JRDB token frame suffix: %r" % (token,))
    return seq_name


def extract_sequences_from_pred_json(pred_json_path):
    with open(pred_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    content = data.get("results", data) if isinstance(data, dict) else data
    sequences = set()

    if isinstance(content, dict):
        for token in content.keys():
            sequences.add(_extract_sequence_from_token(token))
    elif isinstance(content, list):
        for obj in content:
            token = obj.get("sample_token")
            if token is None:
                raise ValueError("Prediction JSON list item is missing sample_token.")
            sequences.add(_extract_sequence_from_token(token))
    else:
        raise ValueError(
            "Prediction JSON must be a dict with 'results' or a list of annotations."
        )

    return sorted(sequences)


def detect_split_from_pred_json(pred_json_path, gt_source_dir=DEFAULT_GT_SOURCE_DIR):
    pred_sequences = set(extract_sequences_from_pred_json(pred_json_path))
    if not pred_sequences:
        raise ValueError("Prediction JSON does not contain any JRDB sequences.")

    matches = []
    for split in ("train", "val", "all"):
        target_sequences = set(get_split_sequences(split, gt_source_dir))
        if pred_sequences == target_sequences:
            matches.append(split)

    if len(matches) == 1:
        return matches[0]

    diagnostics = []
    for split in ("train", "val", "all"):
        target_sequences = set(get_split_sequences(split, gt_source_dir))
        missing = len(target_sequences - pred_sequences)
        extra = len(pred_sequences - target_sequences)
        diagnostics.append("%s(missing=%d, extra=%d)" % (split, missing, extra))
    raise ValueError(
        "Prediction JSON sequences do not exactly match canonical JRDB splits: %s"
        % ", ".join(diagnostics)
    )
