import os
import json
import argparse
import numpy as np

JOINT_NAMES = [
    'HEAD', 'NECK', 'SPINE_CHEST', 'SPINE_NAVEL', 'PELVIS',
    'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT',
    'SHOULDER_RIGHT', 'ELBOW_RIGHT', 'WRIST_RIGHT',
    'HIP_LEFT', 'KNEE_LEFT', 'ANKLE_LEFT',
    'HIP_RIGHT', 'KNEE_RIGHT', 'ANKLE_RIGHT'
]

# Joint name -> index mapping
J = {name: i for i, name in enumerate(JOINT_NAMES)}


def load_skeletons(json_path):
    """Load syn_data.json, return skeleton array (N, 17, 3) and folder name."""
    with open(json_path, 'r', encoding='utf-8') as f:
        frames = json.load(f)
    n = len(frames)
    skeletons = np.zeros((n, 17, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        sk = frame['skeleton_data']
        for j, name in enumerate(JOINT_NAMES):
            joint = sk[name]
            skeletons[i, j] = [joint['x'], joint['y'], joint['z']]
    folder_name = os.path.basename(os.path.dirname(json_path))
    return skeletons, folder_name, n


def fix_skeleton_flip(skeletons):
    """Detect and fix instantaneous skeleton horizontal flips.
    Compares each frame to predecessor; if flipping x/y axes around centroid
    yields smaller total joint distance, applies the flip.
    """
    fixed = skeletons.copy()
    for i in range(1, fixed.shape[0]):
        curr = fixed[i]
        prev = fixed[i - 1]
        dist_original = np.sum(np.linalg.norm(curr - prev, axis=1))
        centroid = np.mean(curr, axis=0)
        flipped = curr.copy()
        flipped[:, 0] = 2 * centroid[0] - curr[:, 0]  # flip x
        flipped[:, 1] = 2 * centroid[1] - curr[:, 1]  # flip y
        dist_flipped = np.sum(np.linalg.norm(flipped - prev, axis=1))
        if dist_flipped < dist_original:
            fixed[i] = flipped
    return fixed


def compute_trunk_angle(skeletons):
    """Compute trunk angle (PELVIS->SPINE_CHEST vs z-axis) per frame in degrees.
    Returns array of shape (N,).
    """
    pelvis = skeletons[:, J['PELVIS']]       # (N, 3)
    chest = skeletons[:, J['SPINE_CHEST']]    # (N, 3)
    vec = chest - pelvis                       # (N, 3)
    vec_len = np.linalg.norm(vec, axis=1)      # (N,)
    vec_len = np.maximum(vec_len, 1e-8)
    cos_angle = vec[:, 2] / vec_len            # z-component / magnitude
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angle))
    return angles


def compute_pelvis_z(skeletons):
    """Return pelvis z-coordinate per frame. Shape (N,)."""
    return skeletons[:, J['PELVIS'], 2].copy()


def compute_head_pelvis_diff(skeletons):
    """Return head_z - pelvis_z per frame. Shape (N,)."""
    head_z = skeletons[:, J['HEAD'], 2]
    pelvis_z = skeletons[:, J['PELVIS'], 2]
    return head_z - pelvis_z


def compute_feature(skeletons, action):
    """Compute the primary discrimination feature for the given action.
    Returns (feature_array, threshold, comparison_direction).
    comparison_direction: 'above' means feature >= threshold -> action label.
                          'below' means feature <= threshold -> action label.
    """
    if action == 'bend':
        feature = compute_trunk_angle(skeletons)
        threshold = 25.0
        return feature, threshold, 'above'

    elif action == 'sit':
        pelvis_z = compute_pelvis_z(skeletons)
        standing_ref = np.percentile(pelvis_z, 90)
        threshold = 0.75 * standing_ref
        return pelvis_z, threshold, 'below'

    elif action == 'lie':
        head_z = skeletons[:, J['HEAD'], 2].copy()
        standing_ref = np.percentile(head_z, 90)
        threshold = 0.75 * standing_ref
        return head_z, threshold, 'below'

    elif action == 'hunker':
        pelvis_z = compute_pelvis_z(skeletons)
        standing_ref = np.percentile(pelvis_z, 90)
        threshold = 0.65 * standing_ref
        return pelvis_z, threshold, 'below'

    else:
        raise ValueError(f"Unknown action: {action}")


def smooth_feature(feature, window_size):
    """Apply moving average smoothing to feature array.
    Uses edge padding to preserve array length. Window size forced to odd.
    """
    if window_size <= 1:
        return feature.copy()
    if window_size % 2 == 0:
        window_size += 1
    kernel = np.ones(window_size) / window_size
    pad = window_size // 2
    padded = np.pad(feature, pad, mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed


def apply_threshold(feature, threshold, direction, action):
    """Apply threshold to produce per-frame labels.
    Returns list of strings: 'walking' or action name.
    """
    if direction == 'above':
        is_action = feature >= threshold
    else:
        is_action = feature <= threshold
    return [action if a else 'walking' for a in is_action]


def remove_short_segments(labels, min_segment):
    """Merge segments shorter than min_segment frames into the preceding segment."""
    if not labels or min_segment <= 1:
        return labels
    result = labels.copy()
    segments = []
    start = 0
    for i in range(1, len(result)):
        if result[i] != result[start]:
            segments.append((start, i - 1, result[start]))
            start = i
    segments.append((start, len(result) - 1, result[start]))

    if len(segments) <= 1:
        return result

    merged = [segments[0]]
    for seg in segments[1:]:
        seg_len = seg[1] - seg[0] + 1
        if seg_len < min_segment:
            prev = merged[-1]
            merged[-1] = (prev[0], seg[1], prev[2])
        else:
            merged.append(seg)

    out = [''] * len(result)
    for s, e, label in merged:
        for i in range(s, e + 1):
            out[i] = label
    return out


def build_segments(labels):
    """Build run-length encoded segments from label list.
    Returns list of dicts: [{"start": 0, "end": 89, "label": "walking"}, ...]
    """
    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            segments.append({"start": start, "end": i - 1, "label": labels[start]})
            start = i
    segments.append({"start": start, "end": len(labels) - 1, "label": labels[start]})
    return segments


def process_sample(json_path, action, smooth_window, min_segment, do_fix_flip):
    """Process one syn_data.json file and return the label dict."""
    skeletons, folder_name, total_frames = load_skeletons(json_path)

    if do_fix_flip:
        skeletons = fix_skeleton_flip(skeletons)

    feature, threshold, direction = compute_feature(skeletons, action)
    feature_smooth = smooth_feature(feature, smooth_window)
    labels = apply_threshold(feature_smooth, threshold, direction, action)
    labels = remove_short_segments(labels, min_segment)
    segments = build_segments(labels)

    return {
        "source": folder_name,
        "action_type": action,
        "total_frames": total_frames,
        "labels": labels,
        "segments": segments
    }


def save_labels(label_dict, output_path):
    """Save label dict as JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(label_dict, f, ensure_ascii=False, indent=2)


def process_action(data_root, action, smooth_window, min_segment, do_fix_flip, overwrite):
    """Process all samples for one action type."""
    action_dir = os.path.join(data_root, action)
    if not os.path.isdir(action_dir):
        print(f"[SKIP] Directory not found: {action_dir}")
        return

    folders = sorted([
        d for d in os.listdir(action_dir)
        if os.path.isdir(os.path.join(action_dir, d))
    ])

    print(f"\n{'='*60}")
    print(f"Action: {action} ({len(folders)} samples)")
    print(f"{'='*60}")

    for folder in folders:
        json_path = os.path.join(action_dir, folder, 'syn_data.json')
        output_path = os.path.join(action_dir, folder, 'frame_labels.json')

        if not os.path.isfile(json_path):
            print(f"  [SKIP] {folder}: syn_data.json not found")
            continue

        if os.path.isfile(output_path) and not overwrite:
            print(f"  [SKIP] {folder}: frame_labels.json exists (use --overwrite)")
            continue

        try:
            result = process_sample(json_path, action, smooth_window, min_segment, do_fix_flip)
            save_labels(result, output_path)
            walk_n = result['labels'].count('walking')
            action_n = result['labels'].count(action)
            seg_n = len(result['segments'])
            print(f"  [OK]   {folder}: {result['total_frames']} frames "
                  f"(walking={walk_n}, {action}={action_n}, {seg_n} segments)")
        except Exception as e:
            print(f"  [ERR]  {folder}: {e}")


MIXED_ACTIONS = ['bend', 'sit', 'lie', 'hunker']


def main():
    parser = argparse.ArgumentParser(
        description='Frame-level action labeler for mixed-action sequences')
    parser.add_argument('--action', required=True,
                        choices=MIXED_ACTIONS + ['all'],
                        help='Action type to label (or "all" for all 4)')
    parser.add_argument('--data-root', default='subdataset/',
                        help='Root directory of subdataset (default: subdataset/)')
    parser.add_argument('--fix-flip', action='store_true',
                        help='Enable skeleton flip correction')
    parser.add_argument('--smooth-window', type=int, default=11,
                        help='Feature moving average window size (default: 11)')
    parser.add_argument('--min-segment', type=int, default=15,
                        help='Minimum segment length in frames (default: 15)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing frame_labels.json files')
    args = parser.parse_args()

    actions = MIXED_ACTIONS if args.action == 'all' else [args.action]

    for action in actions:
        process_action(
            data_root=args.data_root,
            action=action,
            smooth_window=args.smooth_window,
            min_segment=args.min_segment,
            do_fix_flip=args.fix_flip,
            overwrite=args.overwrite,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
