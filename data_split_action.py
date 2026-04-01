"""
行为识别数据预处理脚本（帧级标签版）

从 subdataset/ 读取 syn_data.json + frame_labels.json，
按 segment 边界切出可变长纯动作窗口（40-75帧），
生成带逐帧标签的 .npz 训练样本。

用法:
  cd classification_task/
  python data_split_action.py --data-root subdataset/ --outdir action_dataset/
"""
import os
import json
import argparse
import random
import glob
import numpy as np
from collections import Counter
from scipy.signal import savgol_filter

ACTION_CLASSES = {
    "walking": 0, "sit": 1, "bend": 2, "jogging": 3, "boxing": 4,
    "jumpjack": 5, "lie": 6, "hunker": 7, "sweep": 8, "goosestep": 9,
}
ACTION_NAMES = list(ACTION_CLASSES.keys())

# 混合动作：这些目录下有 frame_labels.json
MIXED_ACTIONS = {"bend", "sit", "lie", "hunker"}

JOINT_ORDER = [
    "HEAD", "NECK", "SPINE_CHEST", "SPINE_NAVEL", "PELVIS",
    "SHOULDER_LEFT", "ELBOW_LEFT", "WRIST_LEFT",
    "SHOULDER_RIGHT", "ELBOW_RIGHT", "WRIST_RIGHT",
    "HIP_LEFT", "KNEE_LEFT", "ANKLE_LEFT",
    "HIP_RIGHT", "KNEE_RIGHT", "ANKLE_RIGHT",
]


def extract_pointcloud(frame):
    """从一帧中提取点云 (N, 4): x, y, z, v"""
    pts = frame.get("pointcloud_data", {}).get("points", [])
    if not pts:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array([[p["x"], p["y"], p["z"], p["v"]] for p in pts], dtype=np.float32)


def extract_skeleton(frame):
    """从一帧中提取骨架 (17, 3)"""
    joints = frame.get("skeleton_data", {})
    if not joints:
        return None
    skel = np.zeros((17, 3), dtype=np.float32)
    for i, name in enumerate(JOINT_ORDER):
        if name in joints:
            j = joints[name]
            skel[i] = [j["x"], j["y"], j["z"]]
    return skel


def smooth_skeleton_sequence(skeletons, window_length=5, polyorder=2):
    """Savitzky-Golay 滤波平滑骨架序列"""
    if skeletons.shape[0] < window_length:
        return skeletons
    smoothed = np.copy(skeletons)
    for j in range(17):
        for c in range(3):
            smoothed[:, j, c] = savgol_filter(skeletons[:, j, c], window_length, polyorder)
    return smoothed


def load_sequence(json_path, max_velocity=5.0):
    """
    读取 syn_data.json，返回点云列表和骨架数组。

    Returns:
        all_pcs: list of (N_i, 4) arrays，已做速度归一化
        all_skels: (F, 17, 3) array，已做 Savitzky-Golay 平滑
        F: 总帧数
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        frames = json.load(f)

    all_pcs = []
    all_skels = []
    for frame in frames:
        pc = extract_pointcloud(frame)
        skel = extract_skeleton(frame)
        if skel is None:
            return None, None, 0
        if pc.shape[0] > 0:
            pc[:, 3] = np.clip(pc[:, 3] / max_velocity, -1.0, 1.0)
        all_pcs.append(pc)
        all_skels.append(skel)

    all_skels = np.stack(all_skels, axis=0)  # (F, 17, 3)
    all_skels = smooth_skeleton_sequence(all_skels)
    return all_pcs, all_skels, len(frames)


def normalize_window(pcs, skels):
    """
    对一个窗口做 XY 归一化：从末帧向前找参考帧（点数>10），减去 XY 质心。
    """
    pcs = [pc.copy() for pc in pcs]
    skels = skels.copy()

    origin = None
    for t in range(len(pcs) - 1, -1, -1):
        if pcs[t].shape[0] > 10:
            origin = pcs[t][:, :3].mean(axis=0)
            break

    if origin is None:
        return pcs, skels

    origin_xy = origin[:2]
    skels[:, :, :2] -= origin_xy
    for pc in pcs:
        if pc.shape[0] > 0:
            pc[:, :2] -= origin_xy

    return pcs, skels


def slice_mixed_sequence(all_pcs, all_skels, segments, action_name, min_len=15, max_len=75, slide_step=25):
    """
    对混合动作序列按 segment 边界切出纯动作窗口。
    """
    results = []
    for seg in segments:
        start, end, label = seg["start"], seg["end"], seg["label"]
        seg_len = end - start + 1

        if seg_len < min_len:
            continue

        label_idx = ACTION_CLASSES.get(label, -1)
        if label_idx < 0:
            continue

        if seg_len <= max_len:
            pcs_win = all_pcs[start:end + 1]
            skels_win = all_skels[start:end + 1]
            frame_labels = np.full(seg_len, label_idx, dtype=np.int64)
            pcs_norm, skels_norm = normalize_window(pcs_win, skels_win)
            results.append((pcs_norm, skels_norm, frame_labels, label_idx))
        else:
            for offset in range(0, seg_len - max_len + 1, slide_step):
                w_start = start + offset
                w_end = w_start + max_len
                pcs_win = all_pcs[w_start:w_end]
                skels_win = all_skels[w_start:w_end]
                frame_labels = np.full(max_len, label_idx, dtype=np.int64)
                pcs_norm, skels_norm = normalize_window(pcs_win, skels_win)
                results.append((pcs_norm, skels_norm, frame_labels, label_idx))

            remainder = seg_len - ((seg_len - max_len) // slide_step * slide_step + max_len)
            if remainder >= min_len:
                w_start = end + 1 - remainder
                pcs_win = all_pcs[w_start:end + 1]
                skels_win = all_skels[w_start:end + 1]
                frame_labels = np.full(remainder, label_idx, dtype=np.int64)
                pcs_norm, skels_norm = normalize_window(pcs_win, skels_win)
                results.append((pcs_norm, skels_norm, frame_labels, label_idx))

    return results


def slice_pure_sequence(all_pcs, all_skels, action_label, trim=15, min_len=40, max_len=75):
    """
    对纯动作序列做滑窗切片，窗口大小随机 [min_len, max_len]。
    """
    total = len(all_pcs)
    if total <= 2 * trim:
        return []

    start_idx = trim
    end_idx = total - trim

    usable = end_idx - start_idx
    if usable < min_len:
        return []

    window_size = random.randint(min_len, max_len)
    window_size = min(window_size, usable)
    step = max(window_size // 2, 1)

    results = []
    for offset in range(0, usable - window_size + 1, step):
        w_start = start_idx + offset
        w_end = w_start + window_size
        pcs_win = all_pcs[w_start:w_end]
        skels_win = all_skels[w_start:w_end]
        frame_labels = np.full(window_size, action_label, dtype=np.int64)
        pcs_norm, skels_norm = normalize_window(pcs_win, skels_win)
        results.append((pcs_norm, skels_norm, frame_labels, action_label))

    return results


def save_sample(pcs, skels, frame_labels, majority_label, out_path):
    """保存一个样本为 .npz 文件"""
    pcs_obj = np.array(pcs, dtype=object)
    np.savez_compressed(
        out_path,
        pointcloud=pcs_obj,
        skeleton=skels.astype(np.float32),
        frame_labels=frame_labels.astype(np.int64),
        majority_label=np.int64(majority_label),
    )


def main():
    parser = argparse.ArgumentParser(description="行为识别数据预处理（帧级标签版）")
    parser.add_argument("--data-root", default="subdataset/", help="subdataset/ 根目录")
    parser.add_argument("--outdir", default="action_dataset/", help="输出目录")
    parser.add_argument("--min-len", type=int, default=40, help="最短窗口帧数")
    parser.add_argument("--max-len", type=int, default=75, help="最长窗口帧数")
    parser.add_argument("--slide-step", type=int, default=25, help="长段滑窗步长")
    parser.add_argument("--trim", type=int, default=15, help="纯动作序列裁剪首尾帧数")
    parser.add_argument("--max-velocity", type=float, default=5.0, help="速度归一化阈值")
    parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        help="train/val/test 比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    for s in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.outdir, s), exist_ok=True)

    all_sequences = []
    for action_name in ACTION_NAMES:
        action_dir = os.path.join(args.data_root, action_name)
        if not os.path.isdir(action_dir):
            continue
        folders = sorted([
            d for d in os.listdir(action_dir)
            if os.path.isdir(os.path.join(action_dir, d))
        ])
        is_mixed = action_name in MIXED_ACTIONS
        for folder in folders:
            folder_path = os.path.join(action_dir, folder)
            json_path = os.path.join(folder_path, "syn_data.json")
            if os.path.isfile(json_path):
                all_sequences.append((folder_path, action_name, is_mixed, folder))

    print(f"共发现 {len(all_sequences)} 个序列")

    # ---- 分层划分：按动作类别分组后各自 80/10/10 ----
    from collections import defaultdict
    action_groups = defaultdict(list)
    for seq in all_sequences:
        action_groups[seq[1]].append(seq)  # seq[1] = action_name

    splits = {}
    for action_name, seqs in action_groups.items():
        random.shuffle(seqs)
        n_act = len(seqs)
        n_train = int(n_act * args.split[0])
        n_val = int(n_act * args.split[1])
        for seq in seqs[:n_train]:
            splits[seq[3]] = "train"
        for seq in seqs[n_train:n_train + n_val]:
            splits[seq[3]] = "val"
        for seq in seqs[n_train + n_val:]:
            splits[seq[3]] = "test"
        print(f"  {action_name:12s}: {n_act} 序列 → train={n_train}, val={n_val}, test={n_act - n_train - n_val}")

    total_samples = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    class_counts = Counter()

    for folder_path, action_name, is_mixed, folder in all_sequences:
        split = splits[folder]
        json_path = os.path.join(folder_path, "syn_data.json")

        all_pcs, all_skels, total_frames = load_sequence(json_path, args.max_velocity)
        if all_pcs is None:
            print(f"  [SKIP] {folder}: 骨架数据缺失")
            continue

        if is_mixed:
            label_path = os.path.join(folder_path, "frame_labels.json")
            if not os.path.isfile(label_path):
                print(f"  [SKIP] {folder}: frame_labels.json 不存在")
                continue
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            segments = label_data["segments"]
            samples = slice_mixed_sequence(
                all_pcs, all_skels, segments, action_name,
                min_len=args.min_len, max_len=args.max_len, slide_step=args.slide_step
            )
        else:
            action_label = ACTION_CLASSES[action_name]
            samples = slice_pure_sequence(
                all_pcs, all_skels, action_label,
                trim=args.trim, min_len=args.min_len, max_len=args.max_len
            )

        for i, (pcs, skels, frame_labels, majority_label) in enumerate(samples):
            fname = f"{folder}_w{i:04d}.npz"
            out_path = os.path.join(args.outdir, split, fname)
            save_sample(pcs, skels, frame_labels, majority_label, out_path)
            total_samples += 1
            split_counts[split] += 1
            class_counts[ACTION_NAMES[majority_label]] += 1

    print(f"\n完成！共生成 {total_samples} 个样本")
    print(f"  Train: {split_counts['train']}")
    print(f"  Val:   {split_counts['val']}")
    print(f"  Test:  {split_counts['test']}")
    print(f"\n类别分布：")
    for name in ACTION_NAMES:
        print(f"  {name:12s}: {class_counts.get(name, 0)}")


if __name__ == "__main__":
    main()
