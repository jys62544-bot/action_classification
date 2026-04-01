"""
推理回放脚本
加载原始 JSON 点云数据，通过 RadarActionNet 推理预测骨架和动作类别，
左右双面板：左=点云+GT骨架，右=点云+预测骨架，
最右侧醒目标签显示预测动作。
支持多段 JSON 文件拼接连续播放。

推理预处理完全对齐训练数据流水线（data_split_action.py + data_split.py）：
  1. 速度归一化: v = clip(v / max_velocity, -1, 1)
  2. 骨架 Savitzky-Golay 平滑
  3. 滑窗推理，每窗口独立做 XY 质心归一化
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from scipy.signal import savgol_filter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 支持多段 JSON 拼接，按列表顺序拼接后连续播放
JSON_PATHS = [
    os.path.join(SCRIPT_DIR, "subdataset", "walking","807_djf_TopDownView_daily_walking_001", "syn_data.json"),
]
BACKBONE_PATH = os.path.join(SCRIPT_DIR, "..", "best_model_0109.pth")
CKPT_PATH     = os.path.join(SCRIPT_DIR, "best_action_model (1).pth")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
FPS           = 15
START_FRAME   = 0
MAX_VELOCITY  = 5.0     # 与 data_split_action.py 的 max_velocity 一致
WINDOW_SIZE   = 25       # 滑窗大小（短窗口避免跨动作边界污染）
WINDOW_STEP   = 3        # 滑窗步长
SAVE          = False    # True 时将回放保存为视频（mp4），不弹出交互窗口
SAVE_PATH     = os.path.join(SCRIPT_DIR, "result", "walking.mp4")  # 输出视频路径

# 将父目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_action import (
    ActionConfig, RadarActionNet, load_pretrained_backbone,
    ACTION_NAMES, NUM_CLASSES,
)
from train_zd_v2 import pad_or_sample_points

# JSON 骨架关节名 → 模型索引的映射顺序
JOINT_ORDER = [
    "HEAD", "NECK", "SPINE_CHEST", "SPINE_NAVEL", "PELVIS",
    "SHOULDER_LEFT", "ELBOW_LEFT", "WRIST_LEFT",
    "SHOULDER_RIGHT", "ELBOW_RIGHT", "WRIST_RIGHT",
    "HIP_LEFT", "KNEE_LEFT", "ANKLE_LEFT",
    "HIP_RIGHT", "KNEE_RIGHT", "ANKLE_RIGHT",
]

# 骨骼连接（index-based）
BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # 脊柱
    (1, 5), (5, 6), (6, 7),               # 左臂
    (1, 8), (8, 9), (9, 10),              # 右臂
    (4, 11), (11, 12), (12, 13),          # 左腿
    (4, 14), (14, 15), (15, 16),          # 右腿
]

# 动作中文名映射
ACTION_NAMES_CN = {
    "walking": "行走", "sit": "坐下", "bend": "弯腰", "jogging": "慢跑",
    "boxing": "拳击", "jumpjack": "开合跳", "lie": "躺下", "hunker": "蹲下",
    "sweep": "扫地", "goosestep": "正步",
}


# ==========================================
# 数据加载 — 完全对齐 data_split_action.py 的 load_sequence
# ==========================================
def load_json(json_path):
    """
    加载单个 syn_data.json，复现训练预处理：
      - 点云速度归一化: clip(v / MAX_VELOCITY, -1, 1)
      - 骨架 Savitzky-Golay 平滑

    返回:
      raw_pcs_display: 原始点云列表（用于可视化，不做速度归一化）
      model_pcs: 模型输入点云列表（速度已归一化）
      gt_skel_display: 原始 GT 骨架（用于可视化）
      gt_skel_smooth: 平滑后 GT 骨架（与模型训练时的 GT 一致）
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_pcs_display = []
    model_pcs = []
    gt_skels = []

    for frame in data:
        # 点云
        pts = frame["pointcloud_data"]["points"]
        pc = np.array([[p["x"], p["y"], p["z"], p["v"]] for p in pts],
                      dtype=np.float32)
        raw_pcs_display.append(pc.copy())

        # 速度归一化（模型输入用）
        pc_model = pc.copy()
        if pc_model.shape[0] > 0:
            pc_model[:, 3] = np.clip(pc_model[:, 3] / MAX_VELOCITY, -1.0, 1.0)
        model_pcs.append(pc_model)

        # GT 骨架
        skel = frame["skeleton_data"]
        joints = np.array([[skel[j]["x"], skel[j]["y"], skel[j]["z"]]
                           for j in JOINT_ORDER], dtype=np.float32)
        gt_skels.append(joints)

    gt_skel_display = np.stack(gt_skels, axis=0)  # (T, 17, 3) 原始

    # 骨架平滑（与训练一致）
    gt_skel_smooth = gt_skel_display.copy()
    T = gt_skel_smooth.shape[0]
    if T >= 5:
        for j in range(17):
            for c in range(3):
                gt_skel_smooth[:, j, c] = savgol_filter(
                    gt_skel_smooth[:, j, c], 5, 2)

    return raw_pcs_display, model_pcs, gt_skel_display, gt_skel_smooth


def normalize_window_xy(pcs, origin_xy):
    """对一个窗口的点云做 XY 归一化。"""
    norm_pcs = []
    for pc in pcs:
        pc_copy = pc.copy()
        if pc_copy.shape[0] > 0:
            pc_copy[:, :2] -= origin_xy
        norm_pcs.append(pc_copy)
    return norm_pcs


def find_window_origin(pcs):
    """
    复现 data_split_action.py 的 normalize_window 逻辑：
    从末帧向前回溯，找到第一个点数>10的帧，取其 xyz 均值的 xy 分量。
    """
    for t in range(len(pcs) - 1, -1, -1):
        if pcs[t].shape[0] > 10:
            return pcs[t][:, :3].mean(axis=0)[:2]  # xy 分量
    # fallback: 用任意有点的帧
    for t in range(len(pcs) - 1, -1, -1):
        if pcs[t].shape[0] > 0:
            return pcs[t][:, :3].mean(axis=0)[:2]
    return np.zeros(2, dtype=np.float32)


# ==========================================
# 模型
# ==========================================
def build_model(config):
    """构建 RadarActionNet 并加载权重。"""
    train_action_dir = os.path.dirname(os.path.abspath(
        sys.modules["train_action"].__file__
    ))
    config.pretrained_path = os.path.relpath(
        os.path.abspath(BACKBONE_PATH), train_action_dir
    )

    backbone = load_pretrained_backbone(config)
    model = RadarActionNet(config, backbone, freeze_temporal=True)

    print(f"加载动作分类权重：{CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict) and "classifier" in ckpt:
        model.classifier.load_state_dict(ckpt["classifier"])
        model.backbone.temporal_layers.load_state_dict(ckpt["temporal_layers"])
        model.backbone.head.load_state_dict(ckpt["head"])
        print("  已加载完整权重（classifier + temporal_layers + head）")
    else:
        model.classifier.load_state_dict(ckpt)
        print("  已加载 classifier 权重")

    device = torch.device(DEVICE)
    model = model.to(device)
    model.eval()
    return model, device


def run_inference_window(model, pcs_norm, config, device):
    """
    对一个归一化窗口推理。
    pcs_norm: 归一化后的点云列表 (每元素 (N_i, 4))
    返回: pred_skel (W, 17, 3), pred_labels (W,), pred_conf (W,)
    """
    # pad/sample
    processed = []
    for pc in pcs_norm:
        processed.append(pad_or_sample_points(pc, config.points_per_frame))
    pointcloud = np.stack(processed, axis=0).astype(np.float32)  # (W, N, 4)

    # padding mask
    is_padding = np.all(pointcloud[:, :, :3] == 0, axis=2)
    for t in range(len(pcs_norm)):
        if np.all(is_padding[t]):
            is_padding[t, 0] = False
            pointcloud[t, 0, :] = 0.0

    W = pointcloud.shape[0]
    x = torch.from_numpy(pointcloud).unsqueeze(0).to(device)
    mask = torch.from_numpy(is_padding).unsqueeze(0).to(device)
    time_mask = torch.ones(1, W, dtype=torch.bool, device=device)

    with torch.no_grad():
        outputs = model(x, mask=mask, time_mask=time_mask)

    pred_skel = outputs["skeleton"][0].cpu().numpy()       # (W, 17, 3)
    logits = outputs["action_frame"][0].cpu().numpy()      # (W, num_classes)

    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # (W, num_classes)

    return pred_skel, probs.argmax(axis=1), probs.max(axis=1), probs


def sliding_window_inference(model, model_pcs, config, device,
                             window_size=WINDOW_SIZE, step=WINDOW_STEP):
    """
    滑窗推理整段序列。每个窗口独立做 XY 归一化，推理后还原。
    重叠区域使用高斯加权平均（窗口中心帧权重最高），减少边界抖动。
    最后对预测骨架做时序平滑（Savitzky-Golay）。

    返回: pred_skel (T, 17, 3) 原始坐标, pred_labels (T,), pred_conf (T,)
    """
    T = len(model_pcs)
    # 累加器：加权求和 + 权重总和
    skel_accum = np.zeros((T, 17, 3), dtype=np.float64)
    skel_weight = np.zeros(T, dtype=np.float64)
    # 分类用置信度加权累加
    label_accum = np.zeros((T, config.num_classes), dtype=np.float64)
    pred_conf = np.zeros(T, dtype=np.float32)

    # 生成窗口起始位置
    starts = list(range(0, max(1, T - window_size + 1), step))
    # 确保最后一个窗口覆盖到末尾
    if starts[-1] + window_size < T:
        starts.append(max(0, T - window_size))

    # 预计算高斯权重（窗口中心权重最高）
    gauss_sigma = window_size / 4.0
    center = window_size / 2.0
    gauss_w = np.exp(-0.5 * ((np.arange(window_size) - center) / gauss_sigma) ** 2)

    for w_start in starts:
        w_end = min(w_start + window_size, T)
        w_len = w_end - w_start
        win_pcs = model_pcs[w_start:w_end]

        # 窗口级 XY 归一化（复现 normalize_window）
        origin_xy = find_window_origin(win_pcs)
        pcs_norm = normalize_window_xy(win_pcs, origin_xy)

        # 推理
        ws, wl, wc, wprobs = run_inference_window(model, pcs_norm, config, device)

        # 还原预测骨架到原始坐标
        ws[:, :, :2] += origin_xy

        # 加权累加（高斯权重）
        w = gauss_w[:w_len]
        skel_accum[w_start:w_end] += ws * w[:, None, None]
        skel_weight[w_start:w_end] += w

        # 分类标签：用完整概率分布加权累加
        for t_local in range(w_len):
            label_accum[w_start + t_local] += wprobs[t_local] * w[t_local]

    # 加权平均
    valid = skel_weight > 0
    pred_skel = np.zeros((T, 17, 3), dtype=np.float32)
    pred_skel[valid] = (skel_accum[valid] / skel_weight[valid, None, None]).astype(np.float32)

    # 分类标签：取累积置信度最高的类别
    pred_labels = label_accum.argmax(axis=1).astype(np.int64)
    label_sum = label_accum.sum(axis=1)
    label_sum[label_sum == 0] = 1.0
    pred_conf = (label_accum.max(axis=1) / label_sum).astype(np.float32)

    # 时序平滑：对预测骨架做 Savitzky-Golay 滤波（与训练时 GT 平滑一致）
    if T >= 5:
        for j in range(17):
            for c in range(3):
                pred_skel[:, j, c] = savgol_filter(pred_skel[:, j, c], 5, 2)

    return pred_skel, pred_labels, pred_conf


# ==========================================
# 可视化辅助
# ==========================================
def compute_axis_limits(raw_pcs, gt_skel, pred_skel):
    """计算统一坐标轴范围。"""
    all_coords = [gt_skel.reshape(-1, 3), pred_skel.reshape(-1, 3)]
    for pc in raw_pcs:
        if len(pc) > 0:
            all_coords.append(pc[:, :3])
    all_coords = np.concatenate(all_coords, axis=0)

    margin = 0.15
    return (
        (all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin),
        (all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin),
        (all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin),
    )


def setup_panel(ax, xlim, ylim, zlim, bone_color, joint_color):
    """初始化一个 3D 面板的绘图元素。"""
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    pc_scatter = ax.scatter([], [], [], s=4, c=[], cmap="coolwarm", alpha=0.5)
    bone_lines = []
    for _ in BONE_CONNECTIONS:
        (line,) = ax.plot([], [], [], "o-", color=bone_color, linewidth=2, markersize=3)
        bone_lines.append(line)
    joint_scatter = ax.scatter([], [], [], s=45, c=joint_color, marker="o", depthshade=True)
    return pc_scatter, bone_lines, joint_scatter


def update_panel(pc_scatter, bone_lines, joint_scatter, raw_pc, skeleton_t):
    """更新一个面板的绘图数据。"""
    if len(raw_pc) > 0:
        px, py, pz, pv = raw_pc[:, 0], raw_pc[:, 1], raw_pc[:, 2], raw_pc[:, 3]
        pc_scatter._offsets3d = (px, py, pz)
        pc_scatter.set_array(pv)
        if pv.max() > pv.min():
            pc_scatter.set_clim(pv.min(), pv.max())
    else:
        pc_scatter._offsets3d = ([], [], [])

    for line, (j1, j2) in zip(bone_lines, BONE_CONNECTIONS):
        line.set_data_3d(
            [skeleton_t[j1, 0], skeleton_t[j2, 0]],
            [skeleton_t[j1, 1], skeleton_t[j2, 1]],
            [skeleton_t[j1, 2], skeleton_t[j2, 2]],
        )
    joint_scatter._offsets3d = (skeleton_t[:, 0], skeleton_t[:, 1], skeleton_t[:, 2])


# ==========================================
# 主函数
# ==========================================
def main():
    # 校验文件
    for path in JSON_PATHS:
        if not os.path.isfile(path):
            print(f"错误：JSON 文件不存在：{path}"); sys.exit(1)
    for name, path in [("backbone", BACKBONE_PATH), ("ckpt", CKPT_PATH)]:
        if not os.path.isfile(path):
            print(f"错误：{name} 文件不存在：{path}"); sys.exit(1)

    config = ActionConfig()

    # ---- 数据加载（多段拼接） ----
    print(f"加载 {len(JSON_PATHS)} 段 JSON 数据...")
    all_raw_pcs = []       # 原始点云（用于可视化）
    all_model_pcs = []     # 速度归一化后的点云（用于推理输入）
    all_gt_skel = []       # 原始 GT 骨架（用于可视化）
    seg_boundaries = []

    for path in JSON_PATHS:
        raw_pcs_disp, model_pcs, gt_skel_disp, gt_skel_smooth = load_json(path)
        offset = len(all_raw_pcs)
        seg_boundaries.append((offset, offset + len(raw_pcs_disp),
                               os.path.basename(os.path.dirname(path))))
        all_raw_pcs.extend(raw_pcs_disp)
        all_model_pcs.extend(model_pcs)
        all_gt_skel.append(gt_skel_disp)

    gt_skel = np.concatenate(all_gt_skel, axis=0)
    T = len(all_raw_pcs)
    for start, end, name in seg_boundaries:
        print(f"  [{start:4d}-{end-1:4d}] {end - start:4d} 帧  {name}")
    print(f"总计 {T} 帧")

    # ---- 模型构建 ----
    model, device = build_model(config)

    # ---- 逐段滑窗推理 ----
    print("开始滑窗推理...")
    pred_skel_all = np.zeros((T, 17, 3), dtype=np.float32)
    pred_labels_all = np.zeros(T, dtype=np.int64)
    pred_conf_all = np.zeros(T, dtype=np.float32)

    for start, end, name in seg_boundaries:
        seg_pcs = all_model_pcs[start:end]
        ps, pl, pc_ = sliding_window_inference(model, seg_pcs, config, device)
        pred_skel_all[start:end] = ps
        pred_labels_all[start:end] = pl
        pred_conf_all[start:end] = pc_
        print(f"  {name}: {end - start} 帧推理完成")

    pred_skel = pred_skel_all
    pred_labels = pred_labels_all
    pred_conf = pred_conf_all
    print("推理完成。")

    # ---- 可视化 ----
    xlim, ylim, zlim = compute_axis_limits(all_raw_pcs, gt_skel, pred_skel)

    state = {
        "paused": False,
        "current_frame": min(START_FRAME, T - 1),
        "interval": 1000 // FPS,
        "slider_syncing": False,
    }

    fig = plt.figure(figsize=(20, 8))
    fig.canvas.manager.set_window_title("推理回放 — GT vs 预测")

    ax_gt   = fig.add_axes([0.01, 0.15, 0.38, 0.78], projection="3d")
    ax_pred = fig.add_axes([0.40, 0.15, 0.38, 0.78], projection="3d")

    ax_gt.set_title("Ground Truth 骨架", fontsize=12, pad=8)
    ax_pred.set_title("模型预测骨架", fontsize=12, pad=8)

    gt_pc_sc, gt_bones, gt_joints = setup_panel(
        ax_gt, xlim, ylim, zlim, bone_color="lime", joint_color="red")
    pred_pc_sc, pred_bones, pred_joints = setup_panel(
        ax_pred, xlim, ylim, zlim, bone_color="cyan", joint_color="dodgerblue")

    # ---- 最右侧醒目动作标签 ----
    action_label = fig.text(
        0.90, 0.65, "", fontsize=30, fontweight="bold",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#222222", edgecolor="white",
                  linewidth=2, alpha=0.9),
        color="white",
    )
    action_conf_text = fig.text(
        0.90, 0.52, "", fontsize=13, ha="center", va="center", color="gray",
    )
    seg_info_text = fig.text(
        0.90, 0.42, "", fontsize=10, ha="center", va="center", color="dimgray",
    )

    # ---- 底部状态栏 ----
    status_text = fig.text(0.40, 0.02, "", fontsize=10, ha="center", va="center")

    # ---- 进度条 ----
    slider_ax = fig.add_axes([0.08, 0.06, 0.68, 0.03])
    frame_slider = Slider(
        slider_ax, "进度", 0, max(1, T - 1),
        valinit=state["current_frame"], valstep=1, valfmt="%0.0f",
    )

    def find_seg(idx):
        for s, e, n in seg_boundaries:
            if s <= idx < e:
                return n, idx - s, e - s
        return "", idx, T

    def on_slider_change(val):
        if state["slider_syncing"]:
            return
        state["current_frame"] = min(max(int(val), 0), T - 1)
        if state["paused"]:
            update(None); fig.canvas.draw_idle()

    frame_slider.on_changed(on_slider_change)

    def update(frame_unused):
        idx = state["current_frame"]
        if idx >= T:
            idx = T - 1; state["paused"] = True

        if int(frame_slider.val) != idx:
            state["slider_syncing"] = True
            frame_slider.set_val(idx)
            state["slider_syncing"] = False

        update_panel(gt_pc_sc, gt_bones, gt_joints, all_raw_pcs[idx], gt_skel[idx])
        update_panel(pred_pc_sc, pred_bones, pred_joints, all_raw_pcs[idx], pred_skel[idx])

        action_en = ACTION_NAMES[pred_labels[idx]]
        action_cn = ACTION_NAMES_CN.get(action_en, action_en)
        action_label.set_text(action_cn)
        action_conf_text.set_text(f"{action_en}  {pred_conf[idx]:.0%}")

        seg_name, seg_frame, seg_len = find_seg(idx)
        seg_info_text.set_text(f"片段: {seg_name}\n段内: {seg_frame}/{seg_len - 1}")

        status = "暂停" if state["paused"] else "播放"
        fps_display = 1000 / state["interval"] if state["interval"] > 0 else 0
        status_text.set_text(
            f"{status}  |  帧: {idx}/{T - 1}  |  "
            f"FPS: {fps_display:.0f}  |  点数: {len(all_raw_pcs[idx])}"
        )

        if not state["paused"]:
            state["current_frame"] = min(idx + 1, T - 1)
        return []

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "right" and state["paused"]:
            state["current_frame"] = min(state["current_frame"] + 1, T - 1)
            update(None); fig.canvas.draw_idle()
        elif event.key == "left" and state["paused"]:
            state["current_frame"] = max(state["current_frame"] - 1, 0)
            update(None); fig.canvas.draw_idle()
        elif event.key in ("+", "="):
            state["interval"] = max(10, state["interval"] - 10)
            anim.event_source.interval = state["interval"]
        elif event.key in ("-", "_"):
            state["interval"] = min(500, state["interval"] + 10)
            anim.event_source.interval = state["interval"]
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    anim = FuncAnimation(
        fig, update, interval=state["interval"],
        blit=False, cache_frame_data=False,
    )

    if SAVE:
        import cv2
        from io import BytesIO
        from PIL import Image as PILImage

        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

        # 先渲染第0帧获取图像尺寸
        state["current_frame"] = START_FRAME
        update(None)
        fig.canvas.draw()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        img0 = np.array(PILImage.open(buf))
        h, w = img0.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(SAVE_PATH, fourcc, FPS, (w, h))

        print(f"保存视频到 {SAVE_PATH}，共 {T} 帧（FPS={FPS}）...")
        for i in range(T):
            state["current_frame"] = i
            state["paused"] = True
            update(None)
            fig.canvas.draw()
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img = np.array(PILImage.open(buf))
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if (i + 1) % 50 == 0 or i == T - 1:
                print(f"\r  {i + 1}/{T}", end="", flush=True)

        writer.release()
        print(f"\n视频已保存：{SAVE_PATH}")
        plt.close(fig)
    else:
        print("回放窗口已打开。快捷键: 空格=暂停/继续, 左右=逐帧, +/-=调速, q=退出")
        plt.show()


if __name__ == "__main__":
    main()
