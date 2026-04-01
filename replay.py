import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
plt.rcParams['font.sans-serif'] = ['SimHei']      # 黑体
plt.rcParams['axes.unicode_minus'] = False

# 骨架连接关系（定义骨骼线段）
SKELETON_CONNECTIONS = [
    # 脊柱
    ("HEAD", "NECK"),
    ("NECK", "SPINE_CHEST"),
    ("SPINE_CHEST", "SPINE_NAVEL"),
    ("SPINE_NAVEL", "PELVIS"),
    # 左臂
    ("NECK", "SHOULDER_LEFT"),
    ("SHOULDER_LEFT", "ELBOW_LEFT"),
    ("ELBOW_LEFT", "WRIST_LEFT"),
    # 右臂
    ("NECK", "SHOULDER_RIGHT"),
    ("SHOULDER_RIGHT", "ELBOW_RIGHT"),
    ("ELBOW_RIGHT", "WRIST_RIGHT"),
    # 左腿
    ("PELVIS", "HIP_LEFT"),
    ("HIP_LEFT", "KNEE_LEFT"),
    ("KNEE_LEFT", "ANKLE_LEFT"),
    # 右腿
    ("PELVIS", "HIP_RIGHT"),
    ("HIP_RIGHT", "KNEE_RIGHT"),
    ("KNEE_RIGHT", "ANKLE_RIGHT"),
]


def load_data(filepath):
    # 允许传入目录路径：自动在目录内寻找数据 JSON。
    if os.path.isdir(filepath):
        candidates = ["syn_data.json", "data.json"]
        for name in candidates:
            candidate_path = os.path.join(filepath, name)
            if os.path.isfile(candidate_path):
                filepath = candidate_path
                break
        else:
            raise FileNotFoundError(
                f"目录中未找到可用数据文件: {filepath}，期望文件名之一: {', '.join(candidates)}"
            )

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"数据文件不存在或不可读: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f), filepath


def compute_axis_limits(data):
    """预先计算所有帧的坐标范围，保证动画中视角固定不跳动"""
    all_x, all_y, all_z = [], [], []
    for frame in data:
        for pt in frame["pointcloud_data"]["points"]:
            all_x.append(pt["x"])
            all_y.append(pt["y"])
            all_z.append(pt["z"])
        for joint in frame["skeleton_data"].values():
            all_x.append(joint["x"])
            all_y.append(joint["y"])
            all_z.append(joint["z"])
    margin = 0.1
    return (
        (min(all_x) - margin, max(all_x) + margin),
        (min(all_y) - margin, max(all_y) + margin),
        (min(all_z) - margin, max(all_z) + margin),
    )


def main():
    parser = argparse.ArgumentParser(description="离线回放点云 + 骨架数据")
    parser.add_argument("--file", default="D:\\Deep_learning_project\\motion_estimate\\classification_task\\subdataset\\goosestep\\807_hzj_TopDownView_daily_goosestep_001\\syn_data.json", help="JSON 数据文件路径")
    parser.add_argument("--fps", type=int, default=15, help="回放帧率 (默认 15)")
    parser.add_argument("--start", type=int, default=0, help="起始帧号 (默认 0)")
    parser.add_argument("--save-gif", action="store_true", help="开启 GIF 导出（默认关闭）")
    args = parser.parse_args()

    gif_path = "replay.gif"
    gif_fps = None
    gif_export_frame_start = 1
    gif_export_frame_end = 300

    print(f"加载数据: {args.file} ...")
    data, data_file = load_data(args.file)
    total_frames = len(data)

    # 尝试加载帧级动作标签
    label_path = os.path.join(os.path.dirname(data_file), "frame_labels.json")
    frame_labels = None
    if os.path.isfile(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        frame_labels = label_data.get("labels", None)
        if frame_labels and len(frame_labels) == total_frames:
            print(f"已加载帧级标签: {label_path} ({len(label_data.get('segments', []))} segments)")
        else:
            print(f"帧级标签长度不匹配或为空，已忽略")
            frame_labels = None
    else:
        print(f"未找到帧级标签文件，跳过: {label_path}")

    print(f"共 {total_frames} 帧，计算坐标范围...")

    xlim, ylim, zlim = compute_axis_limits(data)

    # ---- 状态变量 ----
    state = {
        "paused": False,
        "current_frame": args.start,
        "interval": 1000 // args.fps,  # ms per frame
        "show_skeleton": True,
        "exporting": False,
        "slider_syncing": False,
    }

    # ---- 创建画布 ----
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title("点云 + 骨架回放")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.20, right=0.82)

    # 骨架显示切换按钮
    btn_ax = fig.add_axes([0.02, 0.06, 0.14, 0.05])
    btn_skel = Button(btn_ax, "隐藏骨架")

    # 进度条（拖动跳帧）
    slider_ax = fig.add_axes([0.22, 0.07, 0.72, 0.03])
    slider_max = max(1, total_frames - 1)
    frame_slider = Slider(
        slider_ax,
        "进度",
        0,
        slider_max,
        valinit=min(args.start, total_frames - 1),
        valstep=1,
        valfmt="%0.0f",
    )

    def toggle_skeleton(event):
        state["show_skeleton"] = not state["show_skeleton"]
        btn_skel.label.set_text("显示骨架" if not state["show_skeleton"] else "隐藏骨架")
        fig.canvas.draw_idle()

    def on_slider_change(val):
        if state["slider_syncing"] or state["exporting"]:
            return
        state["current_frame"] = min(max(int(val), 0), total_frames - 1)
        if state["paused"]:
            update(None)
            fig.canvas.draw_idle()

    btn_skel.on_clicked(toggle_skeleton)
    frame_slider.on_changed(on_slider_change)

    # 初始化绘图元素
    pc_scatter = ax.scatter([], [], [], s=4, c=[], cmap="coolwarm", alpha=0.6, label="点云")
    skeleton_lines = []
    for _ in SKELETON_CONNECTIONS:
        (line,) = ax.plot([], [], [], "o-", color="lime", linewidth=2, markersize=4)
        skeleton_lines.append(line)
    joint_scatter = ax.scatter([], [], [], s=40, c="red", marker="o", depthshade=True, label="关节")

    # 标题文本
    title_text = ax.set_title("", fontsize=11)
    # 右侧动作标签文本（放在图窗右侧留白区）
    action_text = fig.text(
        0.84,
        0.55,
        "动作: --",
        fontsize=12,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=8)

    # ---- 更新函数 ----
    def update(frame_idx_unused):
        if state["exporting"] and frame_idx_unused is not None:
            idx = int(frame_idx_unused)
        else:
            idx = state["current_frame"]
        if idx >= total_frames:
            idx = total_frames - 1
            state["paused"] = True

        if not state["exporting"]:
            slider_target = min(max(idx, frame_slider.valmin), frame_slider.valmax)
            if int(frame_slider.val) != int(slider_target):
                state["slider_syncing"] = True
                frame_slider.set_val(slider_target)
                state["slider_syncing"] = False

        frame = data[idx]
        pts = frame["pointcloud_data"]["points"]
        skel = frame["skeleton_data"]

        # 点云
        px = [p["x"] for p in pts]
        py = [p["y"] for p in pts]
        pz = [p["z"] for p in pts]
        pv = [p["v"] for p in pts]

        pc_scatter._offsets3d = (px, py, pz)
        if pv:
            pc_scatter.set_array(np.array(pv))
            pc_scatter.set_clim(min(pv), max(pv))

        # 骨架线段
        for line, (j1, j2) in zip(skeleton_lines, SKELETON_CONNECTIONS):
            if state["show_skeleton"] and j1 in skel and j2 in skel:
                line.set_data_3d(
                    [skel[j1]["x"], skel[j2]["x"]],
                    [skel[j1]["y"], skel[j2]["y"]],
                    [skel[j1]["z"], skel[j2]["z"]],
                )
                line.set_visible(True)
            else:
                line.set_visible(False)

        # 关节点
        if state["show_skeleton"]:
            jx = [j["x"] for j in skel.values()]
            jy = [j["y"] for j in skel.values()]
            jz = [j["z"] for j in skel.values()]
            joint_scatter._offsets3d = (jx, jy, jz)
            joint_scatter.set_visible(True)
        else:
            joint_scatter._offsets3d = ([], [], [])
            joint_scatter.set_visible(False)

        # 标题信息
        status = "暂停" if state["paused"] else "播放"
        fps_display = 1000 / state["interval"] if state["interval"] > 0 else 0
        title_text.set_text(
            f"{status}  帧: {idx}/{total_frames - 1}  |  "
            f"点数: {frame['pointcloud_data']['num_points']}  |  "
            f"FPS: {fps_display:.0f}  |  "
            f"时间: {frame['pointcloud_timestamp']}"
        )

        if frame_labels:
            action_text.set_text(f"动作: {frame_labels[idx]}")
        else:
            action_text.set_text("动作: 无标签")

        # 自动推进帧号
        if not state["paused"] and not state["exporting"]:
            state["current_frame"] = min(idx + 1, total_frames - 1)

        return [pc_scatter, joint_scatter, title_text, action_text] + skeleton_lines

    # ---- 键盘事件 ----
    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "right" and state["paused"]:
            state["current_frame"] = min(state["current_frame"] + 1, total_frames - 1)
        elif event.key == "left" and state["paused"]:
            state["current_frame"] = max(state["current_frame"] - 1, 0)
        elif event.key in ("+", "="):
            state["interval"] = max(10, state["interval"] - 10)
            anim.event_source.interval = state["interval"]
        elif event.key in ("-", "_"):
            state["interval"] = min(500, state["interval"] + 10)
            anim.event_source.interval = state["interval"]
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ---- 启动动画 ----
    anim = FuncAnimation(
        fig,
        update,
        interval=state["interval"],
        blit=False,
        cache_frame_data=False,
    )

    if args.save_gif:
        anim.event_source.stop()
        export_start = max(0, gif_export_frame_start - 1)
        export_end = min(total_frames - 1, gif_export_frame_end - 1)

        if export_start > export_end:
            print(f"GIF 导出失败: 导出区间无效 ({gif_export_frame_start}-{gif_export_frame_end})")
            plt.show()
            return

        export_fps = gif_fps if gif_fps is not None else args.fps
        export_fps = max(1, export_fps)
        export_total = export_end - export_start + 1

        print(f"开始导出 GIF: {gif_path}")
        print(f"导出范围: {export_start + 1} -> {export_end + 1}, FPS: {export_fps}")

        state["exporting"] = True
        export_anim = FuncAnimation(
            fig,
            update,
            frames=range(export_start, export_end + 1),
            interval=1000 // export_fps,
            blit=False,
            repeat=False,
            cache_frame_data=False,
        )
        try:
            export_anim.save(
                gif_path,
                writer=PillowWriter(fps=export_fps),
                progress_callback=lambda i, n: print(f"导出进度: {i + 1}/{n}") if ((i + 1) % 30 == 0 or (i + 1) == export_total) else None,
            )
            print(f"GIF 导出完成: {gif_path}")
        except Exception as e:
            print(f"GIF 导出失败: {e}")
            print("提示: 如缺少依赖可执行: python -m pip install pillow")
        finally:
            state["exporting"] = False
            anim.event_source.start()

    print("回放窗口已打开。快捷键: 空格=暂停/继续, 左右=逐帧, +/-=调速, 拖动进度条=跳帧, q=退出")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
