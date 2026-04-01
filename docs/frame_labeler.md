# 帧级动作标注说明

## 概述

针对 `subdataset` 中的 4 种混合动作序列（bend、sit、lie、hunker），每段序列由"走路"与某种特定动作交替构成。`frame_labeler.py` 基于每帧的 17 点骨架坐标，自动为每帧打上 `walking` 或对应动作名称的标签，输出结果保存为 `frame_labels.json`。

---

## 数据背景

每个样本文件夹下的 `syn_data.json` 包含若干帧，每帧有 17 个骨架关节点的三维坐标（x, y, z），关节顺序如下：

```
HEAD, NECK, SPINE_CHEST, SPINE_NAVEL, PELVIS,
SHOULDER_LEFT, ELBOW_LEFT, WRIST_LEFT,
SHOULDER_RIGHT, ELBOW_RIGHT, WRIST_RIGHT,
HIP_LEFT, KNEE_LEFT, ANKLE_LEFT,
HIP_RIGHT, KNEE_RIGHT, ANKLE_RIGHT
```

坐标系中 z 轴为竖直方向（高度）。

---

## 标注方法

### 整体流程

```
加载 syn_data.json
  → 提取骨架序列 (N, 17, 3)
  → [可选] 修复骨架瞬时翻转
  → 计算逐帧判别特征
  → 移动平均平滑（窗口 11 帧）
  → 阈值判定 → 原始标签序列
  → 去除短片段（< 15 帧的片段并入前一段）
  → 输出 frame_labels.json
```

### 各动作判别规则

每种混合动作只有两个状态：`walking` 和对应动作名。判别基于单一主特征，阈值说明如下：

| 动作 | 判别特征 | walking 条件 | action 条件 | 特征说明 |
|------|----------|-------------|-------------|---------|
| **bend** | 躯干角度 | < 25° | ≥ 25° | PELVIS→SPINE_CHEST 向量与 z 轴的夹角；弯腰时前倾，角度显著增大（走路 ~6°，弯腰 ~59°） |
| **sit** | 骨盆高度（相对） | > 0.75 × 参考值 | ≤ 0.75 × 参考值 | 骨盆 z 坐标；坐下时骨盆大幅下降，参考值取序列骨盆 z 的第 90 百分位 |
| **lie** | 头部高度（相对） | > 0.75 × 参考值 | ≤ 0.75 × 参考值 | 头部 z 坐标；躺下后头部高度极低且在起身过渡阶段仍明显低于站立高度，参考值取序列头部 z 的第 90 百分位 |
| **hunker** | 骨盆高度（相对） | > 0.65 × 参考值 | ≤ 0.65 × 参考值 | 骨盆 z 坐标；深蹲时骨盆下降幅度大于坐下，阈值更低（0.65 vs 0.75） |

**关于 lie 的说明**：早期版本使用 `头部z - 骨盆z` 差值（阈值 0.35）作为特征，但起身过渡阶段头部已抬起而骨盆仍低，导致差值超过阈值被误判为 walking，产生碎片段。改用头部绝对高度（相对于自身站立参考值）后，过渡阶段能正确保持 lie 标签。

### 平滑与后处理

- **移动平均平滑**：对原始特征序列做窗口大小 11 的移动平均，消除帧间噪声抖动，使标签边界更平滑。窗口大小强制为奇数。
- **短片段去除**：平滑后仍可能出现极短的孤立片段（如 2-3 帧的翻转），将长度 < 15 帧的片段并入前一片段的标签。

### 骨架翻转修复（可选）

某些帧的骨架数据存在瞬时水平翻转（x/y 轴镜像）。检测方法：将当前帧相对于质心做 x/y 翻转后，比较与前一帧的关节距离总和；若翻转后距离更小，则认为发生了翻转并自动修正。通过 `--fix-flip` 参数启用。

---

## 输出格式

每个样本文件夹下生成 `frame_labels.json`，与 `syn_data.json` 并列：

```json
{
  "source": "807_djf_TopDownView_daily_bend_001",
  "action_type": "bend",
  "total_frames": 1422,
  "labels": ["walking", "walking", "bend", "bend", ...],
  "segments": [
    {"start": 0,   "end": 89,  "label": "walking"},
    {"start": 90,  "end": 230, "label": "bend"},
    {"start": 231, "end": 495, "label": "walking"},
    ...
  ]
}
```

- `labels`：长度等于总帧数，`labels[i]` 即第 i 帧的标签，可直接按帧索引取用
- `segments`：游程编码，每段连续相同标签的起止帧号，方便快速查看动作分布

---

## 使用方法

```bash
# 标注单种动作
python frame_labeler.py --action bend

# 标注全部 4 种混合动作
python frame_labeler.py --action all

# 启用骨架翻转修复
python frame_labeler.py --action bend --fix-flip

# 覆盖已有标签文件
python frame_labeler.py --action all --overwrite

# 自定义平滑窗口和最短片段长度
python frame_labeler.py --action sit --smooth-window 15 --min-segment 20
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--action` | 必填 | `bend` / `sit` / `lie` / `hunker` / `all` |
| `--data-root` | `subdataset/` | 数据根目录 |
| `--fix-flip` | 关闭 | 启用骨架翻转修复 |
| `--smooth-window` | 11 | 特征移动平均窗口大小 |
| `--min-segment` | 15 | 最短片段帧数（低于此长度的片段并入前段） |
| `--overwrite` | 关闭 | 覆盖已有 frame_labels.json |

---

## 数据统计参考

通过骨架特征分析得到的各动作典型数值，可作为阈值调整的参考：

| 动作 | walking 阶段 | action 阶段 |
|------|-------------|------------|
| bend | 躯干角 ~6°，骨盆z ~0.97 | 躯干角 ~59°，骨盆z ~0.80 |
| sit  | 骨盆z ~0.99，膝盖角 ~157° | 骨盆z ~0.54，膝盖角 ~93° |
| lie  | 头部z ~1.40，头-骨盆差 ~0.57 | 头部z ~0.46，头-骨盆差 ~0.01 |
| hunker | 骨盆z ~0.97，膝盖角 ~157° | 骨盆z ~0.42，膝盖角 ~63° |

sit 与 hunker 均通过骨盆高度下降来判别，区别在于阈值系数（0.75 vs 0.65）和膝盖弯曲程度（93° vs 63°）。
