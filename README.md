# Radar-based Action Classification

基于毫米波雷达点云的人体动作分类系统。利用预训练的 `RadarPoseNet` 骨架估计模型作为 backbone，在其基础上添加时序分类头，实现逐帧动作识别。

## 项目概述

本项目使用毫米波雷达采集的三维点云数据，通过以下流程完成动作分类：

1. **点云输入** — 雷达每帧采集包含 (x, y, z, v) 的稀疏点云
2. **骨架估计** — 预训练 RadarPoseNet 从点云中估计 17 关节人体骨架
3. **特征提取** — 提取隐藏层特征、归一化骨架姿态和运动速度特征
4. **动作分类** — 时序卷积 + MLP 分类头输出逐帧动作类别

### 支持的动作类别（10 类）

| 编号 | 英文名 | 中文名 | 描述 |
|------|--------|--------|------|
| 0 | walking | 行走 | 正常步行 |
| 1 | sit | 坐下 | 从站立到坐下 |
| 2 | bend | 弯腰 | 弯腰动作 |
| 3 | jogging | 慢跑 | 慢跑运动 |
| 4 | boxing | 拳击 | 出拳动作 |
| 5 | jumpjack | 开合跳 | 开合跳运动 |
| 6 | lie | 躺下 | 从站立到躺下 |
| 7 | hunker | 蹲下 | 下蹲动作 |
| 8 | sweep | 扫地 | 扫地动作 |
| 9 | goosestep | 正步 | 正步行进 |

## 项目结构

```
classification_task/
├── train_action.py          # 核心训练脚本（模型定义 + 训练循环）
├── data_split_action.py     # 数据预处理与切片
├── frame_labeler.py         # 混合动作序列的自动帧级标注
├── inference_replay.py      # 推理回放可视化（GT vs 预测对比）
├── replay.py                # 原始数据 3D 回放工具
├── docs/                    # 文档
│   └── frame_labeler.md     # 帧级标注工具说明
├── subdataset/              # 原始数据集（未上传）
├── action_dataset/          # 处理后的训练数据（未上传）
├── best_action_model.pth    # 训练好的模型权重（未上传）
├── runs/                    # TensorBoard 日志（未上传）
└── result/                  # 输出视频（未上传）
```

## 环境依赖

- Python 3.8+
- PyTorch >= 2.0（支持 CUDA）
- NumPy
- SciPy
- matplotlib
- tqdm
- TensorBoard

```bash
pip install torch numpy scipy matplotlib tqdm tensorboard
```

## 数据格式

### 原始数据 (`syn_data.json`)

每个样本文件夹包含一个 `syn_data.json`，存储逐帧的点云和骨架数据：

```json
[
  {
    "pointcloud_data": {
      "num_points": 85,
      "points": [{"x": 0.1, "y": 2.3, "z": 0.8, "v": 0.5}, ...]
    },
    "skeleton_data": {
      "HEAD": {"x": 0.0, "y": 2.1, "z": 1.7},
      "NECK": {"x": 0.0, "y": 2.1, "z": 1.5},
      ...
    }
  },
  ...
]
```

### 骨架关节定义（17 关节）

```
HEAD, NECK, SPINE_CHEST, SPINE_NAVEL, PELVIS,
SHOULDER_LEFT, ELBOW_LEFT, WRIST_LEFT,
SHOULDER_RIGHT, ELBOW_RIGHT, WRIST_RIGHT,
HIP_LEFT, KNEE_LEFT, ANKLE_LEFT,
HIP_RIGHT, KNEE_RIGHT, ANKLE_RIGHT
```

### 训练数据 (`.npz`)

经 `data_split_action.py` 处理后生成，每个文件包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `pointcloud` | `(T,)` object array | 每帧点云 `(N_i, 4)` float32 |
| `skeleton` | `(T, 17, 3)` float32 | GT 骨架坐标 |
| `frame_labels` | `(T,)` int64 | 逐帧动作标签 |
| `majority_label` | int64 标量 | 多数投票主标签 |

其中 `T` 为可变长度（40-75 帧）。

## 使用方法

### 1. 数据预处理

对混合动作序列（bend、sit、lie、hunker）先运行帧级标注：

```bash
python frame_labeler.py --action all --data-root subdataset/ --fix-flip --overwrite
```

然后切片生成训练数据：

```bash
python data_split_action.py --data-root subdataset/ --outdir action_dataset/
```

主要参数：
- `--min-len` / `--max-len` — 窗口帧数范围（默认 40-75）
- `--slide-step` — 长段滑窗步长（默认 25）
- `--trim` — 纯动作序列首尾裁剪帧数（默认 15）
- `--split` — train/val/test 比例（默认 0.8 0.1 0.1）

### 2. 训练

```bash
python train_action.py
```

训练采用两阶段策略：

| 阶段 | Epoch 范围 | Backbone | 分类头 | 学习率 |
|------|-----------|----------|--------|--------|
| Warmup | 1-10 | 全部冻结 | 可训练 | 3e-4 |
| Finetune | 11-100 | Temporal 层解冻 | 可训练 | Backbone 1e-5, 分类头 3e-4 |

其他训练特性：
- 逆频率类别加权 + Label Smoothing (0.1)
- 骨架辅助损失（L1，权重 0.1）
- CosineAnnealing 学习率调度
- Early Stopping（patience=20）
- 数据增强：随机缩放、旋转、平移、噪声

冒烟测试（验证模型结构和前向传播）：

```bash
python train_action.py --smoke
```

TensorBoard 监控：

```bash
tensorboard --logdir runs/
```

### 3. 推理回放

```bash
python inference_replay.py
```

双面板 3D 可视化：左侧显示 GT 骨架，右侧显示模型预测骨架和动作标签。

特性：
- 滑窗推理 + 高斯加权平均（减少边界抖动）
- Savitzky-Golay 时序平滑
- 支持多段 JSON 拼接连续播放
- 支持保存为 MP4 视频

快捷键：空格=暂停/继续，左右=逐帧，+/-=调速，q=退出

### 4. 数据回放

```bash
python replay.py --file path/to/syn_data.json
```

3D 交互式回放原始点云和骨架数据，支持帧级标签显示、GIF 导出。

## 模型架构

```
RadarActionNet
├── Backbone: RadarPoseNet (预训练)
│   ├── PointEmbedding          # 点云特征嵌入
│   ├── PositionalEncoding      # 3D 位置编码
│   ├── SpatialLayers ×3        # 空间 Transformer（始终冻结）
│   ├── MaskedMeanPool           # 掩码平均池化
│   ├── TemporalLayers ×8       # 时序 Transformer（Finetune 阶段解冻）
│   └── Head                     # 骨架回归头 → (B, T, 51)
│
└── ActionClassifier (分类头)
    ├── LayerNorm                # 前置归一化
    ├── CausalConv1D (k=5)       # 时序因果卷积
    ├── Residual + LayerNorm     # 残差连接
    └── MLP: 314 → 256 → 128 → 10  # 深层分类器
```

分类头输入特征（314 维）：
- 256d — Backbone 隐藏层特征
- 51d — 归一化骨架姿态（相对盆骨 + 身高归一化）
- 7d — 运动速度特征（盆骨 3D 速度 + 膝踝 Z 轴速度）

## 帧级标注工具

`frame_labeler.py` 用于为混合动作序列（bend、sit、lie、hunker）自动生成逐帧标签。

各动作的判别特征：

| 动作 | 特征 | 阈值 | 判定方向 |
|------|------|------|----------|
| bend | 躯干角度（盆骨→胸骨 vs Z 轴） | 25° | >= 阈值为 bend |
| sit | 盆骨 Z 坐标 | 站立参考的 75% | <= 阈值为 sit |
| lie | 头部 Z 坐标 | 站立参考的 75% | <= 阈值为 lie |
| hunker | 盆骨 Z 坐标 | 站立参考的 65% | <= 阈值为 hunker |

处理流程：骨架翻转修正 → 特征计算 → 移动平均平滑 → 阈值判定 → 短段合并 → 输出 `frame_labels.json`

## License

本项目仅用于学术研究。