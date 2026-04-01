"""
行为识别训练脚本
基于 RadarPoseNet 预训练 backbone + 分类头
"""
import os
import sys
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.amp
import time

from torch.utils.tensorboard import SummaryWriter

# 将父目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train_zd_v2 import (
    Config as PoseConfig,
    RadarPoseNet,
    pad_or_sample_points,
    JOINT_NAMES,
)
import train_zd_v2


# ==========================================
# 修补 PointTransformerLayer：恢复 self.attn 以匹配预训练权重
# ==========================================
class _PointTransformerLayerCompat(nn.Module):
    """与预训练 checkpoint 兼容的 PointTransformerLayer。
    checkpoint 中 spatial_layers 使用 nn.MultiheadAttention，
    而 train_zd_v2.py 已改为无参数的 F.scaled_dot_product_attention。
    此处恢复 self.attn 以完美加载权重。
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1),
        )

    def forward(self, x, pos_emb, key_padding_mask=None):
        residual = x
        x = self.norm1(x)
        q = k = x + pos_emb
        v = x

        feat, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask,
                             need_weights=False)

        x = x + feat
        x = self.norm1(x)

        residual = x
        x = self.mlp(x)
        x = x + residual
        x = self.norm2(x)
        return x


# monkey-patch：让 RadarPoseNet 构建时使用兼容版本
train_zd_v2.PointTransformerLayer = _PointTransformerLayerCompat

# ==========================================
# 行为类别定义
# ==========================================
ACTION_CLASSES = {
    "walking":   0, "sit":       1, "bend":      2, "jogging":   3,
    "boxing":    4, "jumpjack":  5, "lie":       6, "hunker":    7,
    "sweep":     8, "goosestep": 9,
}
ACTION_NAMES = list(ACTION_CLASSES.keys())
NUM_CLASSES = len(ACTION_CLASSES)


# ==========================================
# 1. 配置参数
# ==========================================
class ActionConfig:
    # 骨架模型参数（必须与预训练一致）
    data_root = "action_dataset"
    num_joints = 17
    input_channels = 4
    device = "cuda"
    seed = 42
    points_per_frame = 120
    dim_model = 256
    dim_feedforward = 1024
    num_heads = 4
    num_spatial_layers = 3
    num_temporal_layers = 8
    dropout = 0.2
    # 分类任务参数
    num_classes = NUM_CLASSES
    cls_mid_dim = 256
    cls_dropout = 0.5
    cls_temporal_kernel = 5
    motion_vel_dim = 7            # 盆骨xyz速度(3) + 左右膝踝z速度(4)
    pretrained_path = "../best_model_0109.pth"
    # 训练参数
    batch_size = 64              # 比原来小，可变长 padding 增加显存
    lr = 3e-4
    backbone_lr = 1e-5
    weight_decay = 1e-2
    epochs = 100
    warmup_epochs = 10
    patience = 20
    skel_aux_weight = 0.1


# ==========================================
# 2. ActionClassifier 分类头
# ==========================================
class ActionClassifier(nn.Module):
    """
    帧级分类头：时序卷积 + 深层 MLP。
    输入 (B, T, 314)，输出 (B, T, num_classes)。
    其中 314 = 256(隐藏特征) + 51(骨架归一化) + 7(运动速度)。
    """
    def __init__(self, config):
        super().__init__()
        in_dim = config.dim_model + config.num_joints * 3 + config.motion_vel_dim  # 256 + 51 + 7 = 314
        mid_dim = config.cls_mid_dim  # 256
        num_classes = config.num_classes  # 10
        ks = config.cls_temporal_kernel  # 5

        # 前置归一化
        self.norm = nn.LayerNorm(in_dim)

        # 时序卷积：捕获邻近帧上下文（因果填充，之后截断尾部）
        self.temporal_conv = nn.Conv1d(
            in_dim, in_dim, kernel_size=ks, padding=ks - 1, groups=1
        )
        self.temporal_norm = nn.LayerNorm(in_dim)
        self.temporal_act = nn.GELU()

        # 深层分类 MLP：314 → 256 → 128 → 10
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(config.cls_dropout),
            nn.Linear(mid_dim, mid_dim // 2),
            nn.GELU(),
            nn.Dropout(config.cls_dropout),
            nn.Linear(mid_dim // 2, num_classes),
        )

    def forward(self, x):
        """
        x: (B, T, 314)
        Returns: (B, T, num_classes) 逐帧分类 logits
        """
        x = self.norm(x)                          # (B, T, 314)

        # 时序卷积
        residual = x
        x_conv = x.transpose(1, 2)                # (B, C, T)
        x_conv = self.temporal_conv(x_conv)        # (B, C, T + ks - 1)
        x_conv = x_conv[:, :, :x.shape[1]]        # (B, C, T) 因果截断
        x_conv = x_conv.transpose(1, 2)            # (B, T, C)
        x = self.temporal_norm(x_conv + residual)  # 残差连接
        x = self.temporal_act(x)

        return self.mlp(x)                         # (B, T, num_classes)


# ==========================================
# 3. RadarActionNet 封装模型
# ==========================================
class RadarActionNet(nn.Module):
    """
    将预训练的 RadarPoseNet 作为 backbone。
    spatial layers 冻结，temporal layers 可微调。
    """
    def __init__(self, config, pose_model: RadarPoseNet, freeze_temporal=True):
        super().__init__()
        self.config = config
        self.backbone = pose_model
        self._finetune_mode = False  # warmup 阶段默认关闭

        # 冻结全部 backbone 参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 如果不冻结 temporal，则解冻 temporal_layers 和 head
        if not freeze_temporal:
            self._unfreeze_temporal()
            self._finetune_mode = True

        # 构建分类头
        self.classifier = ActionClassifier(config)

    def _unfreeze_temporal(self):
        """解冻 backbone 的 temporal_layers 和 head。"""
        for param in self.backbone.temporal_layers.parameters():
            param.requires_grad = True
        for param in self.backbone.head.parameters():
            param.requires_grad = True

    def freeze_temporal(self):
        """冻结 temporal layers（用于 warmup 阶段）。"""
        self._finetune_mode = False
        for param in self.backbone.temporal_layers.parameters():
            param.requires_grad = False
        for param in self.backbone.head.parameters():
            param.requires_grad = False

    def unfreeze_temporal(self):
        """解冻 temporal layers（warmup 结束后调用）。"""
        self._finetune_mode = True
        self._unfreeze_temporal()

    def forward(self, x, mask=None, time_mask=None):
        B, T, N, C = x.shape

        # === Spatial Encoder（始终冻结，无梯度） ===
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            x_f32 = x.float()
            mask_flat = mask.view(B * T, N)
            x_flat = x_f32.view(B * T, N, C)
            xyz = x_flat[:, :, :3]

            feat = self.backbone.point_emb(x_flat)
            pos_emb = self.backbone.pos_enc_gen(xyz)

            for layer in self.backbone.spatial_layers:
                feat = layer(feat, pos_emb, key_padding_mask=mask_flat)

            feat = self.backbone.pool(feat, key_padding_mask=mask_flat)
            feat = feat.view(B, T, -1)

            if time_mask is not None:
                feat = feat.masked_fill(~time_mask.unsqueeze(-1), 0.0)

        # === Temporal Encoder + Head ===
        if self._finetune_mode:
            # finetune 阶段：detach spatial 梯度，但 temporal+head 可训练
            feat = feat.detach()  # 截断 spatial 梯度
            feat.requires_grad_(True)  # 重新开启梯度

            with torch.amp.autocast('cuda', enabled=False):
                feat = feat.float()
                for layer in self.backbone.temporal_layers:
                    feat = layer(feat)

                skel_out = self.backbone.head(feat)  # (B, T, 51)
        else:
            # warmup 阶段：temporal+head 冻结，用 no_grad 节省显存
            feat = feat.detach()

            with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                feat_tmp = feat.float()
                for layer in self.backbone.temporal_layers:
                    feat_tmp = layer(feat_tmp)

                skel_out = self.backbone.head(feat_tmp)  # (B, T, 51)

            # detach 以断开图，让分类头可以正常反向传播
            feat = feat_tmp.detach()
            skel_out = skel_out.detach()

        # === 骨架归一化 ===
        skel_flat = skel_out  # 保留梯度连接

        skel_3d = skel_flat.view(B, T, self.config.num_joints, 3)
        pelvis = skel_3d[:, :, 4:5, :]
        skel_rel = skel_3d - pelvis
        head_pos = skel_3d[:, :, 0:1, :]
        body_height = (head_pos - pelvis).norm(dim=-1, keepdim=True).clamp(min=0.01)
        skel_normed = skel_rel / body_height
        skel_normed_flat = skel_normed.view(B, T, -1)

        # === 运动速度特征 ===
        # 帧间差分：velocity[:, t] = skel_3d[:, t] - skel_3d[:, t-1]
        velocity = skel_3d[:, 1:, :, :] - skel_3d[:, :-1, :, :]  # (B, T-1, 17, 3)
        velocity = F.pad(velocity, (0, 0, 0, 0, 1, 0))            # (B, T, 17, 3) 首帧补零

        # 盆骨整体运动速度 (3d)，用身高归一化
        pelvis_vel = velocity[:, :, 4, :]                           # (B, T, 3)
        pelvis_vel = pelvis_vel / body_height.squeeze(2)            # (B, T, 3)

        # 膝踝关节 z 轴速度 — 捕捉抬腿幅度 (4d)
        # KNEE_LEFT=12, ANKLE_LEFT=13, KNEE_RIGHT=15, ANKLE_RIGHT=16
        knee_ankle_vel_z = velocity[:, :, [12, 13, 15, 16], 2]     # (B, T, 4)
        knee_ankle_vel_z = knee_ankle_vel_z / body_height.squeeze(2)  # (B, T, 4)

        motion_vel = torch.cat([pelvis_vel, knee_ankle_vel_z], dim=-1)  # (B, T, 7)

        # === 拼接分类头输入 ===
        combined = torch.cat([feat, skel_normed_flat, motion_vel], dim=-1)  # (B, T, 314)

        if time_mask is not None:
            invalid_mask = ~time_mask
            combined = combined.masked_fill(invalid_mask.unsqueeze(-1), 0.0)

        frame_logits = self.classifier(combined)

        skeleton = skel_out.view(B, T, self.config.num_joints, 3)

        return {
            "skeleton": skeleton,
            "action_frame": frame_logits,
        }


# ==========================================
# 4. 加载预训练 backbone
# ==========================================
def load_pretrained_backbone(config: ActionConfig) -> RadarPoseNet:
    """
    根据 config 中的骨架参数构建 PoseConfig，
    加载 RadarPoseNet 并填充预训练权重。
    """
    pose_cfg = PoseConfig()
    # 把 ActionConfig 中与骨架相关的参数复制过去（保证与预训练一致）
    pose_cfg.num_joints = config.num_joints
    pose_cfg.input_channels = config.input_channels
    pose_cfg.points_per_frame = config.points_per_frame
    pose_cfg.dim_model = config.dim_model
    pose_cfg.dim_feedforward = config.dim_feedforward
    pose_cfg.num_heads = config.num_heads
    pose_cfg.num_spatial_layers = config.num_spatial_layers
    pose_cfg.num_temporal_layers = config.num_temporal_layers
    pose_cfg.dropout = config.dropout

    # 构建模型
    model = RadarPoseNet(pose_cfg)

    # 解析路径（相对于本文件所在目录）
    ckpt_path = os.path.join(os.path.dirname(__file__), config.pretrained_path)
    ckpt_path = os.path.normpath(ckpt_path)
    print(f"加载预训练权重：{ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # 兼容两种格式：
    # 1. 直接 state_dict（键为模型参数名）
    # 2. checkpoint 字典（含 model_state_dict / state_dict 等键）
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        state_dict = raw

    # 处理 torch.compile 产生的 _orig_mod. 前缀
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        cleaned[new_key] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [警告] 缺失的键 ({len(missing)} 个)：{missing[:5]} ...")
    if unexpected:
        print(f"  [警告] 多余的键 ({len(unexpected)} 个)：{unexpected[:5]} ...")
    print("  预训练权重加载完成。")

    return model


# ==========================================
# 5. ActionDataset 数据集类
# ==========================================
class ActionDataset(Dataset):
    """
    行为识别数据集（帧级标签版）。
    目录结构：data_root/split/*.npz
    每个 npz 包含：
      - pointcloud:     object array (T,)，每元素为 (N_i, 4) float32
      - skeleton:        (T, 17, 3) float32
      - frame_labels:    (T,) int64，逐帧动作标签
      - majority_label:  int64 标量，多数投票主标签
    其中 T 为可变长度（15-75帧）。
    """
    def __init__(self, data_root: str, config: ActionConfig,
                 split: str = "train", augment: bool = False):
        self.config = config
        self.augment = augment

        search_path = os.path.join(data_root, split, "*.npz")
        self.file_list = glob.glob(search_path)

        if len(self.file_list) == 0:
            print(f"[警告] {search_path} 下未找到任何 .npz 文件！")
        else:
            label_counts = {name: 0 for name in ACTION_NAMES}
            for fp in self.file_list:
                try:
                    with np.load(fp, allow_pickle=True) as data:
                        lbl = int(data["majority_label"])
                    if lbl < NUM_CLASSES:
                        label_counts[ACTION_NAMES[lbl]] += 1
                except Exception:
                    pass
            print(f"[{split}] 共 {len(self.file_list)} 个样本，类别分布：")
            for name, cnt in label_counts.items():
                if cnt > 0:
                    print(f"  {name:12s}: {cnt}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]

        with np.load(path, allow_pickle=True) as data:
            raw_pcs = data["pointcloud"]
            skeleton = data["skeleton"].astype(np.float32)
            frame_labels = data["frame_labels"].astype(np.int64)

        # 逐帧 pad_or_sample_points
        processed_pcs = []
        for pc in raw_pcs:
            pc = np.array(pc, dtype=np.float32)
            pc_fixed = pad_or_sample_points(pc, self.config.points_per_frame)
            processed_pcs.append(pc_fixed)

        pointcloud = np.stack(processed_pcs, axis=0).astype(np.float32)  # (T, N, 4)

        # 点级 padding mask（全零点视为 padding）
        is_padding = np.all(pointcloud[:, :, :3] == 0, axis=2)  # (T, N)

        # 哨兵机制
        for t in range(is_padding.shape[0]):
            if np.all(is_padding[t]):
                is_padding[t, 0] = False
                pointcloud[t, 0, :] = 0.0

        # 数据增强
        if self.augment:
            pointcloud, skeleton = self._apply_augmentation(pointcloud, skeleton)
            pointcloud[is_padding] = 0.0

        return (
            torch.from_numpy(pointcloud),                         # (T, N, 4)
            torch.from_numpy(skeleton),                           # (T, 17, 3)
            torch.from_numpy(is_padding),                         # (T, N)
            torch.from_numpy(frame_labels.copy()),                # (T,)
        )

    def _apply_augmentation(self, pc: np.ndarray, skel: np.ndarray):
        """同步数据增强（与原版完全一致）"""
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            pc[:, :, :3] *= scale
            skel *= scale

        if random.random() < 0.5:
            theta = random.uniform(-np.pi, np.pi)
            c, s = np.cos(theta), np.sin(theta)
            rot_mat = np.array(
                [[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32
            )
            pc[:, :, :3] = np.dot(pc[:, :, :3], rot_mat.T)
            skel = np.dot(skel, rot_mat.T)

        if random.random() < 0.5:
            shift = np.random.uniform(-0.05, 0.05, size=(1, 1, 3)).astype(np.float32)
            shift[:, :, 2] = 0.0
            pc[:, :, :3] += shift
            skel += shift

        if random.random() < 0.5:
            noise = np.random.normal(0, 0.01, pc[:, :, :3].shape).astype(np.float32)
            pc[:, :, :3] += noise

        return pc, skel


# ==========================================
# 6. 工具函数
# ==========================================

def action_collate_fn(batch):
    """
    自定义 collate，处理可变长 T 的 batch padding。

    输入: list of (pointcloud, skeleton, padding_mask, frame_labels)
    输出: (pc_padded, skel_padded, pad_mask_padded, frame_labels_padded, time_mask)

    time_mask: (B, T_max) bool, True=有效帧, False=padding帧
    """
    T_max = max(item[0].shape[0] for item in batch)
    N = batch[0][0].shape[1]  # points_per_frame, 固定

    B = len(batch)
    pc_padded = torch.zeros(B, T_max, N, 4)
    skel_padded = torch.zeros(B, T_max, 17, 3)
    pad_mask_padded = torch.ones(B, T_max, N, dtype=torch.bool)  # 全 True = 全 padding
    frame_labels_padded = torch.zeros(B, T_max, dtype=torch.long)
    time_mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, (pc, skel, pmask, flabels) in enumerate(batch):
        T_i = pc.shape[0]
        pc_padded[i, :T_i] = pc
        skel_padded[i, :T_i] = skel
        pad_mask_padded[i, :T_i] = pmask
        frame_labels_padded[i, :T_i] = flabels
        time_mask[i, :T_i] = True

    return pc_padded, skel_padded, pad_mask_padded, frame_labels_padded, time_mask


def compute_class_weights(dataset, num_classes):
    """根据训练集帧级标签分布计算逆频率类别权重。"""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for i in range(len(dataset)):
        _, _, _, frame_labels = dataset[i]
        counts += torch.bincount(frame_labels, minlength=num_classes)
    total = counts.sum().float()
    weights = total / (num_classes * counts.float().clamp(min=1))
    weights = weights.clamp(max=10.0)
    return weights


def set_seed(seed: int):
    """固定所有随机种子，保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, cls_criterion, optimizer, device, config,
                    skel_criterion=None):
    """训练一个 epoch（帧级分类 + 骨架辅助损失）。"""
    model.train()
    model.backbone.point_emb.eval()
    model.backbone.pos_enc_gen.eval()
    for layer in model.backbone.spatial_layers:
        layer.eval()
    model.backbone.pool.eval()

    # 如果 temporal 冻结（warmup 阶段），也将 temporal+head 设为 eval
    if not model.backbone.temporal_layers[0].qkv.weight.requires_grad:
        for layer in model.backbone.temporal_layers:
            layer.eval()
        model.backbone.head.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_skel_loss = 0.0
    correct = 0
    total = 0

    for batch_pc, batch_skel, batch_mask, batch_flabels, batch_tmask in tqdm(loader, desc="训练"):
        batch_pc     = batch_pc.to(device)
        batch_skel   = batch_skel.to(device)
        batch_mask   = batch_mask.to(device)
        batch_flabels = batch_flabels.to(device)
        batch_tmask  = batch_tmask.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(batch_pc, mask=batch_mask, time_mask=batch_tmask)

            frame_logits = outputs["action_frame"]
            valid_logits = frame_logits[batch_tmask]
            valid_labels = batch_flabels[batch_tmask]
            cls_loss = cls_criterion(valid_logits, valid_labels)

            loss = cls_loss
            skel_loss_val = 0.0
            if skel_criterion is not None:
                pred_skel = outputs["skeleton"]
                valid_pred = pred_skel[batch_tmask]
                valid_gt   = batch_skel[batch_tmask]
                skel_loss = skel_criterion(valid_pred, valid_gt)
                skel_loss_val = skel_loss.item()
                loss = cls_loss + config.skel_aux_weight * skel_loss

        if torch.isnan(loss):
            print("WARNING: NaN loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_skel_loss += skel_loss_val

        pred = valid_logits.argmax(dim=1)
        correct += (pred == valid_labels).sum().item()
        total += valid_labels.size(0)

    n_batches = len(loader)
    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls_loss / n_batches,
        "skel_loss": total_skel_loss / n_batches,
        "acc": correct / total if total > 0 else 0.0,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, skel_criterion=None):
    """验证（帧级分类 + 骨架损失监控）"""
    model.eval()

    total_loss = 0.0
    total_skel_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch_pc, batch_skel, batch_mask, batch_flabels, batch_tmask in tqdm(loader, desc="验证"):
        batch_pc     = batch_pc.to(device)
        batch_skel   = batch_skel.to(device)
        batch_mask   = batch_mask.to(device)
        batch_flabels = batch_flabels.to(device)
        batch_tmask  = batch_tmask.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(batch_pc, mask=batch_mask, time_mask=batch_tmask)

            frame_logits = outputs["action_frame"]
            valid_logits = frame_logits[batch_tmask]
            valid_labels = batch_flabels[batch_tmask]
            loss = criterion(valid_logits, valid_labels)

            # 骨架损失（监控用，不影响模型选择）
            skel_loss_val = 0.0
            if skel_criterion is not None:
                pred_skel = outputs["skeleton"]
                valid_pred = pred_skel[batch_tmask]
                valid_gt   = batch_skel[batch_tmask]
                skel_loss_val = skel_criterion(valid_pred, valid_gt).item()

        total_loss += loss.item()
        total_skel_loss += skel_loss_val

        pred = valid_logits.argmax(dim=1)
        correct += (pred == valid_labels).sum().item()
        total += valid_labels.size(0)

        all_preds.extend(pred.cpu().numpy().tolist())
        all_labels.extend(valid_labels.cpu().numpy().tolist())

    n_batches = len(loader)
    return {
        "loss": total_loss / n_batches,
        "skel_loss": total_skel_loss / n_batches,
        "acc": correct / total if total > 0 else 0.0,
        "preds": all_preds,
        "labels": all_labels,
    }


def print_confusion_matrix(preds, labels, class_names):
    """以文本格式打印混淆矩阵。"""
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)
    for p, l in zip(preds, labels):
        matrix[l][p] += 1

    # 表头
    col_width = max(len(name) for name in class_names) + 2
    header = "真实\\预测".ljust(col_width) + "".join(
        name.rjust(col_width) for name in class_names
    )
    print("\n混淆矩阵：")
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = name.ljust(col_width) + "".join(
            str(matrix[i][j]).rjust(col_width) for j in range(n)
        )
        print(row)
    print()

    # 每类准确率
    print("各类准确率：")
    for i, name in enumerate(class_names):
        total_i = matrix[i].sum()
        acc_i = matrix[i][i] / total_i if total_i > 0 else 0.0
        print(f"  {name:12s}: {acc_i*100:.1f}%  ({matrix[i][i]}/{total_i})")


# ==========================================
# 7. EarlyStopping 类
# ==========================================
class EarlyStopping:
    """
    监控验证集准确率（越大越好）。
    达到 patience 个 epoch 无改善时触发早停。
    仅保存 classifier 的 state_dict。
    """
    def __init__(self, patience: int = 15, verbose: bool = True,
                 delta: float = 0.0, path: str = "best_classifier.pth"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val_acc = 0.0

    def __call__(self, val_acc: float, model: RadarActionNet):
        # val_acc 越大越好
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self._save(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping 计数：{self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_acc, model)
            self.counter = 0

    def _save(self, val_acc: float, model: RadarActionNet):
        """只保存 classifier 部分的权重。"""
        if self.verbose:
            print(f"  验证准确率提升 ({self.best_val_acc*100:.2f}% → {val_acc*100:.2f}%)，保存分类头权重...")
        torch.save(model.classifier.state_dict(), self.path)
        self.best_val_acc = val_acc


# ==========================================
# 8. main() 主训练函数
# ==========================================
def main():
    torch.set_float32_matmul_precision("high")

    config = ActionConfig()
    set_seed(config.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # ---- TensorBoard ----
    run_name = f"action_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    print(f"TensorBoard 日志目录：runs/{run_name}")

    # ---- 数据集 ----
    data_root = os.path.join(script_dir, config.data_root)
    train_set = ActionDataset(data_root, config, split="train", augment=True)
    val_set   = ActionDataset(data_root, config, split="val",   augment=False)
    print(f"数据集：Train={len(train_set)}，Val={len(val_set)}")

    if len(train_set) == 0 or len(val_set) == 0:
        print("ERROR: 数据集为空，请先运行 data_split_action.py 生成数据")
        return

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
        persistent_workers=True, collate_fn=action_collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
        persistent_workers=True, collate_fn=action_collate_fn,
    )

    # ---- 模型 ----
    backbone = load_pretrained_backbone(config)
    model = RadarActionNet(config, backbone, freeze_temporal=True).to(config.device)
    print(f"模型已构建，backbone temporal layers 初始冻结。")

    # ---- 类别加权损失 ----
    print("计算类别权重...")
    class_weights = compute_class_weights(train_set, config.num_classes)
    class_weights = class_weights.to(config.device)
    print(f"类别权重：{dict(zip(ACTION_NAMES, [f'{w:.2f}' for w in class_weights.tolist()]))}")
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    skel_criterion = nn.L1Loss()

    # ---- 优化器（初始只含 classifier 参数） ----
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.warmup_epochs, eta_min=1e-6
    )

    early_stopping = EarlyStopping(
        patience=config.patience, verbose=True,
        path=os.path.join(script_dir, "best_action_model.pth"),
    )

    # 修改 EarlyStopping._save 以保存整个模型（含 temporal layers）
    def _save_full(self_es, val_acc, model_to_save):
        if self_es.verbose:
            print(f"  验证准确率提升 ({self_es.best_val_acc*100:.2f}% → {val_acc*100:.2f}%)，保存模型权重...")
        save_dict = {
            "classifier": model_to_save.classifier.state_dict(),
            "temporal_layers": model_to_save.backbone.temporal_layers.state_dict(),
            "head": model_to_save.backbone.head.state_dict(),
        }
        torch.save(save_dict, self_es.path)
        self_es.best_val_acc = val_acc
    early_stopping._save = lambda val_acc, m: _save_full(early_stopping, val_acc, m)

    print(f"\n开始训练：{config.epochs} epochs（前 {config.warmup_epochs} epochs 为 warmup）\n")

    phase = "warmup"
    for epoch in range(config.epochs):
        # ---- 阶段切换：warmup → finetune ----
        if epoch == config.warmup_epochs and phase == "warmup":
            phase = "finetune"
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}: 解冻 backbone temporal layers，启用差分学习率")
            print(f"{'='*60}\n")

            model.unfreeze_temporal()

            # 重置 EarlyStopping，让 finetune 阶段重新开始
            early_stopping.counter = 0
            early_stopping.best_score = None

            optimizer = torch.optim.AdamW([
                {"params": model.classifier.parameters(), "lr": config.lr},
                {"params": model.backbone.temporal_layers.parameters(), "lr": config.backbone_lr},
                {"params": model.backbone.head.parameters(), "lr": config.backbone_lr},
            ], weight_decay=config.weight_decay)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=1e-6
            )

        # ---- 训练 ----
        train_metrics = train_one_epoch(
            model, train_loader, cls_criterion, optimizer, config.device, config,
            skel_criterion=skel_criterion,
        )

        # ---- 验证 ----
        val_metrics = validate(model, val_loader, cls_criterion, config.device,
                               skel_criterion=skel_criterion)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ---- TensorBoard ----
        writer.add_scalar("Train/Loss", train_metrics["loss"], epoch)
        writer.add_scalar("Train/ClsLoss", train_metrics["cls_loss"], epoch)
        writer.add_scalar("Train/SkelLoss", train_metrics["skel_loss"], epoch)
        writer.add_scalar("Train/Acc",  train_metrics["acc"],  epoch)
        writer.add_scalar("Val/Loss",   val_metrics["loss"],   epoch)
        writer.add_scalar("Val/SkelLoss", val_metrics["skel_loss"], epoch)
        writer.add_scalar("Val/Acc",    val_metrics["acc"],    epoch)
        writer.add_scalar("Misc/LR",    current_lr,            epoch)
        writer.add_scalar("Misc/Phase", 0 if phase == "warmup" else 1, epoch)

        print(
            f"Epoch [{epoch+1:3d}/{config.epochs}] ({phase})  LR: {current_lr:.2e}\n"
            f"  [Train] Loss: {train_metrics['loss']:.4f}  "
            f"Cls: {train_metrics['cls_loss']:.4f}  "
            f"Skel: {train_metrics['skel_loss']:.4f}  "
            f"Acc: {train_metrics['acc']*100:.2f}%\n"
            f"  [Val]   Loss: {val_metrics['loss']:.4f}  "
            f"Skel: {val_metrics['skel_loss']:.4f}  "
            f"Acc: {val_metrics['acc']*100:.2f}%"
        )

        early_stopping(val_metrics["acc"], model)
        if early_stopping.early_stop:
            print("早停触发，结束训练。")
            break

    writer.close()

    # ---- 最终混淆矩阵 ----
    print("\n====== 最终验证集混淆矩阵（Frame 级）======")
    print_confusion_matrix(val_metrics["preds"], val_metrics["labels"], ACTION_NAMES)

    best_path = os.path.join(script_dir, "best_action_model.pth")
    if os.path.exists(best_path):
        print(f"最优模型已保存至：{best_path}")
        print(f"最优验证 Frame 准确率：{early_stopping.best_val_acc*100:.2f}%")


# ==========================================
# 9. smoke_test() 冒烟测试
# ==========================================
def smoke_test():
    """
    用随机数据验证前向传播和训练管线是否正常。
    检查：
    1. 输出形状正确（帧级分类 + 骨架）
    2. freeze/unfreeze 控制正确
    3. 反向传播（分类损失 + 骨架辅助损失）
    4. 含 padding 帧时无 NaN
    """
    print("=" * 60)
    print("冒烟测试开始")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    config = ActionConfig()
    config.device = device

    # 加载 backbone
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.normpath(os.path.join(script_dir, config.pretrained_path))

    if not os.path.exists(ckpt_path):
        print(f"[跳过] 预训练权重不存在：{ckpt_path}")
        print("将使用随机初始化的 backbone 进行结构测试。")
        pose_cfg = PoseConfig()
        backbone = RadarPoseNet(pose_cfg)
    else:
        backbone = load_pretrained_backbone(config)

    B, T, N, C = 2, 10, config.points_per_frame, config.input_channels
    x    = torch.randn(B, T, N, C, device=device)
    mask = torch.zeros(B, T, N, dtype=torch.bool, device=device)
    time_mask = torch.ones(B, T, dtype=torch.bool, device=device)

    # ---- 测试 1: 输出形状（temporal 冻结） ----
    print("\n[测试 1] 帧级分类输出形状（temporal 冻结）")
    model = RadarActionNet(config, backbone, freeze_temporal=True).to(device)
    out = model(x, mask=mask, time_mask=time_mask)
    assert out["skeleton"].shape == (B, T, 17, 3), f"骨架输出形状错误：{out['skeleton'].shape}"
    assert out["action_frame"].shape == (B, T, NUM_CLASSES), f"帧级输出形状错误：{out['action_frame'].shape}"
    print(f"  骨架：{out['skeleton'].shape}  帧级：{out['action_frame'].shape}  OK")

    # ---- 测试 2: freeze/unfreeze 梯度控制 ----
    print("\n[测试 2] freeze/unfreeze 梯度控制")
    spatial_grads = [p.requires_grad for p in model.backbone.spatial_layers.parameters()]
    temporal_grads = [p.requires_grad for p in model.backbone.temporal_layers.parameters()]
    cls_grads = [p.requires_grad for p in model.classifier.parameters()]
    assert not any(spatial_grads), "spatial layers 应无梯度！"
    assert not any(temporal_grads), "temporal layers 初始应无梯度！"
    assert all(cls_grads), "classifier 应有梯度！"
    print("  冻结状态: spatial=frozen, temporal=frozen, classifier=trainable  OK")

    model.unfreeze_temporal()
    temporal_grads_after = [p.requires_grad for p in model.backbone.temporal_layers.parameters()]
    head_grads_after = [p.requires_grad for p in model.backbone.head.parameters()]
    assert all(temporal_grads_after), "解冻后 temporal layers 应有梯度！"
    assert all(head_grads_after), "解冻后 head 应有梯度！"
    print("  解冻后: temporal=trainable, head=trainable  OK")

    model.freeze_temporal()
    temporal_grads_refrozen = [p.requires_grad for p in model.backbone.temporal_layers.parameters()]
    assert not any(temporal_grads_refrozen), "重新冻结后 temporal layers 应无梯度！"
    print("  重新冻结: temporal=frozen  OK")

    # ---- 测试 3: 反向传播（分类 + 骨架辅助损失） ----
    print("\n[测试 3] 反向传播（含骨架辅助损失）")
    model_bp = RadarActionNet(config, backbone, freeze_temporal=False).to(device)
    frame_labels = torch.randint(0, NUM_CLASSES, (B, T), device=device)
    gt_skel = torch.randn(B, T, 17, 3, device=device)

    out_bp = model_bp(x, mask=mask, time_mask=time_mask)
    cls_loss = nn.CrossEntropyLoss()(
        out_bp["action_frame"].view(-1, NUM_CLASSES), frame_labels.view(-1)
    )
    skel_loss = nn.L1Loss()(
        out_bp["skeleton"][time_mask], gt_skel[time_mask]
    )
    total_loss = cls_loss + config.skel_aux_weight * skel_loss
    total_loss.backward()

    cls_has_grad = any(p.grad is not None for p in model_bp.classifier.parameters())
    temp_has_grad = any(p.grad is not None for p in model_bp.backbone.temporal_layers.parameters())
    assert cls_has_grad, "classifier 应产生梯度！"
    assert temp_has_grad, "temporal layers 应产生梯度！"
    print(f"  cls_loss={cls_loss.item():.4f}  skel_loss={skel_loss.item():.4f}  梯度正常  OK")

    # ---- 测试 4: 含 padding 帧的 NaN 回归测试 ----
    print("\n[测试 4] 可变长 padding NaN 回归测试")
    model_nan = RadarActionNet(config, backbone, freeze_temporal=True).to(device)
    model_nan.train()
    model_nan.backbone.eval()

    B2, T2 = 2, 10
    x_var = torch.randn(B2, T2, N, C, device=device)
    mask_var = torch.zeros(B2, T2, N, dtype=torch.bool, device=device)
    tm_var = torch.ones(B2, T2, dtype=torch.bool, device=device)
    x_var[1, 5:] = 0.0
    mask_var[1, 5:] = True
    tm_var[1, 5:] = False

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out_nan = model_nan(x_var, mask=mask_var, time_mask=tm_var)
        frm_l = out_nan["action_frame"]

    assert not torch.isnan(frm_l).any(), f"frame logits 含 NaN！{frm_l}"
    print(f"  frame logits 无 NaN  OK")

    print("\n" + "=" * 60)
    print("冒烟测试全部通过！")
    print("=" * 60)


# ==========================================
# 10. 入口
# ==========================================
if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--smoke":
        smoke_test()
    else:
        main()
