import os
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
# import math

from torch.utils.tensorboard import SummaryWriter
import pandas as pd # 用 pandas 保存 csv 最方便
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from io import BytesIO

import torch.utils.checkpoint as cp

# 定义关节名称 (对应你之前的 0-16 索引)
JOINT_NAMES = [
    "Head", "Neck", "Chest", "Navel", "Pelvis",     # 0-4: Torso
    "L_Shoulder", "L_Elbow", "L_Wrist",             # 5-7: Left Arm
    "R_Shoulder", "R_Elbow", "R_Wrist",             # 8-10: Right Arm
    "L_Hip", "L_Knee", "L_Ankle",                   # 11-13: Left Leg
    "R_Hip", "R_Knee", "R_Ankle"                    # 14-16: Right Leg
]

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
class Config:
    # ==========================
    # 1. 缺失的基础定义 (必须补上)
    # ==========================
    data_root = "../dataset_split_firstpoint"     # [缺] 数据集路径，Dataset 类需要
    num_joints = 17                 # [缺] 骨架关键点数量，Model 输出头需要 (17*3)
    input_channels = 4              # [缺] 输入特征维度 (x,y,z,v)，Model 输入层需要
    device = "cuda"                 # [缺] 设备定义，main 函数需要
    seed = 2026                       # [缺] 随机种子，保证结果可复现


    use_preloading = False

    # --- 数据相关 ---
    # 4090D 显存巨大，建议保持 256 点，甚至可以尝试 512 点追求极致精度
    # 但 256 对人体姿态来说通常已经是“信息饱和”了
    points_per_frame = 120
    
    # --- 模型相关 (保持高性能配置) ---
    dim_model = 256
    dim_feedforward = 1024
    num_heads = 4
    num_spatial_layers = 3
    num_temporal_layers = 8
    dropout = 0.2  # Batch大时，适当增加Dropout防止过拟合
    
    # --- 训练核心参数 (关键修改) ---
    # 1. Batch Size: 
    # 对于 62K 的小文件和 Transformer，32 太小了，显卡会“饿死”。
    # 建议起步 64，推荐 128。如果开了混合精度(AMP)，甚至可以冲 256。
    batch_size = 96 
    
    # 2. Learning Rate:
    # Batch 变大后，LR 需要对应调大，否则收敛太慢。
    # 配合 128 的 Batch Size，建议用 5e-4 或 8e-4。
    lr = 6e-4
    
    # 3. Epochs:
    # Batch 大了，每个 Epoch 的迭代次数就变少了，所以总 Epoch 要增加。
    epochs = 500

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.seed)

# 设置 CUDA 内存分配以减少碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# ==========================================
# 2. 数据集与增强 (Dataset & Augmentation)
# ==========================================

def pad_or_sample_points(points, num_points):
    """
    处理点云数量不一致问题：
    - 点数 > num_points: 随机采样
    - 点数 < num_points: 补零
    """
    N = points.shape[0]
    if N >= num_points:
        choice = np.random.choice(N, num_points, replace=False)
        return points[choice, :]
    else:
        if N == 0:
            return np.zeros((num_points, points.shape[1]), dtype=np.float32)
        # 补零策略：可以用重复采样，也可以补纯0
        # 这里采用补0策略，模型需要能学会忽略0点
        padding = np.zeros((num_points - N, points.shape[1]), dtype=np.float32)
        return np.vstack([points, padding])


class RadarPoseDataset(Dataset):
    def __init__(self, data_root, config, split="train", augment=False):
        self.config = config
        self.augment = augment
        self.split = split
        
        # 获取文件列表
        search_path = os.path.join(data_root, split, "*.npz")
        self.file_list = glob.glob(search_path)
        
        if len(self.file_list) == 0:
            print(f"⚠️ Warning: No .npz files found in {search_path}")

        # 模式选择
        self.use_preloading = config.use_preloading
        self.data_cache = []

        if self.use_preloading:
            print(f"🚀 [RAM Cache] 正在预加载 {len(self.file_list)} 个样本到内存...")
            for path in tqdm(self.file_list, desc=f"Loading {split}"):
                # 调用复用的处理函数
                sample = self._process_one_file(path)
                self.data_cache.append(sample)
            print(f"✅ 加载完成！内存已占用。")
        else:
            print(f"🐢 [Lazy Loading] 懒加载模式已开启。将在训练时实时读取硬盘。")

    def _pad_and_create_mask(self, points, max_points):
        """
        【新增核心函数】
        同时处理点云采样/补零，并基于'计数'生成Mask，而不是基于'数值'。
        """
        num_points = points.shape[0]
        
        # 1. 准备容器
        # Mask: True 代表 Padding (无效/忽略), False 代表真实点 (有效)
        # 初始化全为 True (假设全是 Padding)
        mask = np.ones(max_points, dtype=bool) 
        out_points = np.zeros((max_points, 4), dtype=np.float32)

        # 2. 分情况处理
        if num_points == 0:
            # --- 哨兵策略 ---
            # 如果这一帧完全是空的，为了防止 Transformer Attention 计算 NaN，
            # 我们强制把第 0 个点标记为"有效"。虽然它是 (0,0,0,0)，但模型必须得看点什么东西。
            mask[0] = False
            # out_points 已经是全0了，不用动
            
        elif num_points >= max_points:
            # --- 采样 ---
            # 随机选择 max_points 个索引
            choice = np.random.choice(num_points, max_points, replace=False)
            out_points[:] = points[choice]
            # 既然填满了，那就全是有效的
            mask[:] = False 
            
        else:
            # --- 补零 ---
            # 填入前 num_points 个
            out_points[:num_points] = points
            # 【核心修改】只把前 num_points 个标记为有效
            # 这样即使 points[0] 是 (0,0,0,0)，因为 mask[0] 是 False，模型依然会认为它是有效数据！
            mask[:num_points] = False
            
        return out_points, mask

    def _process_one_file(self, path):
        """
        核心辅助函数：读取单个文件并进行预处理。
        """
        try:
            with np.load(path, allow_pickle=True) as data:
                raw_pcs = data['pointcloud'] # Object array (T, object)
                skeleton = data['skeleton'].astype(np.float32) # (T, 17, 3)
            
            # 2. 预处理点云 (循环处理每一帧)
            processed_pcs = []
            processed_masks = []
            
            for pc in raw_pcs:
                # 【修改】调用内部的新函数，同时获取点云和Mask
                pc_fixed, mask_fixed = self._pad_and_create_mask(pc, self.config.points_per_frame)
                processed_pcs.append(pc_fixed)
                processed_masks.append(mask_fixed)
            
            # 堆叠
            pointcloud = np.stack(processed_pcs, axis=0).astype(np.float32) # (T, N, 4)
            mask = np.stack(processed_masks, axis=0) # (T, N) Bool
            
            # 注意：这里不再需要 np.all(xyz==0) 这种危险的逻辑了
            
            return {
                'pointcloud': pointcloud, 
                'skeleton': skeleton,    
                'mask': mask      
            }
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # 异常处理：返回 Dummy 数据
            T = 75 
            N = self.config.points_per_frame
            # 即使是 Dummy 数据，也要遵循哨兵原则，让 mask[0] = False
            dummy_mask = np.ones((T, N), dtype=bool)
            dummy_mask[:, 0] = False 
            
            return {
                'pointcloud': np.zeros((T, N, 4), dtype=np.float32),
                'skeleton': np.zeros((T, 17, 3), dtype=np.float32),
                'mask': dummy_mask
            }

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 1. 获取基础数据
        if self.use_preloading:
            cache_item = self.data_cache[idx]
            pointcloud = cache_item['pointcloud'].copy()
            skeleton = cache_item['skeleton'].copy()
            mask = cache_item['mask'].copy()
        else:
            path = self.file_list[idx]
            item = self._process_one_file(path)
            pointcloud = item['pointcloud']
            skeleton = item['skeleton']
            mask = item['mask']

        # 2. 数据增强
        if self.augment:
            pointcloud, skeleton = self.apply_augmentation(pointcloud, skeleton)
            
            # 【重要】再次清理 Padding 区域
            # 因为 apply_augmentation 里的 Jitter (加噪声) 可能会把 Padding 区的 (0,0,0) 变成 (0.001, -0.002, ...)
            # 虽然 Mask 已经把它们标记为无效了，但为了保持数据整洁，建议把它们再次置零。
            # 利用 Mask 索引，把 True (无效) 的位置全部设为 0
            pointcloud[mask] = 0 

        # 3. 转 Tensor
        pc_tensor = torch.from_numpy(pointcloud)
        skel_tensor = torch.from_numpy(skeleton)
        mask_tensor = torch.from_numpy(mask)

        # ==========================================
        # 【核心修改】时间逆序 (Time Reversal)
        # ==========================================
        # 假设数据维度是 [Time, Points, Channels] 或 [Time, Joints, 3]
        # 在第 0 维 (Time) 上进行翻转
        # 效果：原来的 Frame 75 变成了现在的 Frame 0
        # pc_tensor = torch.flip(pc_tensor, dims=[0])
        # skel_tensor = torch.flip(skel_tensor, dims=[0])
        # mask_tensor = torch.flip(mask_tensor, dims=[0])

        # 返回翻转后的数据
        return pc_tensor, skel_tensor, mask_tensor
        # 3. 转 Tensor 并返回
        # return (
        #     torch.from_numpy(pointcloud), 
        #     torch.from_numpy(skeleton), 
        #     torch.from_numpy(mask)
        # )
    
    def apply_augmentation(self, pc, skel):
        """
        同步对点云和骨架进行增强
        """
        # 1. 随机缩放
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            pc[:, :, :3] *= scale
            skel *= scale

        # 2. 随机旋转
        if random.random() < 0.5:
            theta = random.uniform(-np.pi, np.pi) 
            c, s = np.cos(theta), np.sin(theta)
            rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            
            pc[:, :, :3] = np.dot(pc[:, :, :3], rot_mat.T)
            skel = np.dot(skel, rot_mat.T)

        # 3. 随机平移
        if random.random() < 0.5:
            shift = np.random.uniform(-0.05, 0.05, size=(1, 1, 3)).astype(np.float32)
            shift[:, :, 2] = 0 
            
            pc[:, :, :3] += shift
            skel += shift 

        # 4. 随机抖动 (Jitter)
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.01, pc[:, :, :3].shape)
            pc[:, :, :3] += noise

        return pc, skel


# ==========================================
# 3. 模型组件 (Model Components) - 升级版
# ==========================================

# --- A. 空间位置编码 (Spatial Positional Encoding) ---
class SpatialPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, dim), # 只输入 x, y, z
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, xyz):
        # xyz: [B*T, N, 3]
        return self.mlp(xyz)

# --- B. 空间注意力池化 (Attentive Pooling) ---
# 替代 Max Pooling，更适合回归任务
class LearnableQueryPooling(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        # 1. 定义一个可学习的 Query Token (类似于 BERT 的 [CLS])
        # 形状: [1, 1, Dim]
        self.query_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 2. 使用 Cross-Attention
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, key_padding_mask=None):
        # x: [Batch, Points, Dim] (这是 Key 和 Value)
        # key_padding_mask: [Batch, Points] (True表示是Padding)
        
        B = x.shape[0]
        
        # 1. 扩展 Query 以匹配 Batch Size
        # query: [Batch, 1, Dim]
        query = self.query_token.expand(B, -1, -1)
        
        # 2. Cross Attention
        # Query找Key/Value要信息
        # 输出: [Batch, 1, Dim]
        out, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        
        # 3. 残差 + Norm (可选，但在 Pooling 层通常直接输出即可，或者加个Norm)
        out = self.norm(out + query) # 这里加残差意味着保留 Query 的初始先验
        
        # 4. 挤压维度: [Batch, 1, Dim] -> [Batch, Dim]
        return out.squeeze(1)

# --- C. RoPE (保持不变) ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None):
        if seq_len is None: seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, None, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, pos):
    return (x * pos.cos()) + (rotate_half(x) * pos.sin())

# --- D. Spatial Encoder Block (带位置编码注入) ---
class PointTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, pos_emb, key_padding_mask=None):
        residual = x
        x = self.norm1(x)
        q = k = x + pos_emb
        v = x
        
        # 【修改】传入 mask
        feat, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        
        x = x + feat # Post-Norm 结构，也可以改成 Pre-Norm
        x = self.norm1(x)
        
        residual = x
        x = self.mlp(x)
        x = x + residual
        x = self.norm2(x)
        return x

# --- E. Temporal Encoder (保持不变) ---
class TemporalTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        pos_emb = self.rope(q, seq_len=T)
        q = apply_rotary_pos_emb(q, pos_emb)
        k = apply_rotary_pos_emb(k, pos_emb)
        
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        
        # x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(B, T, C) # <--- 加上这一行！
        x = self.out_proj(x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x

# --- Main Network (集成升级) ---
class RadarPoseNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Input Embedding
        self.point_emb = nn.Sequential(
            nn.Linear(config.input_channels, config.dim_model), # 4 -> 256
            nn.LayerNorm(config.dim_model),
            nn.GELU()
        )
        
        # 2. 显式的位置编码生成器
        self.pos_enc_gen = SpatialPositionalEncoding(config.dim_model)
        
        # 3. Spatial Encoder
        self.spatial_layers = nn.ModuleList([
            PointTransformerLayer(config.dim_model, config.num_heads)
            for _ in range(config.num_spatial_layers)
        ])
        
        # 4. Query Pooling
        self.pool = LearnableQueryPooling(config.dim_model, config.num_heads)
        
        # 5. Temporal Encoder
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerLayer(config.dim_model, config.num_heads, config.dropout)
            for _ in range(config.num_temporal_layers)
        ])
        
        # 6. Regressor Head
        self.head = nn.Sequential(
            nn.LayerNorm(config.dim_model), # 先归一化，很关键
            
            # 第一层：特征整理
            nn.Linear(config.dim_model, config.dim_model), 
            nn.GELU(), # 引入非线性，这很关键！
            
            # 第二层：输出坐标
            nn.Linear(config.dim_model, config.num_joints * 3)
        )

    def forward(self, x, mask=None):
        # x: [Batch, Time, Points, Channels] (Channels=4: x,y,z,v)
        B, T, N, C = x.shape
        
        # Flatten Batch and Time
        x = x.view(B * T, N, C)
        mask_flat = mask.view(B * T, N) # 展平 mask 以匹配 spatial 维度
        
        # 分离几何坐标 (xyz) 和特征
        xyz = x[:, :, :3] # [B*T, N, 3] 用于生成位置编码
        
        # Embedding
        feat = self.point_emb(x) # [B*T, N, Dim]
        
        # 生成位置编码
        pos_emb = self.pos_enc_gen(xyz) # [B*T, N, Dim]
        
        # Spatial Transformer (带位置注入)
        for layer in self.spatial_layers:
            feat = cp.checkpoint(layer, feat, pos_emb, mask_flat, use_reentrant=False)
            # feat = layer(feat, pos_emb, mask_flat)
            
        # Pooling: [B*T, N, Dim] -> [B*T, Dim]
        # 使用 query Pooling 自动学习重点区域的加权平均
        feat = self.pool(feat, key_padding_mask=mask_flat) # [B*T, Dim]
        
        # Temporal Processing
        feat = feat.view(B, T, -1) # [B, T, Dim]
        for layer in self.temporal_layers:
            feat = cp.checkpoint(layer,feat, use_reentrant=False)
            # feat = layer(feat)
            
        # Decoding
        out = self.head(feat)
        out = out.view(B, T, self.config.num_joints, 3)
        
        return out


class PoseLoss(nn.Module):
    def __init__(self, 
                 # --- 1. 基础 Loss 权重 ---
                 w_pos=1.0,         
                 w_root_vel=5.0,    
                 w_local_vel=10.0,  
                 w_bone=2.0,        
                 
                 # --- 2. 部位精细化权重 ---
                 w_torso=1.0,       
                 w_proximal=2.0,    
                 w_middle=4.0,      
                 w_distal=8.0,
                 
                 # --- 3. 【新增】实时模式：时间边缘权重 ---
                 # 重点关注最后几帧，因为实时系统输出的是序列末尾
                 w_temporal_edge=5,  # 最后一帧的权重倍率 (建议 3.0 ~ 5.0)
                 edge_len=10           # 重点关注最后多少帧 (建议 5 ~ 10)
                 ):
        super().__init__()
        
        self.weights = {
            'pos': w_pos,
            'root_vel': w_root_vel,
            'local_vel': w_local_vel,
            'bone': w_bone
        }
        
        # 实时相关参数
        self.w_temporal_edge = w_temporal_edge
        self.edge_len = edge_len
        
        # 全部使用 reduction='none'，因为我们需要先加权(部位+时间)，最后再求 Mean
        self.l1 = nn.L1Loss(reduction='none') 

        # ==========================================
        # A. 定义部位索引
        # ==========================================
        idx_torso = [0, 1, 2, 3, 4]
        idx_proximal = [5, 8, 11, 14]
        idx_middle = [6, 9, 12, 15]
        idx_distal = [7, 10, 13, 16]
        
        # 构建部位权重向量 [1, 1, 17, 1]
        part_weight_tensor = torch.ones(17)
        part_weight_tensor[idx_torso] = w_torso
        part_weight_tensor[idx_proximal] = w_proximal
        part_weight_tensor[idx_middle] = w_middle
        part_weight_tensor[idx_distal] = w_distal
        self.register_buffer('part_weights', part_weight_tensor.view(1, 1, 17, 1))

        # ==========================================
        # B. 骨架连接定义
        # ==========================================
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),     # Torso
            (1, 5), (5, 6), (6, 7),             # L_Arm
            (1, 8), (8, 9), (9, 10),            # R_Arm
            (4, 11), (11, 12), (12, 13),        # L_Leg
            (4, 14), (14, 15), (15, 16)         # R_Leg
        ]
        u_indices = [e[0] for e in edges]
        v_indices = [e[1] for e in edges]
        self.register_buffer('bone_u', torch.tensor(u_indices, dtype=torch.long))
        self.register_buffer('bone_v', torch.tensor(v_indices, dtype=torch.long))

    def _get_temporal_weights(self, T, device):
        """
        生成 J 型时间权重向量: [1, T, 1, 1]
        前面是 1.0, 最后 edge_len 帧线性增加到 w_temporal_edge
        """
        weights = torch.ones(T, device=device)
        if T > self.edge_len:
            # 生成一个从 1.0 到 5.0 的上升坡
            ramp = torch.linspace(1.0, self.w_temporal_edge, self.edge_len, device=device)
            weights[-self.edge_len:] = ramp
        return weights.view(1, T, 1, 1)

    def forward(self, pred, gt):
        """
        pred, gt: [Batch, Time, 17, 3]
        """
        B, T, J, C = pred.shape
        device = pred.device
        
        # 1. 动态生成时间权重 (适配不同的 T)
        # shape: [1, T, 1, 1]
        t_weights = self._get_temporal_weights(T, device)
        
        # ------------------------------------------
        # 1. 动态加权的位置损失 (Weighted Position Loss)
        # ------------------------------------------
        # 维度: [B, T, 17, 3]
        raw_pos_loss = self.l1(pred, gt) 
        
        # 同时应用：部位加权 * 时间加权
        # 这样最后一帧的手腕误差会被放大 (w_distal * w_edge) 倍！
        weighted_pos_loss = (raw_pos_loss * self.part_weights * t_weights).mean()
        
        
        # ------------------------------------------
        # 2. 解耦速度损失 (Decoupled Velocity Loss)
        # ------------------------------------------
        root_idx = 4 
        
        # 准备数据
        pred_root = pred[:, :, root_idx:root_idx+1, :] 
        gt_root = gt[:, :, root_idx:root_idx+1, :]
        
        pred_local = pred - pred_root 
        gt_local = gt - gt_root
        
        # 差分计算速度 (Time - 1)
        pred_root_vel = pred_root[:, 1:] - pred_root[:, :-1]
        gt_root_vel = gt_root[:, 1:] - gt_root[:, :-1]
        
        pred_local_vel = pred_local[:, 1:] - pred_local[:, :-1]
        gt_local_vel = gt_local[:, 1:] - gt_local[:, :-1]
        
        # 获取对应速度的时间权重 (少一帧，取后 T-1 个权重，或者重新生成)
        # 这里直接取 t_weights 的后 T-1 部分，意味着我们也重点关注最后几帧的速度
        t_weights_vel = t_weights[:, 1:, :, :]
        
        # A. Root Vel (应用时间加权)
        raw_root_vel_loss = self.l1(pred_root_vel, gt_root_vel)
        loss_root_vel = (raw_root_vel_loss * t_weights_vel).mean()
        
        # B. Local Vel (应用部位加权 + 时间加权)
        raw_local_vel_loss = self.l1(pred_local_vel, gt_local_vel)
        weighted_local_vel_loss = (raw_local_vel_loss * self.part_weights * t_weights_vel).mean()
        
        
        # ------------------------------------------
        # 3. 骨长一致性损失 (Bone Loss)
        # ------------------------------------------
        pred_u, pred_v = pred[:, :, self.bone_u], pred[:, :, self.bone_v]
        gt_u, gt_v = gt[:, :, self.bone_u], gt[:, :, self.bone_v]
        
        # [B, T, Num_Bones]
        pred_bone_len = torch.norm(pred_u - pred_v + 1e-8, dim=-1)
        gt_bone_len = torch.norm(gt_u - gt_v + 1e-8, dim=-1)
        
        # 扩展维度以便应用时间权重: [B, T, Num_Bones] -> [B, T, Num_Bones, 1]
        raw_bone_loss = self.l1(pred_bone_len.unsqueeze(-1), gt_bone_len.unsqueeze(-1))
        
        # 应用时间加权 (骨长在最后一帧也不能畸变)
        loss_bone = (raw_bone_loss * t_weights).mean()
        
        
        # ------------------------------------------
        # 4. 总 Loss 聚合
        # ------------------------------------------
        total_loss = (
            self.weights['pos'] * weighted_pos_loss +
            self.weights['root_vel'] * loss_root_vel +
            self.weights['local_vel'] * weighted_local_vel_loss + 
            self.weights['bone'] * loss_bone
        )
        
        return total_loss, weighted_pos_loss, loss_root_vel, weighted_local_vel_loss, loss_bone


# --- 在 train_one_epoch 函数中 ---
def train_one_epoch(model, loader, criterion, optimizer, device): 
    model.train()
    
    # 初始化一个字典来记录各项 Loss 的累计值
    loss_meters = {
        'total': 0.0,
        'pos': 0.0,
        'root': 0.0,
        'local': 0.0,
        'bone': 0.0
    }
    
    for batch_pc, batch_skel, batch_mask in tqdm(loader, desc="Training"):
        # 1. 数据送入 GPU
        batch_pc = batch_pc.to(device)
        batch_skel = batch_skel.to(device)
        batch_mask = batch_mask.to(device)
        
        optimizer.zero_grad()
        
        # 2. 混合精度上下文
        with torch.amp.autocast('cuda', dtype=torch.bfloat16): 
            # 传入 mask (如果有补零点，Transformer 需要 mask 掉)
            pred_skel = model(batch_pc, mask=batch_mask) 
            if torch.isnan(pred_skel).any():
                print(f"💥 严重警告：模型在第 {i} 个 Batch 输出了 NaN！(可能是权重已损坏或梯度爆炸)")
                break
            # 【关键修改】：必须接住 5 个返回值
            # total_loss 用于反向传播，其他的用于记录日志
            loss, loss_pos, loss_root, loss_local, loss_bone = criterion(pred_skel, batch_skel)
        
        # 3. 反向传播 (只对 total_loss 进行)
        # scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # scaler.step(optimizer)
        # scaler.update()
        # 直接反向传播 (不要 Scaler)
        loss.backward()
        
        # 梯度裁剪 (保留这个好习惯)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 4. 累计各项 Loss (使用 .item() 转换为 python float，防止显存泄漏)
        loss_meters['total'] += loss.item()
        loss_meters['pos'] += loss_pos.item()
        loss_meters['root'] += loss_root.item()
        loss_meters['local'] += loss_local.item()
        loss_meters['bone'] += loss_bone.item()
        
    # 计算平均值
    num_batches = len(loader)
    avg_losses = {k: v / num_batches for k, v in loss_meters.items()}
    
    # 返回一个字典，包含所有 loss 的平均值
    return avg_losses

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    
    # 1. 初始化 Loss 累计器
    loss_meters = {
        'total': 0.0, 'pos': 0.0, 'root': 0.0, 'local': 0.0, 'bone': 0.0
    }
    
    total_mpjpe = 0 
    total_mpjpe_per_joint = torch.zeros(17, device=device) 
    
    # --- 【新增】特定帧的误差累计器 ---
    # 目标帧: 1, 20, 40, 50, 75 (对应索引 0, 19, 39, 49, 74)
    target_indices = torch.tensor([0, 30, 50, 70, 72, 74], device=device, dtype=torch.long)
    #用于存储这5个时间点的累计误差 [5]
    total_mpjpe_at_frames = torch.zeros(len(target_indices), device=device) 

    num_total_sequences = 0 # 记录总样本数 (Batch总和)
    num_total_frames = 0    # 记录总帧数 (Batch * Time)
    
    for batch_pc, batch_skel, batch_mask in loader:
        batch_pc = batch_pc.to(device)
        batch_skel = batch_skel.to(device)
        batch_mask = batch_mask.to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred_skel = model(batch_pc, mask=batch_mask)
            loss, loss_pos, loss_root, loss_local, loss_bone = criterion(pred_skel, batch_skel)
        
        # 2. 累计 Loss
        loss_meters['total'] += loss.item()
        loss_meters['pos'] += loss_pos.item()
        loss_meters['root'] += loss_root.item()
        loss_meters['local'] += loss_local.item()
        loss_meters['bone'] += loss_bone.item()
        
        # --- 3. MPJPE 计算 ---
        # diff: [B, T, 17, 3]
        diff = (pred_skel - batch_skel).float()
        # error_per_joint: [B, T, 17] (每个关节的误差 mm)
        error_per_joint = torch.norm(diff, dim=-1) * 1000.0
        
        # A. 总平均 MPJPE
        total_mpjpe += error_per_joint.mean().item() # 这里的 mean 是对 B*T*17 求均值，作为粗略参考
        
        # B. 这里的逻辑稍作修正，为了更精确，建议按样本数累加
        # error_per_joint.sum() / (17 * T) 会得到该 Batch 的平均 MPJPE
        # 这里为了保持和你原来逻辑一致，我们只在最后做一次除法
        
        # C. Per Joint MPJPE
        total_mpjpe_per_joint += error_per_joint.sum(dim=(0, 1))
        
        # --- 【新增】计算特定帧的 MPJPE ---
        # error_per_frame: [B, T] (先对关节取平均)
        error_per_frame = error_per_joint.mean(dim=-1) 
        
        # 取出指定帧的误差: [B, 5]
        # 注意：如果 T < 75，这里可能会报错，确保 window >= 75
        current_frames_error = error_per_frame[:, target_indices]
        
        # 累加到总器: [5]
        total_mpjpe_at_frames += current_frames_error.sum(dim=0)

        # 统计数量
        B, T = batch_skel.shape[:2]
        num_total_sequences += B
        num_total_frames += (B * T)
        
    # 4. 计算平均值
    avg_losses = {k: v / len(loader) for k, v in loss_meters.items()}
    
    # 修正总 MPJPE 计算：总误差和 / 总帧数 / 17关节
    # (原代码 total_mpjpe += mean().item() 其实是累加了 mean，最后除以 len(loader) 是对的)
    avg_mpjpe = total_mpjpe / len(loader)
    
    # Per Joint: 总和 / 总样本数(B*T)
    avg_mpjpe_per_joint = total_mpjpe_per_joint / num_total_frames
    
    # 【新增】特定帧: 总和 / 总序列数(B)
    # 因为我们是对 Time 维度切片，所以分母是"有多少个视频序列"
    avg_mpjpe_at_frames = total_mpjpe_at_frames / num_total_sequences
    
    return avg_losses, avg_mpjpe, avg_mpjpe_per_joint.cpu().numpy(), avg_mpjpe_at_frames.cpu().numpy()


def visualize_prediction(model, loader, device, writer, epoch):
    """从验证集取第一个 batch，画出第一张帧的 GT 与 Pred 的 3D 散点图并写入 TensorBoard。"""
    model.eval()
    with torch.no_grad():
        for batch_pc, batch_skel, batch_mask in loader:
            batch_pc = batch_pc.to(device)
            batch_mask = batch_mask.to(device)
            pred = model(batch_pc, mask=batch_mask) # [B, T, J, 3]

            pred_np = pred[0, 0].cpu().numpy()
            gt_np = batch_skel[0, 0].cpu().numpy()

            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(gt_np[:,0], gt_np[:,1], gt_np[:,2], c='green', label='GT')
            ax.scatter(pred_np[:,0], pred_np[:,1], pred_np[:,2], c='red', marker='^', label='Pred')
            ax.legend()
            ax.set_title(f'Epoch {epoch} sample')

            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            img = np.array(img)
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            writer.add_image('Prediction/sample', img, epoch, dataformats='CHW')
            break


class EarlyStopping:
    def __init__(self, patience=90, verbose=False, delta=0, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    # 【修改】增加 optimizer, scheduler, epoch 参数
    def __call__(self, val_metric, model, optimizer, scheduler, epoch):
        score = -val_metric # MPJPE 越小越好，取负

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model, optimizer, scheduler, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model, optimizer, scheduler, epoch)
            self.counter = 0

    # 【修改】保存完整字典
    def save_checkpoint(self, val_loss, model, optimizer, scheduler, epoch):
        if self.verbose:
            print(f'Validation metric improved ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # 构造完整检查点字典
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_score': self.best_score,
            'val_loss_min': val_loss
        }
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss


def main():
    config = Config()
    

    # ==========================
    # 【新增】断点续训配置
    # ==========================
    # 如果想从头练，设为 None。如果想接着练，填入 'best_model.pth' 或具体的 epoch 文件
    resume_path = "best_model.pth"  
    # resume_path = None

    # ==========================================
    # 0. 初始化可视化工具 (TensorBoard & CSV)
    # ==========================================
    run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    csv_path = f"logs/{run_name}_joints.csv"
    os.makedirs("logs", exist_ok=True)
    

    with open(csv_path, 'w') as f:
        f.write("Epoch," + ",".join(JOINT_NAMES) + "\n")

    # ==========================================
    # 1. 数据准备 (RAM Cache 模式)
    # ==========================================
    # augment=True/False 必须在 Dataset __init__ 中处理
    train_set = RadarPoseDataset(config.data_root, config=config, split='train', augment=True)
    val_set = RadarPoseDataset(config.data_root, config=config, split='val', augment=False)
    
    print(f"Dataset Loaded into RAM: Train={len(train_set)}, Val={len(val_set)}")

    # --- DataLoader 设置 (针对 100G 内存优化) ---
    train_loader = DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=True, 
        # 建议开启多进程处理数据增强，4090太快了，CPU单核可能跟不上
        num_workers=8,        
        pin_memory=True,
        persistent_workers=True # 开启这个避免每个epoch重新初始化worker
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4, # 验证集也可以开一点
        pin_memory=True,
        persistent_workers=True
    )
    
    # ==========================================
    # 2. 模型与优化器
    # ==========================================
    model = RadarPoseNet(config).to(config.device)
    
    # --- PyTorch 2.0 编译 (4090D 必开) ---
    print("Compiling model...")
    try:
        model = torch.compile(model) 
    except Exception as e:
        print(f"Warning: torch.compile failed ({e}), running in eager mode.")

    # --- Loss 设置 (使用黄金均衡版权重) ---
    criterion = PoseLoss().to(config.device)     # Loss 里的 buffer 也要上 GPU

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-2)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # --- 混合精度 Scaler ---
    # scaler = torch.amp.GradScaler('cuda')

    # ==========================
    # 【新增】加载检查点逻辑
    # ==========================
    start_epoch = 0
    best_val_mpjpe = np.inf # 用于恢复 EarlyStopping 的状态

    if resume_path and os.path.exists(resume_path):
        print(f"🔄 Resuming training from {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location=config.device)
        
        # 1. 恢复模型权重
        # 注意：如果你的模型被 torch.compile 编译过，key 可能会带有 '_orig_mod.' 前缀
        # 这里做一个简单的处理以防万一
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        
        # 2. 恢复优化器和调度器 (关键！)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # 3. 恢复 Epoch 和 最佳分数
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1 # 从下一轮开始
            print(f"   -> Start from Epoch {start_epoch}")
            
        if 'val_loss_min' in checkpoint:
            best_val_mpjpe = checkpoint['val_loss_min']
            print(f"   -> Previous Best MPJPE: {best_val_mpjpe:.2f}")

    # ==========================================
    # 3. 训练循环
    # ==========================================
    # 初始化早停 (Patience 设大点以适应 WarmRestart)
    early_stopping = EarlyStopping(patience=100, verbose=True, path="best_model.pth")

    # 【新增】如果在续训，需要把 EarlyStopping 的状态也恢复一下，否则一开始就被判定不如历史最佳
    if best_val_mpjpe != np.inf:
        early_stopping.best_score = -best_val_mpjpe
        early_stopping.val_loss_min = best_val_mpjpe
    
    print(f"🚀 Start Training on {config.device} with Batch Size {config.batch_size}...")

    # 定义 Warmup 轮数 (建议 5-10 个 Epoch)
    warmup_epochs = 3
    base_lr = config.lr # 4e-4
    
    for epoch in range(start_epoch, config.epochs):

        # ==========================================
        # 【新增】手动 Warmup 逻辑
        # ==========================================
        if epoch < warmup_epochs:
            # 线性增长: 从 0 增加到 base_lr
            # 比如 epoch 0: lr = 0.2 * base_lr
            # ... epoch 4: lr = 1.0 * base_lr
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            
            # 强制修改优化器里的 lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
                
            print(f"🔥 Warmup Epoch {epoch+1}/{warmup_epochs} | LR force set to: {warmup_lr:.2e}")
        
        # ==========================================
        # --- A. 训练 ---
        # 返回的是一个字典: {'total':..., 'pos':..., 'bone':...}
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device
        )

       # --- B. 验证 ---
        # 【修改】接收第4个返回值: val_frames_mpjpe
        val_metrics, val_mpjpe, val_per_joint, val_frames_mpjpe = validate(
            model, val_loader, criterion, config.device
        )
    
        # 获取 Total Loss 用于 Scheduler 和 TensorBoard
        val_loss_total = val_metrics['total']

        # --- C. 调度器步进 ---
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- D. 记录可视化 (TensorBoard) ---
        # 记录训练集各项 Loss
        writer.add_scalar('Train/Total', train_metrics['total'], epoch)
        writer.add_scalar('Train/Pos', train_metrics['pos'], epoch)
        writer.add_scalar('Train/Root', train_metrics['root'], epoch)
        writer.add_scalar('Train/Local', train_metrics['local'], epoch)
        writer.add_scalar('Train/Bone', train_metrics['bone'], epoch)
        writer.add_scalar('Misc/LR', current_lr, epoch)
        
        # 记录验证集
        # writer.add_scalar('Val/Total', val_loss, epoch)
        # writer.add_scalar('Val/MPJPE', val_mpjpe, epoch)
        writer.add_scalar('Val/Total', val_metrics['total'], epoch)
        writer.add_scalar('Val/Pos', val_metrics['pos'], epoch)      # 新增
        writer.add_scalar('Val/Root', val_metrics['root'], epoch)    # 新增
        writer.add_scalar('Val/Local', val_metrics['local'], epoch)  # 新增
        writer.add_scalar('Val/Bone', val_metrics['bone'], epoch)    # 新增
        writer.add_scalar('Val/MPJPE', val_mpjpe, epoch)

        writer.add_scalar('Val_Frames/Frame_1', val_frames_mpjpe[0], epoch)
        writer.add_scalar('Val_Frames/Frame_30', val_frames_mpjpe[1], epoch)
        writer.add_scalar('Val_Frames/Frame_50', val_frames_mpjpe[2], epoch)
        writer.add_scalar('Val_Frames/Frame_71', val_frames_mpjpe[3], epoch)
        writer.add_scalar('Val_Frames/Frame_73', val_frames_mpjpe[4], epoch)
        writer.add_scalar('Val_Frames/Frame_75', val_frames_mpjpe[5], epoch)
        # 记录每个关节的误差
        for i, name in enumerate(JOINT_NAMES):
            writer.add_scalar(f'Val_Joints/{name}', val_per_joint[i], epoch)

        # --- E. 写入 CSV ---
        err_str = ",".join([f"{x:.2f}" for x in val_per_joint])
        with open(csv_path, 'a') as f:
            f.write(f"{epoch+1},{err_str}\n")

        # --- F. 打印日志 ---
        print(f"Epoch {epoch+1}/{config.epochs} | LR: {current_lr:.2e}")
        print(f"  [Train] Total: {train_metrics['total']:.4f} | "
              f"Pos: {train_metrics['pos']:.4f} | "
              f"Root: {train_metrics['root']:.4f} | "
              f"Local: {train_metrics['local']:.4f} | "
              f"Bone: {train_metrics['bone']:.4f}")

        print(f"  [Val]   Total: {val_metrics['total']:.4f} | "
          f"Pos: {val_metrics['pos']:.4f} | "
          f"Root: {val_metrics['root']:.4f} | "
          f"Local: {val_metrics['local']:.4f} | "
          f"Bone: {val_metrics['bone']:.4f}")
          
        print(f"          MPJPE: {val_mpjpe:.2f} mm")

        print(f"  [Frames]  1st: {val_frames_mpjpe[0]:.1f} | "
              f"30th: {val_frames_mpjpe[1]:.1f} | "
              f"50th: {val_frames_mpjpe[2]:.1f} | "
              f"71th: {val_frames_mpjpe[3]:.1f} | "
              f"73th: {val_frames_mpjpe[4]:.1f} | "
              f"75th: {val_frames_mpjpe[5]:.1f}")
        
        # --- G. 早停与保存 ---
        # EarlyStopping 内部逻辑：
        # 如果 val_mpjpe 创新低 -> 自动保存到 best_model.pth -> 更新 best_score
        # 如果 连续 patience 次没创新低 -> early_stop = True
        # early_stopping(val_mpjpe, model)
        early_stopping(val_mpjpe, model, optimizer, scheduler, epoch)

        if early_stopping.early_stop:
            print("🚀 Early stopping triggered! Training finished.")
            break
            
    # 训练结束，关闭 Writer
    writer.close()
    
    # # 加载最优模型进行最终确认
    # if os.path.exists("best_model.pth"):
    #     model.load_state_dict(torch.load("best_model.pth"))
    #     print(f"Done. Loaded Best Model (MPJPE: {early_stopping.val_loss_min:.2f} mm)")
    # else:
    #     print("Done. No best model saved (maybe did not improve).")
    
    # ==========================================
    # 加载最优模型进行最终确认 (最终修复版)
    # ==========================================
    if os.path.exists("best_model.pth"):
        print(f"Loading best model from best_model.pth ...")
        
        # 1. 加载文件
        checkpoint = torch.load("best_model.pth", map_location=config.device)
        
        # 2. 提取 state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            best_mpjpe = checkpoint.get('val_loss_min', 0.0)
        else:
            state_dict = checkpoint
            best_mpjpe = early_stopping.val_loss_min

        # 3. 统一清洗权重 (去掉 _orig_mod. 前缀，变成标准权重)
        # 这样无论你以后是不编译推理，还是在别的机器上跑，都能用
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v 
            else:
                new_state_dict[k] = v
        
        # 4. 【核心修复】智能加载
        # 检查当前内存中的 model 是否是被编译过的 (Dynamo OptimizedModule)
        # 如果是，它内部会有一个 _orig_mod 属性指向原始模型
        if hasattr(model, '_orig_mod'):
            print("Info: Detected compiled model. Loading weights into underlying '_orig_mod'...")
            msg = model._orig_mod.load_state_dict(new_state_dict, strict=False)
        else:
            print("Info: Loading weights into standard model...")
            msg = model.load_state_dict(new_state_dict, strict=False)
        
        print(f"Done. Loaded Best Model (Recorded MPJPE: {best_mpjpe:.2f} mm)")
        
        # 只打印真正的问题，忽略不重要的警告
        if len(msg.missing_keys) > 0:
            print(f"⚠️ Warning: Missing keys: {msg.missing_keys}")
        if len(msg.unexpected_keys) > 0:
            print(f"⚠️ Warning: Unexpected keys: {msg.unexpected_keys}")
            
    else:
        print("Done. No best model saved (maybe did not improve).")

if __name__ == "__main__":
    main()