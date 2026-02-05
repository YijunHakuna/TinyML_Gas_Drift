import numpy as np
import torch


class GasAugmentor:
    def __init__(self, log_transform=True, sigma_jitter=0.03, sigma_scaling=0.1):
        self.log_transform = log_transform
        self.sigma_jitter = sigma_jitter
        self.sigma_scaling = sigma_scaling

    def preprocess(self, x):
        """基础预处理：Log 转换平衡特征量级"""
        if self.log_transform:
            # 使用 log1p 处理大数值，保持符号
            return np.log1p(np.abs(x)) * np.sign(x)
        return x

    def jitter(self, x):
        """加噪：模拟传感器热噪声"""
        noise = np.random.normal(0, self.sigma_jitter, x.shape)
        return x + noise

    def scaling(self, x):
        """缩放：模拟每个传感器的灵敏度漂移 (Drift)"""
        # 针对 16 个传感器通道 (C) 生成独立缩放因子
        # x shape assumed as (16, 8)
        factors = np.random.uniform(1 - self.sigma_scaling, 1 + self.sigma_scaling, (16, 1))
        return x * factors

    def permutation(self, x):
        """切片交换：增强模型对采样起始点的不敏感度"""
        x_aug = x.copy()
        mid = 4  # T 轴中点
        x_aug[:, :mid], x_aug[:, mid:] = x[:, mid:], x[:, :mid]
        return x_aug

    def __call__(self, x):
        """SimCLR 要求的双视图输出"""
        # 1. 基础预处理
        x_pre = self.preprocess(x).astype(np.float32)

        # 2. 生成第一个视图 (Scaling + Jitter)
        # 模拟：这个时刻传感器发生了漂移，且伴随电路噪声
        v1 = self.jitter(self.scaling(x_pre))

        # 3. 生成第二个视图 (Scaling + Permutation + Jitter)
        # 模拟：另一个时刻的漂移，且由于采样抖动导致时序微偏
        v2 = self.jitter(self.permutation(self.scaling(x_pre)))

        # 转换为 PyTorch FloatTensor 方便后续计算
        return torch.from_numpy(v1), torch.from_numpy(v2)