import numpy as np
import torch

class GasAugmentor:
    def __init__(self, log_transform=True):
        self.log_transform = log_transform

    def preprocess(self, x):
        """基础预处理：Log 转换平衡特征量级"""
        if self.log_transform:
            # 使用 log1p 处理大数值，保持符号
            return np.log1p(np.abs(x)) * np.sign(x)
        return x

    def jitter(self, x, sigma=0.03):
        """加噪：模拟传感器热噪声"""
        return x + np.random.normal(0, sigma, x.shape)

    def scaling(self, x, sigma=0.1):
        """缩放：模拟每个传感器的灵敏度漂移 (Drift)"""
        # 为 16 个通道分别生成缩放因子
        factors = np.random.uniform(1-sigma, 1+sigma, (16, 1))
        return x * factors

    def __call__(self, x):
        """SimCLR 要求的双视图输出"""
        x_pre = self.preprocess(x)
        # 返回两个不同的增强版本作为对比对
        view1 = self.jitter(self.scaling(x_pre))
        view2 = self.jitter(self.scaling(x_pre))
        return view1, view2