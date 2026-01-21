import numpy as np
import os
import glob


class GasSensorLoader:
    """
    针对 UCI Gas Sensor Array Drift 数据集的专用加载器
    该数据集包含 16 个传感器，每个传感器有 8 个特征（共 128 维）
    """

    def __init__(self, data_dir='Dataset'):
        self.data_dir = data_dir
        self.num_features = 128
        self.num_classes = 6

    def _parse_line(self, line):
        """
        解析 UCI 原始数据行
        修正后格式: Label 1:Value 2:Value ... (空格分隔)
        """
        line = line.strip()
        if not line:
            return None, None

        # 使用 split() 默认按空格/制表符切分
        parts = line.split()

        try:
            # 第一位是标签
            label = int(parts[0]) - 1

            # 初始化 128 维特征
            features = np.zeros(self.num_features)

            # 从第二位开始是 1:val 格式的特征
            for item in parts[1:]:
                if ':' in item:
                    idx_str, val_str = item.split(':')
                    idx = int(idx_str) - 1  # 索引转为从 0 开始
                    val = float(val_str)
                    if idx < self.num_features:
                        features[idx] = val

            return features, label

        except (ValueError, IndexError) as e:
            # 打印有问题的行，方便调试
            print(f"解析错误行: {line[:50]}... 错误: {e}")
            return None, None

    def load_batch(self, batch_num):
        """
        加载指定的 Batch (1-10)
        """
        file_pattern = os.path.join(self.data_dir, f'batch{batch_num}.dat')
        files = glob.glob(file_pattern)

        if not files:
            raise FileNotFoundError(f"未找到 Batch {batch_num} 的数据文件，请检查 目录。")

        X, y = [], []
        with open(files[0], 'r') as f:
            for line in f:
                features, label = self._parse_line(line)
                if features is not None:
                    X.append(features)
                    y.append(label)

        return np.array(X), np.array(y)

    def load_all_batches(self):
        """
        依次加载所有 10 个 Batch，返回列表
        """
        all_data = []
        for i in range(1, 11):
            X, y = self.load_batch(i)
            all_data.append((X, y))
            print(f"Batch {i} loaded: {X.shape[0]} samples.")
        return all_data


# 简单单元测试
if __name__ == "__main__":
    # 假设你已经把下载的 .dat 文件放进了 data/raw 文件夹
    loader = GasSensorLoader()
    try:
        X_batch1, y_batch1 = loader.load_batch(1)
        print(f"Successfully loaded Batch 1. Shape: {X_batch1.shape}")
    except Exception as e:
        print(f"Error: {e}")