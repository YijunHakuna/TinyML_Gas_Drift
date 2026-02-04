import os
import yaml
from src.data_loader import GasSensorLoader
from src.preprocess import load_and_preprocess


def main():
    # 1. 环境初始化
    print("🚀 [System] 启动传感器漂移补偿实验流水线...")
    data_path = 'Dataset/'  # 你的 UCI 数据路径
    config_path = 'configs/split.yaml'

    if not os.path.exists(config_path):
        print(f"❌ 错误: 找不到配置文件 {config_path}")
        return

    # 2. 初始化数据加载器 (HAL 层)
    loader = GasSensorLoader(data_dir=data_path)

    # 3. 执行数据准备流程 (Preprocess 层)
    # 此步骤包含：根据 YAML 切分、Batch 2 定向清洗、RobustScaler 基准对齐
    print("⏳ [Data] 正在执行数据切分与 Batch 2 离群点清洗...")
    datasets = load_and_preprocess(loader, config_path=config_path)

    # 4. 验证处理结果 (数据统计与审计)
    print("\n" + "=" * 50)
    print(f"{'数据集部分':<15} | {'样本量':<10} | {'说明'}")
    print("-" * 50)
    for key in ['train', 'ssl', 'val', 'test']:
        X, y = datasets[key]
        desc = "出厂标定" if key == 'train' else "SSL 适应池" if key == 'ssl' else "验证集" if key == 'val' else "长期测试"
        print(f"{key:<15} | {X.shape[0]:<10} | {desc}")
    print("=" * 50 + "\n")

    # 5. 准备进入 Phase 2 (Day 3 - SimCLR 数据增强)
    # 我们将把这里的 datasets['ssl'] 喂入未来的 SimCLR 训练器
    X_ssl, _ = datasets['ssl']
    print(f"✅ 数据准备就绪。SSL 池特征维度: {X_ssl.shape[1]} (16 传感器 x 8 特征)")
    print(f"💡 下一步任务: 对 SSL 数据进行高斯噪声与通道遮蔽增强。")


if __name__ == "__main__":
    main()