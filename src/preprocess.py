import numpy as np
import yaml
from sklearn.preprocessing import RobustScaler

def load_and_preprocess(loader, config_path='configs/split.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    sc = config['split_config']

    # 定义一个内部辅助函数，用来处理列表加载
    def load_multiple_batches(batch_ids):
        X_list, y_list = [], []
        for bid in batch_ids:
            X_b, y_b = loader.load_batch(bid)
            X_list.append(X_b)
            y_list.append(y_b)
        return np.vstack(X_list), np.concatenate(y_list)

    # 1. 修正加载逻辑：使用循环加载并合并
    X_train, y_train = load_multiple_batches(sc['train_batches'])
    X_val, y_val = load_multiple_batches(sc['val_batches'])
    X_test, y_test = load_multiple_batches(sc['test_batches'])

    # 2. 计算 Batch 1 基准均值
    mu_batch1 = np.mean(X_train, axis=0)

    # 3. 定向清洗 SSL 池 (Batch 2-6)
    X_ssl_list, y_ssl_list = [], []
    for bid in sc['ssl_adapt_batches']:
        X_b, y_b = loader.load_batch(bid)
        if bid == 2:
            # 仅对 Batch 2 执行 5% 离群点剔除
            X_b, y_b = _filter_batch2(X_b, y_b, mu_batch1)
        X_ssl_list.append(X_b)
        y_ssl_list.append(y_b)

    X_ssl_clean = np.vstack(X_ssl_list)
    y_ssl_clean = np.concatenate(y_ssl_list)

    # 4. 标准化
    scaler = RobustScaler().fit(X_train)

    return {
        'train': (scaler.transform(X_train), y_train),
        'ssl': (scaler.transform(X_ssl_clean), y_ssl_clean),
        'val': (scaler.transform(X_val), y_val),
        'test': (scaler.transform(X_test), y_test),
        'scaler': scaler
    }

def _filter_batch2(X, y, mu_batch1):
    """
    内部函数：利用欧氏距离阈值剔除 Batch 2 特有的增益异常点
    """
    if X.shape[0] == 0:
        return X, y

    # 计算到 Batch 1 基准的距离
    distances = np.linalg.norm(X - mu_batch1, axis=1)

    # 剔除偏离最大的 5% (即你之前发现的那 63 个点)
    threshold = np.percentile(distances, 95)
    mask = distances <= threshold

    X_filtered = X[mask]
    y_filtered = y[mask]

    print(f"   [Preprocessing] Batch 2 已剔除 {X.shape[0] - X_filtered.shape[0]} 个离群点")
    return X_filtered, y_filtered