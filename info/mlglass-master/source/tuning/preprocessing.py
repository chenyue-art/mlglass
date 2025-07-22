
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold


def preprocess_data(data_path):
    """完整的数据预处理流程"""
    # 加载数据
    data = pd.read_csv(data_path).values

    # 分离特征和目标变量
    X = data[:, :-1]
    y = data[:, -1]

    # 过滤恒定特征
    selector = VarianceThreshold(threshold=0.0)
    X = selector.fit_transform(X)

    # 标准化（使用RobustScaler对异常值更鲁棒）
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y