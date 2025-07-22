import os
import sys
import numpy as np
import pandas as pd
import pickle
import shutil  # 添加shutil模块用于磁盘空间检查
from typing import Tuple, Dict, Any, Callable
from sklearn.base import RegressorMixin

from constants import OUTPUT_PATH as output_path
from constants import REGRESSORS_DEFAULT as regressors
from constants import TARGETS_LIST as targets
from constants import N_FOLDS_OUTER as n_folds
from constants import SPLIT_DATA_PATH as data_path


# 添加磁盘空间检查函数
def check_disk_space(path, min_gb=1):
    """检查指定路径的可用磁盘空间是否满足最低要求"""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    if free_gb < min_gb:
        raise Exception(f"磁盘空间不足: 剩余 {free_gb:.2f} GB，需要至少 {min_gb} GB")


def train_default_models(
        train_data: np.ndarray,
        regressors: Dict[str, Tuple[Callable[..., RegressorMixin], Dict[str, Any]]],
        output_path: str,
        target: str,
        fold: int,
        must_normalize: bool
) -> None:
    """使用默认参数训练模型"""
    for id_reg, (regressor_class, params) in regressors.items():
        # 构建模型保存路径
        model_dir = os.path.join(output_path, id_reg)
        os.makedirs(model_dir, exist_ok=True)
        model_name = os.path.join(
            model_dir,
            f'default_{id_reg}_{target}_fold{fold:02d}.model'
        )

        # 如果模型已存在，就跳过训练
        if os.path.exists(model_name):
            print(f"模型已存在: {model_name}")
            continue

        try:
            # 检查磁盘空间
            check_disk_space(output_path)

            # 创建并训练模型
            model = regressor_class(**params)

            # 准备训练数据
            X = train_data[:, :-1]
            y = train_data[:, -1]

            # 对目标变量进行归一化处理（如果需要）
            if must_normalize:
                y = np.log(y)

            # 训练模型
            model.fit(X, y)
            print(f'{id_reg} 模型已生成。')

            # 再次检查磁盘空间
            check_disk_space(output_path)

            # 保存模型
            with open(model_name, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            print(f"训练 {id_reg} 模型时出错: {str(e)}")


def train_best_models(
        train_data: np.ndarray,
        regressors: Dict[str, Tuple[Callable[..., RegressorMixin], Dict[str, Any]]],
        output_path: str,
        target: str,
        fold: int,
        must_normalize: bool
) -> None:
    """使用最优参数配置训练模型"""
    for id_reg, (regressor_class, _) in regressors.items():
        # 加载最优参数配置
        config_path = os.path.join(
            output_path, id_reg,
            f'best_configuration_{id_reg}_{target}_fold{fold:02d}_.rcfg'
        )

        # 构建模型保存路径
        model_dir = os.path.join(output_path, id_reg)
        os.makedirs(model_dir, exist_ok=True)
        model_name = os.path.join(
            model_dir,
            f'best_{id_reg}_{target}_fold{fold:02d}.model'
        )

        # 如果模型已存在，就跳过训练
        if os.path.exists(model_name):
            print(f"模型已存在: {model_name}")
            continue

        try:
            # 加载保存的参数配置
            with open(config_path, 'rb') as f:
                conf = pickle.load(f)

            # 获取模型参数（假设格式为(class_args, class_kwargs)）
            _, params = conf

            # 修正决策树的criterion参数（如果需要）
            if regressor_class.__name__ == 'DecisionTreeRegressor' and 'criterion' in params:
                if params['criterion'] == 'mse':
                    params['criterion'] = 'squared_error'
                    print("已将决策树的criterion参数从'mse'修正为'squared_error'")

            # 特殊处理CatBoost
            if id_reg == 'catboost':
                params['thread_count'] = None

            # 检查磁盘空间
            check_disk_space(output_path)

            # 创建并训练模型
            model = regressor_class(**params)

            # 准备训练数据
            X = train_data[:, :-1]
            y = train_data[:, -1]

            # 对目标变量进行归一化处理（如果需要）
            if must_normalize:
                y = np.log(y)

            # 训练模型
            model.fit(X, y)
            print(f'{id_reg} 模型已生成。')

            # 再次检查磁盘空间
            check_disk_space(output_path)

            # 保存模型
            with open(model_name, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        except FileNotFoundError:
            print(f"找不到参数配置文件: {config_path}")
        except Exception as e:
            print(f"训练 {id_reg} 模型时出错: {str(e)}")


def main(parameters: list) -> None:
    """主函数，控制整个训练流程"""
    # 检查命令行参数
    if len(parameters) < 2:
        print("使用方法: python script.py [True/False]")
        print("参数:")
        print("  [True/False]: 是否对目标变量进行对数归一化")
        sys.exit(1)

    must_normalize = parameters[1].lower() == 'true'

    # 为每个目标变量训练模型
    for target in targets:
        print(f'正在为 {target} 训练模型')

        # 对每个fold进行训练
        for k in range(1, n_folds + 1):
            # 加载训练数据
            input_file = f'{data_path}/{target}_train_fold{k:02d}.csv'

            try:
                train_data = pd.read_csv(input_file)
                train_data = train_data.values
                print(f'默认参数模型 - Fold {k:02d}')

                # 训练默认参数模型
                train_default_models(
                    train_data, regressors, output_path, target, k, must_normalize
                )

                print(f'调优参数模型 - Fold {k:02d}')

                # 训练调优后的模型
                train_best_models(
                    train_data, regressors, output_path, target, k, must_normalize
                )

            except FileNotFoundError:
                print(f"找不到训练数据文件: {input_file}")
                continue
            except Exception as e:
                print(f"处理 fold {k} 时出错: {str(e)}")
                continue


if __name__ == '__main__':
    main(sys.argv)