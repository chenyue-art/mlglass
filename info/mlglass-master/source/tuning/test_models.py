import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
from constants import TARGETS_LIST as targets
from constants import REGRESSORS_LIST as regressors
from constants import OUTPUT_PATH as output_path
from constants import SPLIT_DATA_PATH as data_path
from constants import N_FOLDS_OUTER as n_folds
from preprocessing import preprocess_data
from sklearn.preprocessing import RobustScaler


def test_models(test_path, model, metrics):
    """测试模型性能"""
    test_data = pd.read_csv(test_path).values
    X = test_data[:, :-1]
    y = test_data[:, -1]

    # 过滤恒定特征
    X, _ = preprocess_data(X)

    # 标准化（可选）
    X_scaled = RobustScaler().fit_transform(X)

    # 预测并评估
    predictions = model.predict(X_scaled)
    results = {name: metric(y, predictions) for name, metric in metrics.items()}

    return results


def get_predictions(input_path, output_path, regressors, target, fold,
                    must_normalize, model_type='default'):
    """获取模型预测结果"""
    test_data = pd.read_csv(input_path).values
    log = np.zeros((len(test_data), 2 * len(regressors)))
    for j, regressor in enumerate(regressors):
        with open(
                os.path.join(
                    output_path, regressor,
                    f'{model_type}_{regressor}_{target}_fold{fold:02d}.model'
                ), 'rb'
        ) as f:
            regressor = pickle.load(f)
            log[:, 2 * j] = test_data[:, -1]
            if not must_normalize:
                log[:, 2 * j + 1] = regressor.predict(test_data[:, :-1])
            else:
                log[:, 2 * j + 1] = np.exp(regressor.predict(test_data[:, :-1]))

    columns = [[r, '{}_pred'.format(r)] for r in regressors]
    columns = [n for subset in columns for n in subset]
    df = pd.DataFrame(
        columns=columns,
        data=log
    )
    return df


def relative_deviation(obs, pred):
    """计算相对偏差，遇到除零返回默认值"""
    denominator = obs
    # 检查分母是否为零
    if np.any(denominator == 0):
        print("警告: 观测值包含零，相对偏差无法计算，返回默认值NaN")
        return np.nan

    # 计算相对偏差
    rel_dev = np.abs(obs - pred) / denominator
    return np.mean(rel_dev) * 100


def R2(target, pred):
    """计算R平方值"""
    return np.corrcoef(target, pred)[0, 1] ** 2


def RRMSE(target, pred):
    """计算相对均方根误差，遇到除零返回默认值"""
    # 计算分子（均方误差）
    mse = np.mean((target - pred) ** 2)

    # 计算分母（目标值的方差）
    target_var = np.var(target)

    # 处理方差为零的情况
    if target_var == 0:
        print("警告: 目标值方差为零，RRMSE无法计算，返回默认值NaN")
        return np.nan

    # 计算RRMSE
    return np.sqrt(mse / target_var)


def RMSE(target, pred):
    """计算均方根误差"""
    return np.sqrt(np.mean((target - pred) ** 2))


def evaluate_models(input_path, output_path, regressors, target, metrics,
                    fold, must_normalize, type='default'):
    errors = np.zeros((len(metrics), len(regressors)))
    test_data = pd.read_csv(input_path).values

    for j, regressor_name in enumerate(regressors):
        model_path = os.path.join(
            output_path, regressor_name,
            f'{type}_{regressor_name}_{target}_fold{fold:02d}.model'
        )

        # 检查文件存在性和完整性
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在 - {model_path}")
            errors[:, j] = np.nan
            continue

        file_size = os.path.getsize(model_path)
        if file_size < 1024:
            print(f"警告: 模型文件过小可能损坏 - {model_path} (大小: {file_size}字节)")

        # 加载模型并处理异常
        model = None
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except EOFError:
            print(f"错误: 模型文件截断 - {model_path}")
        except pickle.UnpicklingError:
            print(f"错误: 模型文件格式错误 - {model_path}")
        except Exception as e:
            print(f"错误: 加载模型失败 - {model_path}, 原因: {str(e)}")

        if model is None:
            errors[:, j] = np.nan
            continue

        # 进行预测
        try:
            if not must_normalize:
                predictions = model.predict(test_data[:, :-1])
            else:
                predictions = np.exp(model.predict(test_data[:, :-1]))
        except Exception as e:
            print(f"错误: 模型预测失败 - {model_path}, 原因: {str(e)}")
            errors[:, j] = np.nan
            continue

        # 计算评估指标
        for i, metric in enumerate(metrics.values()):
            try:
                errors[i, j] = metric(test_data[:, -1], predictions)
            except Exception as e:
                print(f"错误: 计算指标失败 - {model_path}, 原因: {str(e)}")
                errors[i, j] = np.nan

    df = pd.DataFrame(
        index=[m for m in metrics.keys()],
        columns=regressors,
        data=errors
    )
    return df


def get_predictions(input_path, output_path, regressors, target, fold,
                    must_normalize, type='default'):
    test_data = pd.read_csv(input_path).values
    log = np.zeros((len(test_data), 2 * len(regressors)))
    for j, regressor in enumerate(regressors):
        with open(
                os.path.join(
                    output_path, regressor,
                    '{0}_{1}_{2}_fold{3:02d}.model'.format(
                        type, regressor, target, fold
                    )
                ), 'rb'
        ) as f:
            regressor = pickle.load(f)
            log[:, 2 * j] = test_data[:, -1]
            if not must_normalize:
                log[:, 2 * j + 1] = regressor.predict(test_data[:, :-1])
            else:
                log[:, 2 * j + 1] = np.exp(regressor.predict(test_data[:, :-1]))

    columns = [[r, '{}_pred'.format(r)] for r in regressors]
    columns = [n for subset in columns for n in subset]
    df = pd.DataFrame(
        columns=columns,
        data=log
    )
    return df


def generate4fold(input_path, output_path, log_path, regressors, target,
                  metrics, fold, must_normalize):
    print('Fold {}'.format(fold))
    test_path = '{}fold{:02d}.csv'.format(input_path, fold)
    errors_standard = evaluate_models(test_path, output_path, regressors,
                                      target, metrics, fold, must_normalize)
    errors_best = evaluate_models(
        test_path, output_path, regressors, target, metrics, fold,
        must_normalize, 'best'
    )
    errors_standard.to_csv(
        os.path.join(
            log_path,
            'performance_standard_models_{0}_fold{1:02d}.csv'.format(
                target, fold
            )
        )
    )
    errors_best.to_csv(
        os.path.join(
            log_path,
            'performance_best_models_{0}_fold{1:02d}.csv'.format(
                target, fold
            )
        )
    )

    pred_standard = get_predictions(test_path, output_path, regressors,
                                    target, fold, must_normalize)
    pred_best = get_predictions(test_path, output_path, regressors,
                                target, fold, must_normalize, 'best')
    pred_standard.to_csv(
        os.path.join(
            log_path,
            'predictions_standard_models_{0}_fold{1:02d}.csv'.format(
                target, fold
            )
        )
    )
    pred_best.to_csv(
        os.path.join(
            log_path,
            'predictions_best_models_{0}_fold{1:02d}.csv'.format(
                target, fold
            )
        )
    )


def merge_errors(target, output_path, log_path, type='standard'):
    dfs = []

    for k in range(1, n_folds + 1):
        df = pd.read_csv(
            os.path.join(
                log_path,
                'performance_{0}_models_{1}_fold{2:02d}.csv'.format(
                    type, target, k
                )
            )
        )
        col_names = list(df)
        col_names[0] = 'metric'
        df.columns = col_names
        df = df.set_index('metric')
        df = df.assign(fold='fold{:02d}'.format(k))
        dfs.append(df)

    p = pd.concat(dfs)
    means = p.groupby(['metric']).mean(numeric_only=True)
    means.to_csv(
        os.path.join(
            output_path, 'mean_performance_{0}_{1}_all.csv'.format(
                type, target
            )
        )
    )
    stds = p.groupby('metric').std(numeric_only=True)
    stds.to_csv(
        os.path.join(
            output_path, 'std_performance_{0}_{1}_all.csv'.format(
                type, target
            )
        )
    )


metrics = OrderedDict(
    {'relative_deviation': relative_deviation, 'R2': R2, 'RMSE': RMSE,
     'RRMSE': RRMSE}
)

if __name__ == '__main__':
    must_normalize = sys.argv[1] == 'True'
    print()
    print('Testing trained models')
    print()
    log_path = '{0}/logs'.format(output_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    for target in targets:
        for k in range(1, n_folds + 1):
            input_path = '{0}/{1}_test_'.format(
                data_path, target
            )
            generate4fold(
                input_path, output_path, log_path, regressors, target,
                metrics, k, must_normalize
            )
        merge_errors(target, output_path, log_path)
        merge_errors(target, output_path, log_path, 'best')
