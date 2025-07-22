from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import numpy as np


def RRMSE(target, pred):
    """计算相对均方根误差，处理目标值全相同的情况"""
    num = np.sum((target - pred) ** 2)
    den = np.sum((np.mean(target) - target) ** 2)

    if np.isclose(den, 0):
        if np.isclose(num, 0):
            return 0.0
        else:
            return np.inf

    return np.sqrt(num / den)


def MSE(target, pred):
    N = len(target)
    return np.sum((target-pred)**2)/N


def RMSE(target, pred):
    return np.sqrt(MSE(target, pred))


def aRRMSE(targets, predictions):
    n_targets = targets.shape[1]
    rrmse_sum = 0.0
    for t in range(n_targets):
        rrmse_sum += RRMSE(targets[:, t], predictions[:, t])

    return rrmse_sum/n_targets


def aRMSE(targets, predictions):
    n_targets = targets.shape[1]
    rmse_sum = 0.0
    for t in range(n_targets):
        rmse_sum += RMSE(targets[:, t], predictions[:, t])

    return rmse_sum/n_targets


def get_regressor(regressor, seed):
    reg = None
    params = None
    if regressor == 'RF':
        reg = RandomForestRegressor
        params = {
            'n_estimators': 100,
            'criterion': 'squared_error',
            'max_depth': None,
            'min_samples_split': 0.01,
            'min_samples_leaf': 0.005,
            'n_jobs': 10,
            'random_state': seed
        }
    elif regressor == 'DT':
        reg = DecisionTreeRegressor
        params = {
            'criterion': 'squared_error',
            'splitter': 'best',
            'max_depth': None,
            'min_samples_split': 0.01,
            'min_samples_leaf': 0.005,
            'random_state': seed
        }
    elif regressor == 'XG':
        reg = xgb.XGBRegressor
        params = {
            'objective': 'reg:linear',
            'n_jobs': 10,
            'n_estimators': 100,
            'random_state': seed
        }
    elif regressor == 'SVM':
        reg = SVR
        params = {
            'kernel': 'rbf',
            'gamma': 'auto',
            'C': 1.0,
            'epsilon': 0.1,
            'max_iter': 10000000
        }
    elif regressor == 'MLP':
        reg = MLPRegressor
        params = {
            'hidden_layer_sizes': 50,
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'max_iter': 800,
            'early_stopping': True,
            'random_state': seed
        }
    else:
        raise ValueError('"regressor" not valid.')

    return reg, params
# 测试代码
regressor_name = 'DT'
reg, params = get_regressor(regressor_name, seed=42)
print(f"{regressor_name} with criterion={params['criterion']}")