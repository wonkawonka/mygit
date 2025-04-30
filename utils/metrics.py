import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe


def optimized_metric(pred, true, epsilon=1e-10):
    # 预先计算差值和绝对差值
    diff = pred - true
    abs_diff = np.abs(diff)

    # 计算基础指标
    mae = np.mean(abs_diff)
    squared_diff = diff ** 2
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)

    # 安全处理零除问题
    true_safe = np.where(np.abs(true) > epsilon, true, epsilon)
    relative_diff = diff / true_safe
    abs_relative_diff = np.abs(relative_diff)

    mape = np.mean(abs_relative_diff) * 100
    mspe = np.mean(relative_diff ** 2)

    return mae, mse, rmse, mape, mspe