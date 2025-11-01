from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    """
    Calculates a dictionary of regression metrics.
    """
    metrics = {}
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred)) # Was overwriting 'metrics'
    metrics['R2'] = r2_score(y_true, y_pred) # Was overwriting 'metrics'
    metrics['MSE'] = mean_squared_error(y_true, y_pred) # Was overwriting 'metrics'
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)

    # RPD: Ratio of Performance to Deviation
    metrics['RPD'] = np.std(y_true) / metrics['RMSE'] # Was overwriting 'metrics' and dividing by MSE

    # MBE: Mean Bias Error
    metrics['MBE'] = np.mean(y_pred - y_true) # Was overwriting 'metrics'
    
    # MAPE: Mean Absolute Percentage Error
    metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return metrics