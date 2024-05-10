def rmse(true_target, predicted_target):
    if len(true_target) != len(predicted_target):
        raise ValueError("true_target and predicted_target must have the same length")
    
    n = len(true_target)
    squared_error = sum([(true_target[i] - predicted_target[i])**2 for i in range(n)])
    rmse = (squared_error/n)**0.5
    return rmse
    
    