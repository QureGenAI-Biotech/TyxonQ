import numpy as np

def mean_absolute_error(true_values, predicted_values):
    return np.mean(np.abs(true_values - predicted_values))

def root_mean_squared_error(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values) ** 2))

def mean_absolute_percentage_error(true_values, predicted_values):
    return np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

def explained_variance_score(true_values, predicted_values):
    return 1 - np.var(true_values - predicted_values) / np.var(true_values)


