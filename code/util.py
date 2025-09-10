import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
import env
import numpy as np

def evaluate(target, prediction, task_type):
    if task_type == 'regression':
        return mean_squared_error(target, prediction)
    elif task_type == 'classification':
        return log_loss(target, prediction)


def get_task_type(Y):
    return 'classification' if Y[0] == 0 or Y[0] == 1 else 'regression'