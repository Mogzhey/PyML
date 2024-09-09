import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.square(y_true - y_pred).mean()

def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -1 / y_true.size * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return binary_crossentropy(y_true, y_pred)

def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # grad = return -y_true/(y_pred)
    return -np.sum(y_true * np.log(y_pred))

def cce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return categorical_crossentropy(y_true, y_pred)
