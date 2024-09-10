import numpy as np

class Loss:
    def f(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    def df(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.f(y_true, y_pred)


class MSE(Loss):
    def f(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.square(y_true - y_pred).mean()

    def df(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -2 * (y_true - y_pred).mean()


class BinaryCrossEntropy(Loss):
    def f(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()

    def df(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -((y_true / y_pred) - ((1 - y_true) / (1 - y_pred))).mean()


class BCE(BinaryCrossEntropy):
    def f(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return super(y_true, y_pred)

    def df(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return super.df(y_pred)


class CategoricalCrossEntropy(Loss):
    def f(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -np.sum(y_true * np.log(y_pred))

    def df(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -np.sum(y_true / y_pred)


class CCE(CategoricalCrossEntropy):
    def f(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return super(y_true, y_pred)

    def df(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return super.df(y_pred)
