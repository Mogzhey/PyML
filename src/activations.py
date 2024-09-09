import numpy as np

class Activation:
    def f(self, x: np.ndarray) -> np.ndarray:
        pass
    
    def df(self, x: np.ndarray) -> np.ndarray:
        pass


class Linear(Activation):
    def f(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def df(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class Sigmoid(Activation):
    def f(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def df(self, x: np.ndarray) -> np.ndarray:
        fx = Sigmoid.f(x)
        return fx * (1 - fx)


class ReLU(Activation):
    def f(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(np.zeros_like(x), x)
    
    def df(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0, 0, 1)


class Softmax(Activation):
    def f(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x)
        return exp / np.sum(exp)

    def df(self, x: np.ndarray) -> np.ndarray:
        s = Softmax.f(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
