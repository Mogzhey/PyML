import numpy as np

class Activation:
    def f(self, x: np.ndarray) -> np.ndarray:
        pass
    
    def df(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.f(x)


class Linear(Activation):
    def f(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def df(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class ReLU(Activation):
    def f(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(np.zeros_like(x), x)
    
    def df(self, x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0, 0, 1)
    

class Sigmoid(Activation):
    def _positive_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def _negative_sigmoid(self, x):
        exp = np.exp(x)
        return exp / (exp + 1)


    def f(self, x):
        # stable sigmoid, naiive implementation overflows
        positive = x >= 0
        negative = ~positive

        result = np.zeros_like(x, dtype=float)
        result[positive] = self._positive_sigmoid(x[positive])
        result[negative] = self._negative_sigmoid(x[negative])

        return result
    
    def df(self, x: np.ndarray) -> np.ndarray:
        fx = self.f(x)
        return fx * (1 - fx)


class Softmax(Activation): # broken
    def f(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x - np.max(x)) # subtract np.max(x) from arg to make it stable
        return exp / np.sum(exp)

    def df(self, x: np.ndarray) -> np.ndarray:
        s = self.f(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
