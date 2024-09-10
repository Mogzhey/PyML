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
    def f(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def df(self, x: np.ndarray) -> np.ndarray:
        fx = self.f(x)
        return fx * (1 - fx)


class Softmax(Activation):
    def f(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x - np.max(x)) # subtract np.max(x) from arg to make it stable
        return exp / np.sum(exp)

    def df(self, x: np.ndarray) -> np.ndarray:
        # https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/
        S = self.f(x)
        S_vector = S.reshape(-1, 1)
        S_matrix = np.tile(S_vector, S.shape[0])

        return np.diag(S) - (S_matrix * S_matrix.T)
