import numpy as np

from layer import Layer
from activations import Activation, Linear

class Dense(Layer):
    def __init__(self, input_size: int, units: int, seed: int = None, activation: Activation = Linear()):
        self.input_size = input_size
        self.units = units
        self.activation = activation

        self.weights = self.glorot_uniform(shape=(input_size, units), seed=seed)
        self.bias = np.zeros(shape=(units,))

    def glorot_uniform(self, shape: tuple[int], seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        limit = np.sqrt(6 / (self.input_size + self.units))
        return rng.uniform(low=-limit, high=limit, size=shape)

    def forward(self, inp: np.ndarray) -> np.ndarray:
        return self.activation(np.tensordot(inp, self.weights, axes=1) + self.bias)

    def backward(self, inp: np.ndarray, dl_dy: np.ndarray, lr: float = 1e-3) -> np.ndarray:
        # feedforward
        X = self.forward(inp)

        # calculate derivatives
        dy_db = self.activation.df(X)
        dy_dw = np.tensordot(inp, dy_db, axes=0)

        """
        print("\n")
        print(f"weights shape = {self.weights.shape}, dy_dw shape = {dy_dw.shape}")
        print(f"bias shape = {self.bias.shape}, dy_db shape = {dy_db.shape}")
        print(f"X.shape = {X.shape}")
        print(self.activation)
        print("\n")
        """

        # update weights and biasses
        self.weights -= lr * dl_dy * dy_dw
        self.bias -= lr * dl_dy * dy_db

        return X
