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
        return self.activation.f(np.tensordot(inp, self.weights, axis=1) + self.bias)

    def backward(self, xs: np.ndarray, ys: np.ndarray) -> None:
        pass
