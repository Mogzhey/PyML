import numpy as np
from typing import Callable

from layer import Layer

class Model:
    def __init__(self, loss_func: Callable[[np.ndarray], np.ndarray], layers: list[Layer] = []):
        self.layers = []

    def add_layer(self, layer: Layer) -> None:
        self.layers.push(layer)

    def add_layers(self, layers: list[Layer]) -> None:
        self.layers += layers

    def eval(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)

        return x   

    def train(self, xs: np.ndarray, ys: np.ndarray, epochs: int, lr: float = 1e-3):
        pass
