import numpy as np
from typing import Callable

from layer import Layer
from losses import Loss

class Model:
    def __init__(self, layers: list[Layer] = []):
        self.layers = layers

    def add_layer(self, layer: Layer) -> None:
        self.layers.push(layer)

    def add_layers(self, layers: list[Layer]) -> None:
        self.layers += layers

    def eval(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def train(self, xs: np.ndarray, ys: np.ndarray, loss_func: Loss, epochs: int = 1, lr: float = 1e-3, verbose: bool = True) -> list[float]:
        losses = []
        for epoch in range(epochs):
            if verbose:
                print(f"epoch {epoch + 1} / {epochs}:", end="\n\t")

            loss = self._trainstep(xs, ys, loss_func, lr, verbose)
            losses.append(loss)

        return losses

    def _trainstep(self, xs: np.ndarray, ys: np.ndarray, loss_func: Loss, lr: float, verbose: bool) -> float:
        total_loss = 0
        for x, y_true in zip(xs, ys):
            # forward
            y_pred = self.eval(x)
            loss_value = loss_func(y_true, y_pred)
            total_loss += loss_value

            # backward
            dl_dy = loss_func.df(y_true, y_pred)

            z = x
            for layer in self.layers:
                z = layer.backward(z, dl_dy, lr)

        if verbose:
            print(f"loss: {total_loss}")

        return total_loss
