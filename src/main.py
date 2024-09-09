import numpy as np

from dense import Dense
from activations import ReLU, Softmax
from losses import cce
from model import Model

if __name__ == "__main__":
    model = Model(cce, [
        Dense(input_size=28*28, units=128, activation=ReLU()),
        Dense(input_size=128, units=32, activation=ReLU()),
        Dense(input_size=32, units=10, activation=Softmax())
    ])
