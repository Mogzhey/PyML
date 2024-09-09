import numpy as np

from dense import Dense
from activations import ReLU, Softmax
from losses import cce
from model import Model

test_xs = np.ones(shape=(64, 28*28))

if __name__ == "__main__":
    model = Model(cce, [
        Dense(input_size=28*28, units=128, activation=ReLU()),
        Dense(input_size=128, units=32, activation=ReLU()),
        Dense(input_size=32, units=10, activation=Softmax())
    ])

    pred_ys = model.eval(test_xs)
    print(pred_ys[0].shape) # shape is wrong
