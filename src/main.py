import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

from model import Model
from dense import Dense
from activations import ReLU, Sigmoid
from losses import MSE

DATA_PATH = Path(__file__).parent.parent / "data"

if __name__ == "__main__":
    # load banknotes dataset to test model training
    df = pd.read_csv(DATA_PATH / "banknotes.csv")
    xs = np.array(df[df.columns[:-1]])
    ys = np.array(df[df.columns[-1]])

    train_xs = xs[:int(0.8 * len(xs))]
    train_ys = ys[:int(0.8 * len(xs))]

    test_xs = xs[int(0.8 * len(xs)):]
    test_ys = ys[int(0.8 * len(xs)):]

    # build model
    model = Model([
        Dense(input_size=4, units=8, activation=ReLU()),
        Dense(input_size=8, units=8, activation=ReLU()),
        Dense(input_size=8, units=2, activation=Sigmoid())
    ])

    # train model for 200 epochs using MSE loss function
    train_history = model.train(train_xs, train_ys, MSE(), epochs=200, lr=1e-4, verbose=True)

    # plot training
    fig, ax = plt.subplots()
    ax.plot(range(len(train_history)), train_history, color="#0af")
    plt.show()

# TODO: fix broken Loss and Activation functions
#     - BCE Loss broken, returns [1. 1.] when the sum should be 1.
#     - CCE Loss broken
#     - Softmax Activation broken
# TODO: Add support for batches
