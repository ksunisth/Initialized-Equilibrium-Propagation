"""
for loading dataset (MNIST)
"""
import math
import os
import mnist
from PIL import Image
import numpy as np

def load_data(train_ratio):
    """
    load data
    """
    path = "data/mnist"
    if not os.path.isdir(path):
        os.mkdir(path)
    mnist.temporary_dir = lambda: path

    train_val = mnist.train_images(), mnist.train_labels()
    test = mnist.test_images(), mnist.test_labels()

    # shuffle training set
    size = len(train_val[0])
    shuffled_indices = [i for i in range(size)]
    np.random.shuffle(shuffled_indices)

    # calculate training size
    train_size = math.floor(size * train_ratio)

    train = train_val[0][shuffled_indices[:train_size]], train_val[1][shuffled_indices[:train_size]]
    val = train_val[0][shuffled_indices[train_size:]], train_val[1][shuffled_indices[train_size:]]

    return train, val, test



if __name__=="__main__":
    train, val, test = load_data(0.7)
    im = Image.fromarray(train[0][0,:,:] * -1 + 256)
    im.show()
