"""
for loading dataset (MNIST)
"""
import os
import mnist
from PIL import Image

def load_data():
    """
    load data
    """
    path = "data/mnist"
    if not os.path.isdir(path):
        os.mkdir(path)
    mnist.temporary_dir = lambda: path

    return (mnist.train_images(), mnist.train_labels()), (mnist.test_images(), mnist.test_labels())

if __name__=="__main__":
    train, test = load_data()
    im = Image.fromarray(train[0][0,:,:] * -1 + 256)
    im.show()
