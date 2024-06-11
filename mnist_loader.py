Created on Tue Jun  4 12:59:28 2024

@author: danch
"""

"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library


from mlxtend.data import mnist_data
# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import time


def load_data_wrapper():
    vec_img, ex_output = mnist_data()
    
    '''These two are the same but should be different'''
    training_inputs = [np.reshape(x, (784, 1)) for x in vec_img[:-1000]]
    training_results = [vectorized_result(y) for y in ex_output[:-1000]]
    training_data = list(zip(training_inputs, training_results))
    
    '''this is testing that must be asjusted'''
    test_inputs =[np.reshape(x, (784, 1)) for x in vec_img[-1000:]]
    test_outputs = [vectorized_result(y) for y in ex_output[-1000:]]
    test_data = list(zip(test_inputs, test_outputs))
    return (training_data, test_data)

def plot_digit(X, y, idx):
    img = X[idx].reshape(28, 28)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.title('Digit Corresponding to img: %d' % y[idx])

def plot_digit_testing(X, y):
    img = X.reshape(28, 28)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.title('Digit Corresponding to img: %d' % y)
    time.sleep(.5)
    plt.show()
    
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
