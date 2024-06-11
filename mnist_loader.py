# -*- coding: utf-8 -*-
"""
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


def load_data_wrapper():
    vec_img, ex_output = mnist_data()
    
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    '''These two are the same but should be different'''
    training_inputs = [np.reshape(x, (784, 1)) for x in vec_img]
    training_results = [vectorized_result(y) for y in ex_output]
    training_data = list(zip(training_inputs, training_results))
    
    test_inputs =[np.reshape(x, (784, 1)) for x in vec_img]
    test_outputs = [vectorized_result(y) for y in ex_output]
    test_data = list(zip(test_inputs, test_outputs))
    return (training_data, test_data)

def plot_digit(X, y, idx):
    img = X[idx].reshape(28, 28)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.title('Digit Corresponding to img: %d' % y[idx])
    
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e