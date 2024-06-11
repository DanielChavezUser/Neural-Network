# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:18:46 2024

@author: danch
"""
import mnist_loader
import network

def main():
    print("insdide")
    nn = network([784, 40, 10])
    training_data, testing_data = mnist_loader.load_data_wrapper()
    
    nn.SGD(training_data, 30, .3, testing_data)
    return 0