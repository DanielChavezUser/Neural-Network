"""
Created on Tue Jun  4 13:18:46 2024

@author: danch
"""
import mnist_loader1
import network1

def main():
    
    nn = network1.Network([784, 40, 10])
    training_data, testing_data = mnist_loader1.load_data_wrapper()
    
    nn.SGD(training_data, 30, 10, 3.0, testing_data)
    return 0

if __name__ == "__main__":
    main()
    
    
