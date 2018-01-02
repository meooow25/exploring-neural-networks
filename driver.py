# -*- coding: utf-8 -*-
"""
Driver program to load data and train the neural network
"""

from matplotlib import pyplot as plt
import numpy as np
import mnist_loader as loader
import neuralnetwork as nn

training_data_images_path = r'.\mnist\train-images-idx3-ubyte.gz'
training_data_labels_path = r'.\mnist\train-labels-idx1-ubyte.gz'
test_data_images_path = r'.\mnist\t10k-images-idx3-ubyte.gz'
test_data_labels_path = r'.\mnist\t10k-labels-idx1-ubyte.gz'

def main():
    
    training_data_images = loader.load_images(training_data_images_path)
    training_data_labels = loader.load_labels(training_data_labels_path)
    test_data_images = loader.load_images(test_data_images_path)
    test_data_labels = loader.load_labels(test_data_labels_path)
    
    training_data = [(i.reshape((-1, 1)) / 255.,
                      (np.arange(0, 10) == j).astype('f').reshape((-1, 1)))
                    for i, j in zip(training_data_images, training_data_labels)]
    test_data = [(i.reshape((-1, 1)) / 255.,
                  (np.arange(0, 10) == j).astype('f').reshape((-1, 1)))
                for i, j in zip(test_data_images, test_data_labels)]
    
    print('Training data: count =', len(training_data))
    print('Test data: count =', len(test_data))
    
    print('Sample images:')
    for i in range(4):
        for j in range(4):
            c = i*4 + j
            plt.subplot(4, 4, c+1)
            plt.imshow(training_data[c][0].reshape((28, 28)), cmap='gray')
    plt.show()
    
    print('Training neural network')
    net = nn.NeuralNetwork([784, 30, 10])
    net.stochastic_gradient_descent(training_data, eta=3., epochs=20,
                                    batch_size=10, test_data=test_data)
    
if __name__ == '__main__':
    main()