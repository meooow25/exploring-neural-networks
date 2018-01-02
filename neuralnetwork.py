# -*- coding: utf-8 -*-
"""
A feedforward neural network that learns using backpropagation and
stochastic gradient descent.
"""

import numpy as np
import random

class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes[:]
        self.layers = len(sizes) - 1
        self.wt = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.bs = [np.random.randn(x, 1) for x in sizes[1:]]
        
    def feedforward(self, a):
        for w, b in zip(self.wt, self.bs):
            a = sigmoid(w.dot(a) + b)
        return a
    
    def backprop(self, x, y):
        aa, zz = [x], []
        a = x
        for w, b in zip(self.wt, self.bs):
            z = w.dot(a) + b
            zz.append(z)
            a = sigmoid(z)
            aa.append(a)
        
        delta_C_b = [None] * self.layers
        delta_C_w = [None] * self.layers
        delta = (a - y) * sigmoid_1(zz[-1])
        delta_C_b[-1] = delta
        delta_C_w[-1] = delta.dot(aa[-2].T)
        for layer in range(2, self.layers+1):
            delta = (self.wt[-layer+1].T.dot(delta)) * sigmoid_1(zz[-layer])
            delta_C_b[-layer] = delta
            delta_C_w[-layer] = delta.dot(aa[-layer-1].T)
        
        return (delta_C_b, delta_C_w)
            
    def update(self, batch, eta):
        delta_C_b = [np.zeros(b.shape) for b in self.bs]
        delta_C_w = [np.zeros(w.shape) for w in self.wt]
        for x, y in batch:
            d_C_b, d_C_w = self.backprop(x, y)
            for d1, d2 in zip(delta_C_b, d_C_b): d1 += d2
            for d1, d2 in zip(delta_C_w, d_C_w): d1 += d2
        for b, db in zip(self.bs, delta_C_b): b -= eta * db / len(batch)
        for w, dw in zip(self.wt, delta_C_w): w -= eta * dw / len(batch)
        
    def evaluate(self, batch):
        cost = correct = 0
        for x, y in batch:
            a = self.feedforward(x)
            cost += np.sum((a - y) ** 2)
            correct += a.argmax() == y.argmax()
        cost *= 2. / len(batch)
        correct *= 100 / len(batch)
        return (cost, correct)
        
    def stochastic_gradient_descent(self, training_data, eta, epochs,
                                    batch_size, test_data=None):
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            for c in range(0, n, batch_size):
                self.update(training_data[c:c+batch_size], eta)
            if test_data is not None:
                cost, accuracy = self.evaluate(test_data)
                print('Test data: cost = {:.4f}, accuracy = {:.2f}%'.format(cost, accuracy))
            print('Epoch {}/{} complete'.format(i+1, epochs))
        print('Training complete')
    
    
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_1(x):
    return sigmoid(x) * (1. - sigmoid(x))
