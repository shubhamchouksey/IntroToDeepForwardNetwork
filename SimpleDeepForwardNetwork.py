#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 01:05:31 2019

@author: ins
"""

import numpy as np
import matplotlib.pyplot as plt

# Neural Network for solving x1-x2 Problem
# 1 1 --> 0
# 1 0 --> 1
# 0 1 --> -1
# 0 0 --> 0

# Activation function: sigmoid
from math import exp
def sigmoid(x): return np.exp(x)/(1 + np.exp(x))

# Sigmoid deriative
def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

# Forward function
def forward(x,w1,w2,predict=False):
    a1 = np.matmul(x,w1)
    #print(a1)
    z1 = sigmoid(a1)
    '''for x in np.nditer(z1, op_flags = ['readwrite']):
        x[...] = reclu(x)'''
      
    # create and add the bias
    bias = np.ones((len(z1),1))
    z1 = np.concatenate((bias,z1),axis = 1)
    a2 = np.matmul(z1,w2)
    z2 = sigmoid(a2)
    if(predict):
        return z2
    return a1,z1,a2,z2

# Backprop fuction
def backprop(a2,z0,z1,z2,y):
    a1, z1, a2, z2 = forward(x,w1,w2)
    delta2 = (z2-y)
    Delta2 = np.matmul(z1.T,delta2)
    delta1 = (delta2.dot(w2[1:,:].T))*sigmoid_deriv(a1)
    Delta1 = np.matmul(z0.T,delta1)
    return delta2, Delta1, Delta2

# First column is the bias
x = np.array([[1,1,1],[1,2,1],[1,3,1],[1,4,1],[1,1,2],[1,2,2],[1,3,2],[1,4,2],[1,1,3],[1,2,3],[1,3,3],[1,4,3],[1,1,4],[1,2,4],[1,3,4],[1,4,4]])
y = np.array([[0],[1],[1],[1],[-1],[0],[1],[1],[-1],[-1],[0],[1],[-1],[-1],[-1],[0]])


# init weights 
np.random.seed(1231)
w1 = np.random.randn(3,10)
w2 = np.random.randn(11,1)

# init learning rate
lr = 0.001
costs = []

# init epochs
epochs = 27322
m = len(x)

# Start training
for i in range(epochs):
    
    # Forward
    a1, z1, a2, z2 = forward(x, w1, w2)
    
    # Backprop
    delta2, Delta1, Delta2 = backprop(a2, x, z1, z2, y)
    #print(delta2,Delta1,Delta2)
    
    w1 -= lr*(1/m)*Delta1
    w2 -= lr*(1/m)*Delta2
    
    # Add costs to list for plotting
    c = np.mean(np.abs(delta2))
    costs.append(c)
    
    if i % 1000 == 0:
        print("Iteration:",i, "Error: ",c)
        
# Training complete
print('Training completed.')
plt.plot(costs)
plt.show()

# Make Predictions for the training inputs
z3 = forward(x,w1,w2,True)
print('Percentages: ')
print(z3)
print('Predictions: ')
k = z3
k[(k > 1e-05)] = 1
k[(k > 1e-10) & (k < 1e-05)] = 0
k[(k < 1) & (k>0)] = -1
k
print(k)

# Make Predictions for testing inputs
test = np.array([[1,6,6],[1,8,9],[1,100,9],[1,-20,30]])
z3 = forward(test,w1,w2,True)
print('Percentages: ')
print(z3)
print('Predictions: ')
k = z3
k[(k > 1e-05)] = 1
k[(k > 1e-10) & (k < 1e-05)] = 0
k[(k < 1) & (k>0)] = -1
k
print(k)