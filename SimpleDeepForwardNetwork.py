#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 01:05:31 2019

@author: ins
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from math import exp

# Neural Network for solving x1-x2 Problem
# 1 1 --> 0
# 1 0 --> 1
# 0 1 --> -1
# 0 0 --> 0



class SimpleForwardNetwork:
    
    w1=w2=0
    
    # Activation function: sigmoid
    def sigmoid(self,x): return np.exp(x)/(1 + np.exp(x))
    
    # Sigmoid deriative
    def sigmoid_deriv(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    # Forward function
    def forward(self,x,predict=False):
        a1 = np.matmul(x,self.w1)
        #print(a1)
        z1 = self.sigmoid(a1)
        '''for x in np.nditer(z1, op_flags = ['readwrite']):
            x[...] = reclu(x)'''
      
        # create and add the bias
        bias = np.ones((len(z1),1))
        z1 = np.concatenate((bias,z1),axis = 1)
        a2 = np.matmul(z1,self.w2)
        z2 = self.sigmoid(a2)
        if(predict):
            return z2
        return a1,z1,a2,z2

    # Backprop fuction
    def backprop(self,a2,z0,z1,z2,y):
        a1, z1, a2, z2 = self.forward(z0)
        delta2 = (z2-y)
        Delta2 = np.matmul(z1.T,delta2)
        delta1 = (delta2.dot(self.w2[1:,:].T))*self.sigmoid_deriv(a1)
        Delta1 = np.matmul(z0.T,delta1)
        return delta2, Delta1, Delta2






# Make Predictions for the training inputs
# z3 = model.forward(x,True)

def main():
    # creating an object for SimpleForwardNetwork class.
    simplefnw = SimpleForwardNetwork()

    # First column is the bias
    x = np.array([[1,1,1],[1,2,1],[1,3,1],[1,4,1],[1,1,2],[1,2,2],[1,3,2],[1,4,2],[1,1,3],[1,2,3],[1,3,3],[1,4,3],[1,1,4],[1,2,4],[1,3,4],[1,4,4]])
    y = np.array([[0],[1],[1],[1],[-1],[0],[1],[1],[-1],[-1],[0],[1],[-1],[-1],[-1],[0]])


    # init weights 
    np.random.seed(1231)
    simplefnw.w1 = np.random.randn(3,10)
    simplefnw.w2 = np.random.randn(11,1)

    #print(simplefnw.w1)


    # init learning rate
    lr = 0.001
    costs = []
    
    # init epochs
    epochs = 27322
    m = len(x)

    # Start training
    for i in range(epochs):
    
        # Forward
        a1, z1, a2, z2 = simplefnw.forward(x)
        
        # Backprop
        delta2, Delta1, Delta2 = simplefnw.backprop(a2, x, z1, z2, y)
        #print(delta2,Delta1,Delta2)
    
        simplefnw.w1 -= lr*(1/m)*Delta1
        simplefnw.w2 -= lr*(1/m)*Delta2
    
        # Add costs to list for plotting
        c = np.mean(np.abs(delta2))
        costs.append(c)
        
        if i % 1000 == 0:
            print("Iteration:",i, "Error: ",c)
        
    # Training complete
    print('Training completed.')
    plt.plot(costs)
    plt.show()
        
    # Saving the model to disk
    pickle.dump(simplefnw,open('model.pkl','wb'))
        
    '''
    # loading the model
    model = pickle.load(open('model.pkl','rb'))
        
    print('Percentages: ')
    print(model.forward(x,True))
    print('Predictions: ')

    # Initialize Threshold
    thresh = model.forward(x,True)
    thresh[(thresh > 1e-05)] = 1
    thresh[(thresh > 1e-10) & (thresh < 1e-05)] = 0
    thresh[(thresh < 1) & (thresh>0)] = -1
    print(thresh)'''


if __name__ == '__main__':
    main()




'''
# Make Predictions for testing inputs
test = np.array([[1,6,6],[1,8,9],[1,100,9],[1,-20,30]])
#model = model(test,w1,w2,True)
#z3 = forward(test,w1,w2,True)
print('Percentages: ')
print(model.forward(test,True))
print('Predictions: ')
thresh = model.forward(test,True)
thresh[(thresh > 1e-05)] = 1
thresh[(thresh > 1e-10) & (thresh < 1e-05)] = 0
thresh[(thresh < 1) & (thresh>0)] = -1
print(thresh)'''