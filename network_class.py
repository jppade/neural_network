#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:12:00 2020

@author: pade
"""


from math import exp
import numpy as np
from functools import partial
from scipy.optimize import fmin


class NN:
    
    """
    This is the neural network class.
    A neural network is uniquely defined by weights and biases.
    @weights is a list of numpy arrays. Each array corresponds to a coupling 
    matrix acting between two layers.
    @biases is a list of numpy arrays. Each array corresponds to a bias vector.
    """

    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases


# -----------------------------------------------------------------------------
#        Methods
# ----------------------------------------------------------------------------- 

    def get_dim(self):
        """
        Returns a list with integer entries corresponding to the number of 
        neurons per layer.
        """
        dim_layers = [wmat.shape[0] for wmat in self.weights]
        dim_layers.insert(0, self.weights[0].shape[1])
        return dim_layers
    
    def layer_out(self, num_layer, inp_layer, activation):
        """
        Return the output of layer num_layer for a given input 'inp_layer'.
        For the input layer we have num_layer = 0.
	    @num_layer is the number of the considered layer (hence an int)
	    @inp_layer is a numpy array corresponding to the input vector into
		     layer num_layer
	    @activation is the activation function of the layer.
        """
        dim_out = self.get_dim()[num_layer]
        outp_layer = np.zeros(dim_out)
        if num_layer < 1:
            raise ValueError("The first layer is a fixed input and does not have an output.")
        layer_state = self.weights[num_layer-1].dot(inp_layer) + self.biases[num_layer-1]
        for i in range(dim_out):
            outp_layer[i] = activation(layer_state[i])
        return outp_layer

    def netw_out(self, inp, activation):
        """
        Return the output of the whole network by applying layer_out consecutively.
	    @inp is a numpy array corresponding to the input vector into the network.
	    @activation is the activation function of the network.
        """
        for i in range(1,len(self.get_dim())):
            inp = self.layer_out(i, inp, activation)
        return inp

    def train_netw(self, training_data, labels, activation, target):
        """
        Find minimum of the goal_function with respect to the variable xInit.
        @training_data is the training data.
        @lab are the (boolean?) labels attached to the training data.
	    @activation is the network's activation function.
	    @target is the target function for the optimization.
        """
        goal_func_optim = partial(goal_function, dim_layers=self.get_dim(), 
                          data=training_data, labels=labels,
                          activation=activation, target=target
                          )
        xInit, d = shapeForth(self.weights, self.biases)
        print("gradient descent started...")
        x_opt = fmin(goal_func_optim, xInit, disp=False)
        self.weights, self.biases = shapeBack(x_opt, self.get_dim())
        print("gradient descent finished")


# -----------------------------------------------------------------------------
#   Functions
# ----------------------------------------------------------------------------- 

def shapeForth(weight_matrices, biases):
    """
    This function puts the weight matrices and biases into one large numpy
    array in the following order
    x = [weight_matrix[0], bias[0],...,weight_matrix[n], bias[n]]
    where each weight_matrix[i] is unzipped row-wise. Furthermore it returns
    a list of dimensions of the layers (including input and output layer).
    """
    dim_layers = [wm.shape[0] for wm in weight_matrices]
    dim_layers.insert(0,weight_matrices[0].shape[1])    # Dim of input layer
    x = np.zeros(0)
    for wm, bias in zip(weight_matrices, biases):
        x = np.append(x, wm)
        x = np.append(x, bias)
    return x, dim_layers

def shapeBack(y,dim_layers):
    """
    This function is the inverse operation of shapeForth. It takes a numpy
    array which is organized as follows:
    y = [weight_matrix[0], bias[0],...,weight_matrix[n], bias[n]]
    @dim_layers is a list of dimensions/sizes of the layers
    The function returns a list of weight matrices (as numpy arrays) and biases.
    """
    index = 0
    weight_matrices = list()
    biases = list()
    for i in range(len(dim_layers)-1):
        weight_matrices.append(np.reshape(y[index:(index + (dim_layers[i] * 
                        dim_layers[i+1]))] ,(dim_layers[i+1],dim_layers[i])))
        index += dim_layers[i]*dim_layers[i+1]
        biases.append(y[index:(index + dim_layers[i+1])])
        index += dim_layers[i+1]
    return weight_matrices, biases

def goal_function(z, dim_layers, data, labels, activation, target):
    """
    This function computes the distance of the values of the training data
    and the network output (and hence returns a positive real number).
    @z is the numpy array of weights and biases as obtained by shapeForth
    @dim_layers is a list of dimensions of the network's layers
    @data is a list of numpy arrays representing the training data
    @labels is a list representing the labels attached to the training data
    @activation is an activation function for the coupling between layers
    @target is a target function
    """
    weight_matrices, biases = shapeBack(z, dim_layers)
    network = NN(weight_matrices, biases)
    # Sum over all differences between target function and (training) data
    outp = sum([( np.linalg.norm(target(label) \
                  - network.netw_out(dat, activation)) )**2\
                    for label, dat in zip(labels, data)])/len(data)
    return outp

def init_rand(dim_layers, scaling=0.1):
    """
    Returns weights and biases of a random initialization of a neural network
    given the dimensions of the layers as @dim_layers.
    """
    weight_matrices = []
    biases = []
    for i in range(len(dim_layers) - 1):
        weight_matrices.append(scaling * np.random.rand(dim_layers[i+1],
                                                                dim_layers[i]))
        biases.append(np.zeros(dim_layers[i+1]))
    return weight_matrices, biases


# -----------------------------------------------------------------------------
#     Activation functions and target function
# ----------------------------------------------------------------------------- 

def sigmoid(x, a=1, b=1):
    if abs(x) < 40:
        return a / (b + exp(-x))
    elif x > 40:
        return 1
    else:
        return 0

def relu(x):
    return max(0,x)

def hyptang(x):
    return (1 - exp(-2*x))/(1 + exp(-2*x))
    
def myTarget(label, labellist):
    """
    This is the target function for data with n different labels. It maps the 
    k-th label from @labellist to the k-th unit vector in R^n.
    """
    valuelist = [np.insert(np.zeros(len(labellist) - 1), ind, 1) for ind, item
                 in enumerate(labellist)]
    for k, item in enumerate(labellist):
        if item == label:
            return valuelist[k]
