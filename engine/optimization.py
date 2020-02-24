#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:42:57 2020

@author: pade
"""

import numpy as np
from functools import partial
from classif_app.engine import network_class as nc
from classif_app.engine import persistence as ps
    

class Training:
    
    """
    Train a network with dimensions given by the list @layers with the data 
    given by @training_set.
    """
    
    def __init__(self, training_set, layers, all_labels):
        self.training_set = training_set
        self.layers = layers
        self.all_labels = all_labels
    
    
    def prep_train(self):
        """
        Returns the training data and labels in the format used by the Class
        "network class".
        """
        training_labels = [key for key in self.training_set for value in
                           self.training_set[key]]
        training_data = []
        for key in self.training_set:
            training_data += self.training_set[key]
        return training_data, training_labels
     
    def train_network(self):
        """
        Returns a network from the NN class trained with the training set given
        by the Training class attribute @training_set.
        """
        # Initialize network from network class...
        weights, biases = nc.init_rand(self.layers)
        myNetwork = nc.NN(weights, biases)
        
        # ...initialize the target function for the training...
        target_part = partial(nc.myTarget, labellist=self.all_labels)
        
        # ...and run gradient descent for given network configuration
        myNetwork.train_netw(self.prep_train()[0], self.prep_train()[1],
                                                       nc.sigmoid, target_part)
        return myNetwork
    
            
class Testing:
    
    """
    Testing class for a trained neural network from Training class.
    """
    
    def __init__(self, testing_set, trained_network, all_labels):
        
        self.testing_set = testing_set
        self.network = trained_network
        self.all_labels = all_labels
        
        
    def test_network(self):
        """
        Test network with given testing set.
        """
        # Get trained network and labels from Training class
        myNetwork = nc.NN(self.network.weights, self.network.biases)
        
        # Iterate through testing set and collect boolean values for (successful)
        # classification in the list @cl_success
        cl_success = []
        for key in self.testing_set:
            for point in self.testing_set[key]:
                output = myNetwork.netw_out(point, nc.sigmoid)
                # Check whether point is classified correctly
                distances = [np.linalg.norm(output - nc.myTarget(lab, self.all_labels)) for
                            lab in self.all_labels]
                if self.all_labels[distances.index(min(distances))] == key:
                    cl_success.append(True)
                else:
                    cl_success.append(False)
        return cl_success


def optim_network(data_path, col_label, threshold):
    """
    Returns the optimal network configuration for given training data at 
    @data_path such that the success rate is above @threshold. In case such a 
    configration is not found, it returns the configuration with the highest
    (classification) success rate.
    @col_label is an integer that designates the column number of the label in 
              the data set.
    @threshold is a number between 0 and 1. If possible, a configuration with 
              success rate above threshold is returned.
    """
    
    # Request data from the persistence layer
    myData = ps.Data(data_path, col_label)
    training_set, testing_set = myData.get_training_set()
    
    # Compute classification rate for all network configurations
    all_layers = myData.get_configs()
    num_configs = len(all_layers)
    # Set actual best value of classification rate and counter for configurations
    ths = 0
    count = 0
    
    # Train and test until either sufficient configuraiton is found or no con-
    # figurations are left
    all_labels = myData.labels
    while ths < threshold and count < num_configs:
        # Train and test
        myTraining = Training(training_set, all_layers[count], all_labels)
        myNetw = myTraining.train_network()
        myTesting = Testing(testing_set, myNetw, all_labels)
        success = myTesting.test_network()
        success_rate = sum(success)/len(success)
        
        # Check quality of configuration
        if success_rate > threshold:
            return myNetw, success_rate, all_labels
        elif success_rate <= threshold and success_rate > ths:
            # Temporarily best network
            ths = success_rate
            best_network = myNetw
            count += 1
        else:
            count += 1
    return best_network, ths, all_labels  