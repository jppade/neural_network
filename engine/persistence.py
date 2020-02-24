#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:31:30 2020

@author: pade
"""

import csv
import numpy as np
import random


class Data:
    
    """
    This class along with the auxilliary functions acts as an interface between
    the data layer and the business layer.    
    """
    
    def __init__(self, data_path, col_label):
        
        assert(type(col_label) == int)
        self.data = get_data(data_path, col_label)[0]
        self.labels = get_data(data_path, col_label)[1]
        self.col_label = col_label


# -----------------------------------------------------------------------------
#        Methods
# ----------------------------------------------------------------------------- 
    
    def get_dimensions(self):
        """
        Returns the integer sizes of input and output layer of any neural 
        network acting on the given data.
        """
        dim_in = len(self.data[0]) - 1
        dim_out = len(self.labels)
        return dim_in, dim_out
     
    def get_configs(self):
        """
        !! This still has to be done !!
        Returns a list of configurations for possible neural networks acting on
        the given data.
        """
        # This is just a placeholder for the cofigurations list (or dict?)
        configs = [[self.get_dimensions()[0], 3, self.get_dimensions()[1]],
                   [self.get_dimensions()[0], 4, self.get_dimensions()[1]],
                   [self.get_dimensions()[0], 4, 3, self.get_dimensions()[1]]
                   ]
        return configs
             
    def get_training_set(self):
        """
        Get a training set along with a testing set. Both are returned as dicts
        where each key is a label and the corresponding value is a list of data
        points with this label.
        """
        print("training-set getter is called")
        
        # !!! For now, we restrict the training/testing set to 100 data points
        # In case the minimization of the goal function can be done faster, we
        # could increase this number (up to the full data set) (see commented
        # code below
        num_data = len(self.data)
        num_training = 66
        # Make a disjoint random choice of indices for training and testing 
        sample1 = random.sample(range(0, num_data), num_training)
        set_complement = set(range(0, num_data)).difference(set(sample1))
        sample2 = random.choices(list(set_complement), k=33)
        
#         # Use 2/3 of the data to train and the rest to test
#        num_data = len(self.data)
#        num_training = round(2*num_data/3)
#        
#        # Make a disjoint random choice of indices for training and testing 
#        sample1 = random.sample(range(0, num_data), num_training)
#        set_complement = set(range(0, num_data)).difference(set(sample1))
#        sample2 = list(set_complement)
#        
        # Get training and testing data with the random choice
        data1 = [self.data[k] for k in sample1]
        data2 = [self.data[k] for k in sample2]
        
        
        training_data = {}
        testing_data = {}
        for label in self.labels:
            training_data[label] = [np.array(row[:self.col_label-1] + row[self.col_label:])
                             for row in data1 if row[self.col_label - 1] == label]
            testing_data[label] = [np.array(row[:self.col_label-1] + row[self.col_label:])
                             for row in data2 if row[self.col_label - 1] == label]
        
        return training_data, testing_data
    
    
# -----------------------------------------------------------------------------
#        Functions
# -----------------------------------------------------------------------------

def get_data(data_path, col_label):
    """
    Returns data as list of lists, each list entry correponsing to a data point
    with corresponding label at position @col_label - 1.
    Also returns a list of all labels. 
    """
    print("data getter is called")
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        data = []
        for row in reader:
            aux = []
            for entry in row:
                try:
                    aux.append(float(entry))
                except ValueError:
                    aux.append(entry)
            data.append(aux)
            
    # Get list of all possible distinct  labels
    labels = list(set([row[col_label - 1] for row in data]))
  
    return data, labels

    

    
        