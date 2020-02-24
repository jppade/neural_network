#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 18:53:23 2020

@author: pade
"""

import numpy as np
import pickle
from classif_app.engine import optimization as opt
from classif_app.engine import network_class as nc


def delegate(path_data, col_label, point, threshold):

    """
    This is a simple customer delegate. It receives input from the customer 
    screen and calls the business logic. The return value is either the result
    of a successful request or an appropriate error message which are both sent
    back to the customer screen respectively.
    """
    
    # If the network was already trained, look for the corresponding file (try)
    # otherwise run the engine (except).
    try:
        with open('opt_network.pkl', 'rb') as input:
            opt_network = pickle.load(input)
            th, labellist = pickle.load(input)        
    except FileNotFoundError:
        # Compute network with optimized layer configuration
        opt_network, th, labellist = opt.optim_network(path_data, col_label, threshold)
        
        # Save the data for possible future requests
        with open('opt_network.pkl', 'wb') as output:
            pickle.dump(opt_network, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump((th, labellist), output, pickle.HIGHEST_PROTOCOL)
            
        
    # Determine the label of the point given by the user
    output = opt_network.netw_out(point, nc.sigmoid)
    distances = []
    for label in labellist:
        distances.append( np.linalg.norm(output - nc.myTarget(label, labellist)) )
    ind = distances.index(min(distances))
    label = labellist[ind]
    
    return {"The label is ": label,
            "The classification rate during training was ": "%.2f" % round(th,2)
            }
