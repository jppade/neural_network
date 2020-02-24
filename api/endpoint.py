#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:25:39 2020

@author: pade
"""

import sys
sys.path.append('../')
import os
from classif_app.engine import delegate as dl

# Set path at which the data uploaded by the user is saved
data_path = "data/uploaded_data/myData.csv"

    
def data(myFile):
    """
    Stores @myFile at the above path @data_path. Deletes previously trained
    network data in case it exists.
    """
     # Delete trained network from previous run in case it exists
    try:
        print("Deleting previously trained network...")
        os.remove("opt_network.pkl")
    except OSError:
        print("No trained network stored.")
    
    # Store uploaded data file
    try:
        myFile.save(data_path)
        return "Successful upload"
    except FileNotFoundError:
        return "File not found."
    
    
def network_engine(col_label, threshold, point):
    """
    Calls the delegate with the parameters from the user screen.
    """
    return dl.delegate(data_path, col_label, point, threshold)
    
