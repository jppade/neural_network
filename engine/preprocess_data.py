#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:47:25 2020

@author: pade
"""
import pandas as pd

class Preprocess:

    """
    This class provides two methods to prepare data for the business layer.
    !! It has yet to be included. !!
    """

    def __init__(self, path):
        self.path = path


    def float_representation(self):
        """
        For a csv-file at path, this function corrects the representation of floats
        written as an object such as 3,14 to the correct float representation 3.14.
        """
        data = pd.read_csv(self.path)
        types = data.dtypes
        for i,type in enumerate(types):
            if type == "object":
                try:
                    data[data.columns[i]] = [float(num.replace(',','.')) for \
                                            num in data[data.columns[i]]]
                except ValueError:
                    pass
        data.to_csv(self.path, index=False)
            
    def strip_strings(self):    
        """
        Strip the strings in a csv-file in order to get rid of blank spaces at the
        end (or the beginning) of a string.
        """
        data = pd.read_csv(self.path)
        types = data.dtypes
        for i,type in enumerate(types):
            if type == "object":
                try:
                    data[data.columns[i]] = [strng.strip() for strng in
                                           data[data.columns[i]]]
                except AttributeError:
                    pass
        data.to_csv(self.path, index=False)


if __name__ == '__main__':
    
    myPath = "path_to_data"
    for path in myPaths:
        path_handle = Preprocess(path)
        path_handle.strip_strings()
        path_handle.float_representation()
