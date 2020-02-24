#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:44:46 2020

@author: pade
"""

import unittest
import network_class as nc
import numpy as np
from functools import partial


class TestNN(unittest.TestCase):
    """
    A couple of unittests for the methods and functions in the module 
    @network_class.
    """
    
    
    def setUp(self):
        # Set up a couple of test cases from the NN-class
        weights1 = [np.array([[1, 1]])]
        biases1 = [np.array([1])]
        weights2 = [np.array([[1, 0], [0, 1]]), np.array([[1, 0]])]
        biases2 = [np.array([1, 1]), np.array([1])]
        weights3 = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]
        biases3 = [np.array([1, 1]), np.array([1, 0])]
        self.network1 = nc.NN(weights1, biases1)
        self.network2 = nc.NN(weights2, biases2)
        self.network3 = nc.NN(weights3, biases3)
        
    def tearDown(self):
        self.network1
        self.network2
        self.network3


# -----------------------------------------------------------------------------
# Test methods    
# -----------------------------------------------------------------------------

    
    def test_get_dim(self):
        self.assertEqual(self.network1.get_dim(), [2, 1])
        self.assertEqual(self.network2.get_dim(), [2, 2, 1])
        self.assertEqual(self.network3.get_dim(), [2, 2, 2])
        
    def test_layer_out(self):
        np.testing.assert_array_equal(
                self.network2.layer_out(1, np.array([0, 0]), nc.sigmoid), 
                np.array([nc.sigmoid(1), nc.sigmoid(1)])
                )
      
    def test_netw_out(self):
        self.assertEqual(self.network1.netw_out(np.array([0, -1]), nc.sigmoid), 0.5)
        self.assertEqual(self.network2.netw_out(np.array([-1, -1]),nc.sigmoid),
                         nc.sigmoid(1.5)
                         )
        np.testing.assert_array_equal(
                 self.network3.netw_out(np.array([-1, -1]), nc.sigmoid),
                 np.array([nc.sigmoid(1.5), nc.sigmoid(0.5)])                 
                 )

# -----------------------------------------------------------------------------
# Test functions    
# -----------------------------------------------------------------------------

    def test_shapeForth(self):
        np.testing.assert_array_equal(
                nc.shapeForth(self.network1.weights, self.network1.biases)[0],
                np.array([1, 1, 1])
                )
        np.testing.assert_array_equal(
                nc.shapeForth(self.network1.weights, self.network1.biases)[1],
                np.array([2, 1])
                )

    def test_shapeBack(self):
        np.testing.assert_array_equal(
                nc.shapeBack([1, 1, 1] , [2, 1])[0],
                np.array([[[1, 1]]])
                )
        np.testing.assert_array_equal(
                nc.shapeBack([1, 1, 1] , [2, 1])[1],
                np.array([[1]])
                )
        
    def test_goal_function(self):
        data = [np.array([0, 0]), np.array([0, 1])]
        labels = [1, 0]
        target = partial(nc.myTarget, labellist=labels)
        self.assertAlmostEqual(
                nc.goal_function(np.array([1, 1, 1, 1, 1, 1]), [2, 2], data,
                labels, nc.sigmoid, target), ((1-nc.sigmoid(1))**2 
                + nc.sigmoid(1)**2 + nc.sigmoid(2)**2 + (1-nc.sigmoid(2))**2)/2)


# -----------------------------------------------------------------------------
# Test activation and target functions    
# -----------------------------------------------------------------------------
        
    def test_sigmoid(self):
        self.assertEqual(nc.sigmoid(0), 0.5)
        self.assertAlmostEqual(nc.sigmoid(1e10), 1)
        self.assertAlmostEqual(nc.sigmoid(-1e10), 0)
     
    def test_myTarget(self):
        np.testing.assert_array_equal(
                nc.myTarget('c', ['a', 'b', 'c', 'd']),
                np.array([0, 0, 1, 0])
                )      
  
      
if __name__ == "__main__":
    
    unittest.main()
