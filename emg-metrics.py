# -*- coding: utf-8 -*-
"""
------------------ OVERVIEW -------------------------------
EMG-Metrics

Created on Thu Jan 21 21:52:04 2016
@author: Jeff Barrett and Dan Viggiani


This module is designed to operate on electromyography (EMG)
data. 


Dependencies: numpy, a python package for efficient numeric
              calculations.





---------------- CHANGE LOG -----------------------------
Jan 21 2016 - Co-contraction index computations
Quantifying the degree of co-activation between a group of 
muscles. It goes hand-in-hand with an (expected) 2016 paper 
which elaborates on the technique being employed.


--------------------------------------------------------


"""
import numpy as np



# CCI : array(nxm) -> array(nx1)
"""
PURPOSE:  This function computes the co-contraction index
          of a group of muscles using the approach defined
          in Winter (1990)
INPUT:  M is an nxm numeric array where m represents the 
        number of muscles and n the number of frames
OUTPUT: The co-contraction index
TIME: O(m)
"""
def CCI(M):
    return len(M[0]) * np.min(M, axis=1) / np.sum(M, axis=1)



# mean_activity : array (nxm) -> array(nx1)
"""
PURPOSE: The time-varying mean-activity of a group of
         m-muscles with n-frames of data each
INPUT: M is an nxm array, where n is the number of frames
         and m is the number of muscles
OUTPUT: returns an nx1 array detailing the time-varying
        mean-activation
TIME: O(n)
"""
def mean_activity(M):
    return np.mean(M, axis=1)



# compute_path : array(nxm) -> array(nx1)
"""
PURPOSE: Computes the two-dimensional trajectory through
         Dan-Space.
INPUT: M is an nxm array where n is the number of frames
         and m is the number of muscles
OUTPUT: Returns the co-contraction index and the mean
        activity
TIME: O(n)
"""
def compute_path(M):
    return CCI(M), mean_activity(M)



# VBC2016metric : array(nxm), function(twople array -> number) -> number
"""
PURPOSE: Computes the co-activity metric as per Viggiani, Barrett and Callaghan
         2016
INPUT: M  : nxm array where n is the frame-count, and 
            m is the number of muscles
      phi : the Potential for computing the metric.
            this is a function that maps a co-contraction index and mean-activation
            to a heigh-value on the potential
"""
def VigBarCal2016metric(M, phi):
    x,y = compute_path(M)
    dz = np.sqrt(np.gradient(x,edge_order=2)**2 + np.gradient(y, edge_order=2)**2)
    return np.sum(phi(x,y) * dz)/len(M)























