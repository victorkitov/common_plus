from pylab import *
import numpy as np
import pandas as pd


def knn_linear_weights(dists):
    '''Linear dacaying weights for K-NN custom weight specification.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''
    
    min_dist = np.min(dists)
    max_dist = np.max(dists)
    if min_dist==max_dist:
        return ones(dists.shape)
    else:
        return (max_dist-dists)/(max_dist-min_dist)   