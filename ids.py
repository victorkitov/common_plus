#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





def int2bool_indexes(int_inds,max_ind):
    '''Convert np.array of integers to boolean mask np.array'''
    X=np.zeros(max_ind,dtype=bool)
    X[int_inds]=True
    return X


