#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from common.functions import vec, row
from pylab import *




def get_category_ids_encoding(x, return_correspondence=False):
    '''Function returns a pandas series consisting of ids, corresponding to objects in input pandas series x
    Example:
    get_series_ids(pd.Series(['a','a','b','b','c'])) returns pd.Series([0,0,1,1,2], dtype=int)
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.
    '''
    values = np.unique(x)
    values2nums = dict( zip(values,list(range(len(values)))) )
    if return_correspondence:
        return x.replace(values2nums), values2nums
    else:
        return x.replace(values2nums)

    
