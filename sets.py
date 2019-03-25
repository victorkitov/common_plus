#!/usr/bin/env python
# encoding: utf-8
'''
Common functions for operations with sets
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain,combinations
from itertools import combinations


def get_pairs(selected_features):
    '''For list of objects returns a list of tuple of all unordered pairs.
    Example: [f1,f2,f3]->[(f1,f2),(f1,f3),(f2,f3)]'''
    feature_pairs = list(combinations(selected_features,2))
    return feature_pairs


def powerset(iterable):
    '''Return set of all sets (as tuples).
     Example: powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)'''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def concat_elementwise(base,addition):
    '''
    concat_elementwise([1,2,3],[11,12,13]) returns:
    [[1, 2, 3, 11], [1, 2, 3, 12], [1, 2, 3, 13]]
    '''
    return [base+[addition[i]] for i in range(len(addition))]


def concat_cumulative(base,addition):
    '''
    concat_cumulative([1,2,3],[11,12,13]) returns:
    [[1, 2, 3, 11], [1, 2, 3, 11, 12], [1, 2, 3, 11, 12, 13]]
    '''
    return [base+addition[:i] for i in range(1,len(addition)+1)]


def rm_from_list(lst,values):
    '''Remove a sequence of values (values) from list (lst). List is modified inplace.'''
    for value in values:
        lst.remove(value)