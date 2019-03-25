#!/usr/bin/env python
# encoding: utf-8
'''
Common math functions
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# all_num(pd.Series([1,2]))

def scalar(x,y):
    # scalar product of 2 numpy vectors
    return np.dot(x,y)

def cov(x,y):
    # covariance between two numpy vectors
    return np.cov(x,y)[0,1]

def corr(x,y):
    # correlation between two numpy vectors
    c=np.cov(x,y)
    return c[0,1]/np.sqrt( c[0,0]*c[1,1] )



def is_semipos_def(X):
    '''is_semipos_def(X) checks if matrix x is positive semidefinite or not
    Input: matrix x
    Output: True if x is positive semidefinite, False otherwise.
    Example: is_semipos_def([[-1,10],[0,1]])'''
    return np.all(np.linalg.eigvals(X) >= 0)


def is_convex(fun, D, M=100, trials_count=10000):
    '''Check convexity of function fun, depending on D-dimensional numpy.array by making trials trial_count times taking x_i from uniform(-M,M,size=D)'''
    for n in range(trials_count):
        x1=np.random.uniform(-M,M,size=D)
        x2=np.random.uniform(-M,M,size=D)
        a=np.random.uniform()

        if not fun(a*x1+(1-a)*x2) <= a*fun(x1)+(1-a)*fun(x2):
            print('Function convexity: False')
            print('fun(a*x1+(1-a)*x2)={}, a*fun(x1)+(1-a)*fun(x2)={}'.format( fun(a*x1+(1-a)*x2), a*fun(x1)+(1-a)*fun(x2) ))
            print('x1={}, x2={}'.format( x1, x2 ))
            return False

    print('Function convexity: True')
    return True

#print is_convex(lambda x:x[0]**2+x[1]**2-x[2]**2,3,)
