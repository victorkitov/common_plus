#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def generate_offset_normal_data(D,M,K=None,count=10,sigma=0.9):
    '''Generate M class dataset with X normally distributed with eye covariance matrix with sigma^2 on diagonal
    maen(X)=0 everywhere except 1 for particular features subset fixed for each class.'''

    if K is None:
        K=int(D/M)

    X=np.zeros( (0,D) )
    Y=[]
    cov = sigma*np.eye(D)

    for m in range(M):
        mean=np.zeros(D)
        #mean=-np.ones(D)*K/(D-K)
        mean[m*K:m*K+K] = 1
        #mean[:] = m*np.random.uniform(0,1,D)
        X = np.vstack( (X, np.random.multivariate_normal(mean,cov,count) ) )
        Y+=count*[m]

    Y=np.array(Y)
    return (X,Y)
