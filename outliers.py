#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.decomposition import PCA
from pylab import *
from common.visualize.colors import COLORS


def get_outlier_sels(X, outliers_fraction=0.05, support_fraction=None, cum_explained_ratio=0.95):
    pca=PCA()
    X_pca = pca.fit_transform(X)
    ind = find(cumsum(pca.explained_variance_ratio_)>cum_explained_ratio)[0]
    print('Input dimensionality %d converted to %d (which explain >%.3f of variance)' %(X.shape[1],ind+1,cum_explained_ratio))

    X_pca = X_pca[:,:ind+1]
    outliner_detector = skl.covariance.EllipticEnvelope(contamination=outliers_fraction, support_fraction=support_fraction)
    outliner_detector.fit(X_pca)
        
    sels = outliner_detector.predict(X_pca)
    outlier_sels = (sels==-1)
    return outlier_sels


def get_outlier_sels_within_classes(X,Y, outliers_fraction=0.1, support_fraction=None, cum_explained_ratio=0.95):
    outlier_sels = zeros(len(Y),dtype=bool)
    unique_y = unique(Y)
    for y in unique_y:
        sels = get_outlier_sels(X[Y==y], outliers_fraction=0.1, support_fraction=support_fraction, cum_explained_ratio=cum_explained_ratio)
        outlier_sels[ find(Y==y)[sels] ] = True
        
    return outlier_sels
    
    
    
    
if __name__=='__main__':

    # get_outlier_sels demo

    mean = array([0,0])
    cov = array([[1,0.9], [0.9,1]])
    N=100

    X = numpy.random.multivariate_normal(mean, cov, N)
    X[0,:]=[-2,2]

    outlier_sels = get_outlier_sels(X, outliers_fraction=0.2)
    scatter(X[:,0], X[:,1], c=[COLORS[y] for y in outlier_sels])  



    # get_outlier_sels_within_classes demo

    mean = array([0,0])
    cov = array([[1,0.9], [0.9,1]])
    N=100

    X1 = numpy.random.multivariate_normal(mean, cov, N)
    X2 = numpy.random.multivariate_normal(mean+[-10,10], cov, N)
    X=vstack([X1,X2])
    Y=[0]*N+[1]*N

    #outlier_sels = get_outlier_sels(X, outliers_fraction=0.2)

    outlier_sels = get_outlier_sels_within_classes(X,Y, outliers_fraction=0.2)
    scatter(X[:,0], X[:,1], c=[COLORS[y] for y in outlier_sels])  