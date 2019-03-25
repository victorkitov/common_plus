#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from common.functions import vec, row
from pylab import *
from common_plus.outliers import get_outlier_sels_within_classes
from common.visualize.colors import COLORS


   
def get_supervised_directions(X, y, clf, directions_count=inf):
    '''For design matrix <X> and vectors of outputs <y> get <directions_count> directions, 
    which are obtained as weights vector of classifier <clf> using SDA procedure 
    (get first class discriminating direction, then project data onto orthogonal complement, 
    find next Fisher LDA in the complement, then project to
    orthogonal complement of second direction (data will still be othogonal to the 1st direction) and so on).
    
    OUTPUT: directions - [D x directions_count] matrix of supervised directions
    
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''

    X=X.copy()
    D=X.shape[1]
    directions_count = min(directions_count, D)
    directions = eye(D) 
    directions = directions[:, :directions_count]
    
    if all(y==y[0]): # all object belong to one class - degenerate case
        return directions
    
    for dir_num in range(directions_count):
        clf.fit(X, y)
        d = clf.coef_.ravel() 
        for i in range(dir_num):  # ensure orthogonality of directions
            d = d - dot(d,directions[:,i])*directions[:,i]
        d = d/norm(d)
        '''
            try:
                #clf = LinearDiscriminantAnalysis(n_components=1) # was used in real experiments, artificial experiments.
                #clf = skl.linear_model.LogisticRegression(C=10**-6)
                #clf = skl.linear_model.LogisticRegressionCV(Cs=5)
                clf.fit(X, y)
                d = clf.coef_.ravel() 
                for i in range(dir_num):  # ensure orthogonality of directions
                    d = d - dot(d,directions[:,i])*directions[:,i]
                d = d/norm(d)
            except:
                A = hstack([directions[:,:dir_num], eye(D)])
                rotation,R = np.linalg.qr(A)
                #assert all( rotation[:,:dir_num]==directions[:,:dir_num] )
                #assert det(rotation)==1
                print('+',end='')
                directions = rotation[:,:directions_count]
                break
                '''
        directions[:,dir_num] = d
        X = X - vec(X.dot(d))*row(d) # TODO: make transformation from X to reduced space and solve classification in reduced space for numerical stability.
    return directions
    
    


def get_projections(X,directions):
    '''
    INPUT: 
        X: NxD feature matrix
        directions: [D x directions_count] matrix of directions on which to project. They should be vectors on unit norm.
    OUTPUT:
        [N x directions_count] matrix of projections of feature matrix <X>
    
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''
    
    assert X.shape[1]==directions.shape[0]
    assert all(around(sum(directions**2,axis=0),4)==1) # verify that all directions are unit-norm
    return X.dot(directions) 
    
    
    



def visualize_2D(X_train,Y_train, clf, dataset_name, outliers_fraction = 0.05, support_fraction=0.9, cum_explained_ratio=0.75):  
    '''Show in 2 dimensions the data. Totally 4 graphs are shown.
    1) Original data in the first and second principal component    
    2) Original data in the first and second principal component
    3) Data filtered from outliers in the first and second principal component
    4) Data filtered from outliers in the first and second LDA discriminating component  
    
    Outlier filtering is based on fitting Gaussian distribution onto points within each class. 
    Within each class 5% of least probable points are excluded. Gaussian parameters are estimated in a robust ways using 90% of data.
    To make the data undegenerate it is projected onto first K principal component, where K is selected so that 75% of variance of 
    original data is explained.
    
    <name> is dataset name.
    '''


    pca=skl.decomposition.PCA()
    X_train_pca = pca.fit_transform(X_train)
    directions = get_supervised_directions(X_train,Y_train,clf,2)
    X_train_sda = get_projections(X_train,directions)    
    
    
    f,axes = subplots(1,4,figsize=[18,6])
    
    sca(axes[0])
    title('%s:\n PCA (%.3f)' %(dataset_name, norm(X_train_pca)/norm(X_train)))
    scatter(X_train_pca[:,0], X_train_pca[:,1], c=[COLORS[y] for y in Y_train])

    sca(axes[1])
    title('%s:\n SDA (%.3f)' %(dataset_name, norm(X_train_sda)/norm(X_train)))
    scatter(X_train_sda[:,0], X_train_sda[:,1], c=[COLORS[y] for y in Y_train])    
    
    outlier_sels = get_outlier_sels_within_classes(X_train, Y_train, outliers_fraction=outliers_fraction, 
                                                                     support_fraction=support_fraction, 
                                                                     cum_explained_ratio=cum_explained_ratio)
    
    X_train = X_train[~outlier_sels,:]
    Y_train = Y_train[~outlier_sels]  
    
    (X_train)
    
    X_train_pca = pca.fit_transform(X_train)
    directions = get_supervised_directions(X_train,Y_train,2)
    X_train_sda = get_projections(X_train,directions)    
    
    
    sca(axes[2])
    title('%s-no outliers:\n PCA (%.3f)' %(dataset_name, norm(X_train_pca)/norm(X_train)))
    scatter(X_train_pca[:,0], X_train_pca[:,1], c=[COLORS[y] for y in Y_train])

    sca(axes[3])
    title('%s-no outliers:\n SDA (%.3f)' %(dataset_name, norm(X_train_sda)/norm(X_train)))
    scatter(X_train_sda[:,0], X_train_sda[:,1], c=[COLORS[y] for y in Y_train])    
    
    show()