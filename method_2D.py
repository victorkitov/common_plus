#!/usr/bin/env python
# encoding: utf-8

from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn import tree
from sklearn.metrics import accuracy_score
from common.iteration import piter
from common.functions import vec, normalize
from pylab import *




def plot_predictions(clf, X_train, Y_train, train_clf=True, n=50, interpolation = 'nearest', offset=0.05):
    '''Plots decision regions for classifier clf trained on design matrix X=[x1,x2] with classes y.
    n is the number of ticks along each direction
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
    
    x1, x2 = X_train[:,0],X_train[:,1]
    if train_clf:
        clf.fit(X_train, Y_train)
    
    margin1 = offset*(x1.max()-x1.min())
    margin2 = offset*(x2.max()-x2.min())
        
    # create a mesh to plot in
    x1_min, x1_max = x1.min() - margin1, x1.max() + margin1
    x2_min, x2_max = x2.min() - margin2, x2.max() + margin2
    
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, n),
                         np.linspace(x2_min, x2_max, n))

    X_test = hstack( [vec(xx1.ravel()), vec(xx2.ravel())] )    
    Y_test = clf.predict(X_test)
    yy = Y_test.reshape(n,n)
    
    figure()
    img = plt.imshow(yy, extent=(x1_min, x1_max, x2_min, x2_max), interpolation=interpolation, origin='lower', alpha=0.5)
    colorbar()
    
    C = normalize(Y_train)
    scatter(x1, x2, facecolor=Y_train)
    plt.axis([x1_min, x1_max, x2_min, x2_max])


    
def plot_decision_regions(X, y, clf, n=30, interpolation = 'gaussian', margin1=None, margin2=None):
    '''Plots decision regions for classifier clf trained on design matrix X=[x1,x2] with classes y.
    n is the number of ticks along each direction.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
    
    x1, x2 = X[:,0],X[:,1]
    clf.fit(hstack([vec(x1),vec(x2)]), y)
    
    if margin1==None:
        margin1 = 0.05*(x1.max()-x1.min())
    if margin2==None:
        margin2 = 0.05*(x2.max()-x2.min())
        
    # create a mesh to plot in
    x1_min, x1_max = x1.min() - margin1, x1.max() + margin1
    x2_min, x2_max = x2.min() - margin2, x2.max() + margin2
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, n),
                         np.linspace(x2_min, x2_max, n))

    X = hstack( [vec(xx1.ravel()), vec(xx2.ravel())] )    
    Y = clf.predict(X)
    yy = Y.reshape(len(xx1),len(xx2))

    colors = [[0,0.5,1],[1,0,0],[0,1,0.5],[0.7,0.7,0.7]]

    cmap = mpl.colors.ListedColormap(colors)
    bounds=list(range(len(colors)))+[len(colors)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    img = plt.imshow(yy, extent=(x1_min, x1_max, x2_min, x2_max), interpolation=interpolation, origin='lower', cmap=cmap, norm=norm, alpha=0.5)
    
    scatter(x1, x2, facecolor=[colors[int(y1)] for y1 in y])
    plt.axis([x1_min, x1_max, x2_min, x2_max])

    
    
def plot_2features_class_scatter(feature1,feature2,Z,train_sels,filter_sels=None,target='y',figsize=(8,8)):
    '''Plot objects from subset of Z. Objects are colored with different colors, representing class (given in Z.y) in axes
    Z.feature1 and Z.feature2.

    Subset is given by train_sels & filter_sels.
    If filter_sels==None, filtering is done only using train_sels.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    

    if filter_sels is None:
        filter_sels = np.ones(sum(train_sels),dtype=bool)

    xx1 = Z.loc[train_sels,feature1]
    xx2 = Z.loc[train_sels,feature2]

    if all_ints(xx1):
        xx1 = xx1+0.4*np.random.rand(len(xx1))-0.2
    if all_ints(xx2):
        xx2 = xx2+0.4*np.random.rand(len(xx2))-0.2

    #(Z.Date[train_sels]>=pd.Timestamp('2000-01-01')) & (Z.Date[train_sels]<=pd.Timestamp('2016-01-01'))
    yy = Z.loc[train_sels,target].values

    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    ax.scatter(xx1.values[(yy==False) & filter_sels],xx2.values[(yy==False) & filter_sels],s=3,c='b',linewidths=0)
    ax.scatter(xx1.values[(yy==True) & filter_sels],xx2.values[(yy==True) & filter_sels],s=3,c='r',linewidths=0)
    plt.xlabel(xx1.name)
    plt.ylabel(xx2.name)

