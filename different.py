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




def get_line_style(i):
    lsLineStyles = ['-','--','-.',':']
    return lsLineStyles[i]    


def plot_contour(fun,L=5, axis_points_count=50,levels_count=20):
    '''Plots contour of function fun of two variables.
    x belongs [-L,L], y belongs [-L,L]
    xlim and ylim define limits along x and y axis
    axis_points_count define number of points along each axis at which fun is evaluated
    levels_count defines the number of levels in the contour plot
    
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    


    x1 = np.linspace(-L, L, axis_points_count)
    x2 = np.linspace(-L, L, axis_points_count)

    xx1,xx2 = np.meshgrid(x1,x2)
    fv=np.vectorize(fun)
    yy = fv(xx1,xx2)
    plt.contourf(xx1, xx2, yy, levels = np.linspace(yy.min(),yy.max(),levels_count))
    plt.show()


def plot_bars_sorted(values, title_str='',step=1,figsize=(24,5)):
    '''Plots sorted(values) as bar plot
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
    
    plt.figure(figsize=figsize)
    D=len(values)
    plt.bar(left=np.arange(-0.4,D-0.4,1),height=sorted(values,reverse=True))
    plt.xticks( list(range(0,D,step)) );
    plt.xlim((-1,D+1))
    plt.title(title_str)