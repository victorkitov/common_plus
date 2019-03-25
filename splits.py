#!/usr/bin/env python
# encoding: utf-8
'''
Common functions for splitting samples into train and test subsamples.
'''

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def shuffle(df,seed=None):
    '''Not inplace!'''
    if seed:
        np.random.seed(seed)
    # shuffle dataframe by rows preserving original index
    index_copy = df.index.copy()
    index = list(df.index)
    assert all(np.unique(index)==np.arange(len(df)))
    np.random.shuffle(index)
    df = df.loc[index]
    df.index = index_copy
    return df


def stratified_train_test_split(data,class_column,train_proportion=0.7):
    '''splits DataFrame data into two dataframes train and test so that
       1) len(train)=len(data)*proportion, len(test)=len(data)*(1-proportion)
       2) class distribution is preserved, where class is a categorical variable,
          specified in column class_column of data.'''

    train = pd.DataFrame(columns=data.columns)
    test = pd.DataFrame(columns=data.columns)
    for col in data.columns:
        train[col] = train[col].astype(data[col].dtype)
        test[col] = test[col].astype(data[col].dtype)

    segments = [pd.DataFrame(group[1]) for group in data.groupby(class_column, sort = False)]
    for segment in segments:
        segment = shuffle(segment)
        N=len(segment)
        N_split = int(N*train_proportion)
        train_subsegment = segment[:N_split]
        test_subsegment = segment[N_split:]
        train = train.append(train_subsegment, ignore_index=True)
        test = test.append(test_subsegment, ignore_index=True)

    return (shuffle(train), shuffle(test))



def get_train_test_sels(y,train_proportion=2/3,seed=None):
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and
    test sets are preserved (stratified sampling).
    '''
    if seed is not None:
        np.random.seed(seed)
    y=np.array(y)
    train_sels = np.zeros(len(y),dtype=bool)
    test_sels = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))

        train_sels[value_inds[:n]]=True
        test_sels[value_inds[n:]]=True

    return train_sels,test_sels
