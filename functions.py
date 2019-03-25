#!/usr/bin/env python
# encoding: utf-8
'''
Common unclassified functions
'''

import numpy as np
import pandas as pd
from itertools import chain
from collections import deque
import sys
from datetime import datetime
        
        
        

def all_ints(X):
    '''Returns True if sequence X consists only of integer values,
    False otherwise'''
    try:
        for x in X:
            if x!=int(x):
                return False
    except: # could not transform x to int(x)
        return False
    return True




def nans(shape):
    X=np.empty(shape)
    X[:]=np.nan
    return X






def get_year_sels(date_series,year):
    '''
    Return boolean index selection of dates belonging to a given year
    date_series - pandas datetime series
    '''
    sels = (date_series>=pd.Timestamp('%s-01-01' % year)) & (date_series<=pd.Timestamp('%s-12-31' % year))
    #if isinstance(sels,pd.Series):
    #    sels=sels.values
    return sels



def columns2float32(X):
    for col in X.columns:
        if X[col].dtype==np.float64:
             X[col] = X[col].astype(np.float32)

def columns2float16(X):
    for col in X.columns:
        if (X[col].dtype==np.float64) or (X[col].dtype==np.float32):
             X[col] = X[col].astype(np.float16)

    
    
def convert_td64_dt(dt64):
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(ts).date()


def total_size(o, handlers={}):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    ##### Example call #####
    d = dict(a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
    print(total_size(d, verbose=True))
    """

    dict_handler = lambda d: chain.from_iterable(list(d.items()))
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)       # estimate sizeof object without __sizeof__

    def getsizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in list(all_handlers.items()):
            if isinstance(o, typ):
                s += sum(map(sys.getsizeof, handler(o)))
                break
        return s

    return sys.getsizeof(o)

    
    
def vars_size():
    # show all variables ordered by their size
    glob=globals()
    variables = array([k for k in glob.keys()])
    sizes = zeros(len(variables))
    for i,variable in enumerate(variables):
        sizes[i] = sys.getsizeof(glob[variable])

    inds = argsort(sizes)[::-1]
    sizes=sizes[inds]
    variables=variables[inds]

    cum_size=0
    for i in range(20):
        cum_size+=sizes[i]
        print('%30s %.3f %.3f' % (variables[i],sizes[i]/1024/1024,cum_size/1024/1024 ) )

    print('Total size: %.3f'% (sum(sizes)/1024/1024) )
    
    
