from common.classes.Struct import Struct
import numpy as np
import sklearn as skl
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *



def first_true_ind(s,not_found_value=None):
    '''Return index of first True entry in boolean series object s. If all elements are False, returns -1.'''
    inds = find(s)
    if len(inds)==0:
        if not_found_value is None:
            return len(s)
        else:
            return not_found_value
    else:
        return inds[0]

def first_value_above_threshold(s,threshold=0):
    '''Return first entry>threshold. If all elements <=threshold, returns threshold.'''
    inds = nonzero(s.values>threshold)[0]
    if len(inds)==0:
        return threshold
    else:
        return s.values[inds[0]]

def first_val_above_threshold(s,threshold):
    '''Returns first value in sequence s above threshold.
    If none found returns threshold'''
    for e in s:
        if e>threshold:
            return e
    return threshold

def first_ind_above_threshold(s,threshold):
    '''Returns index of first value in sequence s above threshold.
    If none found returns threshold'''
    for i,e in enumerate(s):
        if e>threshold:
            return i
    return i+1

def last_val_abs_large(s,threshold):
    '''Returns first value in sequence s, whose absolute value is above threshold.
    If none found returns 0.'''

    for e in s:
        if (e>threshold) or (e<-threshold):
            return e
    return 0



def max_sign_changing_range(s,threshold=0,zero_terminates=False,zero_accounts_in_length=True):
    '''[-1,2,-3,5]-example of sign changing sequence. Length of such sequences is returned.'''
    s=[e for e in s if (e>threshold or e<-threshold)]
    if len(s)==1:
        return 1
    else:
        N=1
        for i in range(1,len(s)):
            if s[i]*s[i-1]==0:
                if zero_terminates:
                    break
                else:
                    N+=zero_accounts_in_length
            else:
                if s[i]*s[i-1]>0:
                    break
                else:
                    N+=1
    return N

def max_true_sequence_length(s):
    '''Return length of maximum zero sequence. Sequence is any iterable object. Sequence may be boolean or numeric.'''
    max_true_count=0
    true_count=0
    true_seq_now=False
    for e in s:
        if true_seq_now==True:
            if (e==True):
                true_count+=1
            else: # (e is False)
                max_true_count=max(true_count,max_true_count)
                true_seq_now=False
        else: # true_seq_now is False
            if (e==True):
                true_seq_now=True
                true_count=1
            else: # (e is Fa|lse)
                pass
    if true_seq_now==True:
        max_true_count=max(true_count,max_true_count)
    return max_true_count