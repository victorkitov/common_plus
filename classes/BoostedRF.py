#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


class BoostedRF(object):

    def __init__(self,**kwargs):
        self.max_improvement_count = kwargs.pop('max_improvement_count',2)
        self.alpha0 = kwargs.pop('alpha0',0.1)
        self.penalty_decay = kwargs.pop('penalty_decay',0.5)
        self.max_failures_num = kwargs.pop('max_failures_num',2)
        self.min_alpha = kwargs.pop('min_alpha',0.0001)
        self.display = kwargs.pop('display',False)

        assert 0<self.penalty_decay<1,'penalty_decay={} not in (0,1) interval!'.format(self.penalty_decay)
        self.clf = RandomForestClassifier(**kwargs)


    def fit(self,X,y,sample_weight=None):

        X=np.asarray(X)
        y=np.asarray(y).ravel()
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        good_accuracy = 0
        alpha = self.alpha0
        improvement_num=0
        failures_num=0
        weights = sample_weight.copy()

        while improvement_num<self.max_improvement_count:
            self.clf.fit(X,y,weights)
            all_pred_probs = self.clf.predict_proba(X)
            pred_inds = all_pred_probs.argmax(axis=1)
            pred_probs = np.asarray([all_pred_probs[i,j] for i,j in enumerate(pred_inds)]) # probabilities of predicted classes

            true_inds = np.nonzero(y[:,np.newaxis]==self.clf.classes_)[1]
            true_probs = np.asarray([all_pred_probs[i,j] for i,j in enumerate(true_inds)]) # probabilities of true classes

            f = self.clf.classes_[pred_inds]
#            print f
#            print y
            accuracy = np.sum((f==y)*sample_weight)/np.sum(sample_weight)

            accuracy_improvement = accuracy-good_accuracy
            if accuracy_improvement>=0:
                good_weights = weights.copy()
                good_accuracy = accuracy
                weights = weights*np.exp((pred_probs-true_probs)*alpha)   # (pred_probs-true_probs) - measure of error
                improvement_num += 1
                failures_num=0
            else: # no improvement
                alpha = alpha*self.penalty_decay
                weights = good_weights*np.exp((pred_probs-true_probs)*alpha)
                failures_num += 1
                if failures_num>=self.max_failures_num or abs(alpha)<self.min_alpha:
                    return

            if self.display:
                print('improvement_num {}) failures_num {}) accuracy {:.4f}, accuracy_improvement={:.4f}, alpha={:.4f}'.format(
                                                                                            improvement_num,
                                                                                            failures_num,
                                                                                            accuracy,
                                                                                            accuracy_improvement,
                                                                                            alpha))

    def predict_proba(self,X):
        return self.clf.predict_proba(X)

    def predict(self,X):
        return self.clf.predict(X)

    def get_params(self,deep=True):
        params = self.clf.get_params(deep)
        params.update({ 'max_improvement_count':self.max_improvement_count,
                        'alpha0':self.alpha0,
                        'penalty_decay':self.penalty_decay,
                        'max_failures_num':self.max_failures_num,
                        'min_alpha':self.min_alpha,})
        return params

    def set_params(self,**kwargs):
        self.max_improvement_count = kwargs.pop('max_improvement_count',2)
        self.alpha0 = kwargs.pop('alpha0',0.1)
        self.penalty_decay = kwargs.pop('penalty_decay',0.5)
        self.max_failures_num = kwargs.pop('max_failures_num',2)
        self.min_alpha = kwargs.pop('min_alpha',0.0001)

        assert 0<self.penalty_decay<1,'penalty_decay={} not in (0,1) interval!'.format(self.penalty_decay)
        self.clf = RandomForestClassifier(**kwargs)
        return self


    @property
    def classes_(self):
        return self.clf.classes_