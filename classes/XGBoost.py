#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import xgboost as xgb
from common.functions import vec


class XGBoost(object):

    def __init__(self,**params):
        self.num_round = params.pop('num_round',10)
        self.eval_prop = params.pop('eval_prop',None)
        self.early_stopping_rounds = params.pop('early_stopping_rounds',20)
        self.params=params
        self.clf = None
        self.classes_ = np.array([0,1])

    def fit(self, X, y):
        if self.eval_prop is None:
            dtrain = xgb.DMatrix(X, y)
            self.clf = xgb.train( self.params, dtrain, self.num_round);
        else:
            ones_count = int(self.eval_prop*len(X))
            zeros_count = len(X)-ones_count

            sels=np.concatenate([np.ones(ones_count),np.zeros(zeros_count)]).astype(bool)
            randomizer = np.random.RandomState(0)
            randomizer.shuffle(sels)
            dtrain = xgb.DMatrix(X[~sels,:], y[~sels])
            deval = xgb.DMatrix(X[sels,:], y[sels])
            evallist  = [(deval,'eval')]
            self.clf = xgb.train( self.params, dtrain, self.num_round, evals=evallist, early_stopping_rounds=self.early_stopping_rounds );
            self.best_iteration = self.clf.best_iteration
            self.clf = xgb.train( self.params, dtrain, self.best_iteration);
            #print '(self.num_round=%s, self.best_iteration=%s)' %(self.num_round, self.best_iteration)

    def set_params(self,**params):
        if 'num_round' in params:
            self.num_round = params.pop('num_round')
        if 'eval_prop' in params:
            self.eval_prop = params.pop('eval_prop')
        if 'early_stopping_rounds' in params:
            self.early_stopping_rounds = params.pop('early_stopping_rounds')
        self.params.update(params)

    def get_params(self):
        all_params = self.params.copy()
        all_params['num_round'] = self.num_round
        all_params['eval_prop'] = self.eval_prop
        all_params['early_stopping_rounds'] = self.early_stopping_rounds
        return all_params

    def predict(self,X):
        if self.eval_prop is None:
            return self.clf.predict( xgb.DMatrix(X) )
        else:
            return self.clf.predict( xgb.DMatrix(X), ntree_limit=self.best_iteration )


    def predict_proba(self,X):
        if self.eval_prop is None:
            p_hat = self.clf.predict( xgb.DMatrix(X) )
        else:
            p_hat = self.clf.predict( xgb.DMatrix(X), ntree_limit=self.best_iteration )
        return np.concatenate([vec(1-p_hat),vec(p_hat)],axis=1)