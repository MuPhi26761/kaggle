#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 21:33:58 2025

@author: sachadegoix
"""

import lightgbm as lgb
from sklearn.metrics import mean_squared_error


def train_lgbm(X_train, y_train, X_val, y_val):
    parms = {
        'objective': 'regression',
        'metric':'rmse',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'seed': 42
        }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        params, dtrain, num_boost_round=3000,
        valid_sets=[dtrain, dval],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    return model
    