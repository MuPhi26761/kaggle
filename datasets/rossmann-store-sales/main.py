#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 22:15:48 2025

@author: sachadegoix
"""

from features import features
from processing import load_data, time_features
from model import train_lgbm


# LOAD + PREPROCESS
df = load_data()
df = time_features(df)


# Split Time-based
split_date = df['Date'].max() - pd.Timedelta(weeks=6)

train_data = df[df['Date'] < split_date].copy()
val_data = df[df['Date'] >= split_date].copy()

# Feature engineering

train_data = features(train_data)
val_data = features(val_data) # Leakage warning


features_cols = ['Store','CompetitionDistance','Promo','year','month','day_of_week','avg_sales_store_day']
target_col = 'log_sales'

X_train = train_data[features_cols]
y_train = train_data[target_col]
X_val = val_data[features_cols]
y_val = val_data[target_col]


# TRAIN
model = train_lgbm(X_train,y_train,X_val,y_val)

# Evaluate (Inverse log transform)
preds_log = model.predict(X_val)
preds_real = np.expm1(preds_log)
y_val_real = np.expm1(y_val)