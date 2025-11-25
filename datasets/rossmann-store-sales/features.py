#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 21:53:01 2025

@author: sachadegoix
"""

def features(df):
    df = df.sort_values(['Store','Date'])
    
    sales_per_store_day = df.groupby(['Store','day_of_the_week'])['Sales'].mean().reset_index()
    sales_per_store_day.rename(columns = {'Sales':'avg_sales_store_day'}, inplace = True)
    df = pd.merge(df, sales_per_store_day, on = ['Store','day_of_the_week'], how = 'left')
    
    return df

