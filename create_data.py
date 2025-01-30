#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:54:17 2024

@author: k.g.vanwiggen
"""
import numpy as np
import pandas as pd
from scipy.stats import wishart, multivariate_normal

def create_df_lag(df, lags, length, rows, columns):
    df_lag = np.full((length, rows, columns), np.nan)
    for t in range(length-lags):
        df_lag[t,:,:] = df[t+lags,:,:]
    df = df[0:length-lags,:,:]
    df_lag = df_lag[0:length-lags,:,:]
    return df, df_lag

def create_data_sim(length, lags, row, col, beta, Et):
    data = np.full((length+lags+1, row*col), np.nan)
    data[0,:] = Et[0,:]
    for t in range(lags, length+lags+1):
        data[t,:] = beta @ data[t-1,:] + Et[t,:]
    vec_Xt = data[lags+1:,:]
    vec_Xt_1 = data[1:data.shape[0]-lags,:]
    vec_Et = Et[lags+1:,:]
    return vec_Xt, vec_Xt_1, vec_Et

def vectorize_df(df):
    columns = []
    for i in range(df.shape[2]):
        columns.append(df[:, :, i]) 
    vec_df = np.hstack(columns)
    return vec_df

def devectorize_df(vec_df, length, amount_row, amount_col):
    df = np.full((length, amount_row, amount_col), np.nan)
    for t in range(length):
        for row in range(amount_row):
            for col in range(amount_col):
                df[t, row, col] = vec_df[t, (col*(amount_col-(amount_col-amount_row))) + row]
    return df

def create_dataframe_sim(data, row, col):
    df = pd.DataFrame(data)
    new_columns = []
    for c in range(col):
        for i in range(row):
            new_columns.append(f'row_{i} col_{c}')
    df.columns = new_columns
    df = df.rename_axis('Time')
    return df

def create_dataframe_emp(data, row, col):
    df = pd.DataFrame(data)
    new_columns = []
    for c in col:
        for i in row:
            new_columns.append(f'{i} {c}')
    df.columns = new_columns
    df = df.rename_axis('Time')
    return df

