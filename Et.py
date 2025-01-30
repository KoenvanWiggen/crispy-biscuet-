#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:50:56 2024

@author: k.g.vanwiggen
"""
import numpy as np

def create_Et(vec_df, vec_df_lag, coef):
    vec_Et = np.full((vec_df.shape[0], vec_df.shape[1]), np.nan)
    for t in range(vec_df.shape[0]):
        vec_Et[t,:] = vec_df[t,:] - coef @ vec_df_lag[t,:]
    return vec_Et