#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:09:59 2024

@author: k.g.vanwiggen
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gIRF2d(h, delta, row_shock, col_shock, sig_e, coef, m, n, dataframe):
    shock_index = (col_shock*(n-(n - m))) + row_shock
    gIRF = np.zeros((h, m * n))
    e = np.zeros(m * n)
    e[shock_index] = 1
    for t in range(h):
        A_h = np.linalg.matrix_power(coef, t)
        numerator = delta * (A_h @ sig_e @ e)
        denominator = e.T @ sig_e @ e
        gIRF[t, :] = numerator / denominator
    
    gIRF2d = pd.DataFrame(gIRF)
    gIRF2d.columns = dataframe.columns
    return gIRF2d

def gIRF3d(h, gIRF2d, amount_row, amount_col):
    gIRF3d = np.zeros((h, amount_row, amount_col))
    for t in range(h):
        for col in range(amount_col):
            for row in range(amount_row):
                gIRF3d[t, row, col] = gIRF2d.iloc[t, (col*(amount_col-(amount_col - amount_row))) + row]
    return gIRF3d