#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:08:17 2024

@author: k.g.vanwiggen
"""
import numpy as np 
import matplotlib.pyplot as plt


def calculate_difference(h, row, col, amount_row, amount_col, df_method, df_real):
    MSE = np.zeros((h+1))
    comb = (col * (amount_col - (amount_col - amount_row))) + row
    for t in range(h):
        MSE[t] = np.square(df_method.iloc[t,comb] - df_real.iloc[t,comb])
    MSE = MSE/h
    MSE[h] = np.sum(MSE[0:h-1])
    return MSE

def difference_table(dataframe):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    tbl = ax.table(cellText=dataframe.values, colLabels=dataframe.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.2)
    plt.tight_layout()
    plt.show()
    
def IRF_values(row, col, amount_row, amount_col, df_method):
    comb = (col * (amount_col - (amount_col - amount_row))) + row
    estimated_values = np.array(df_method.iloc[:, comb]).reshape(-1, 1)
    return estimated_values