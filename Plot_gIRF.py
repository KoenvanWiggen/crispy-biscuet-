#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 19:46:04 2024

@author: k.g.vanwiggen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_gIRF(prior, row, col, gIRF3d, amount_row, amount_col, dataframe, method):
    plt.plot(np.arange(-prior, gIRF3d.shape[0], 1), np.concatenate((np.zeros(prior), gIRF3d[:, row, col])), color = 'black') 
    plt.xticks(np.arange(-prior, gIRF3d.shape[0], 1)) 
    plt.ylabel(dataframe.columns[(col*(amount_col-(amount_col - amount_row))) + row])
    plt.xlabel('quarters in the future')
    plt.title(method)
    #plt.savefig(dataframe.columns[(col*(amount_col-(amount_col - amount_row))) + row])
    plt.show()

def plot_gIRF2(prior, row, col, gIRF3d, amount_row, amount_col, dataframe, titles):
    plt.plot(np.arange(-prior, gIRF3d[0].shape[0], 1), np.concatenate((np.zeros(prior), gIRF3d[0][:, row, col])), color = 'black', linestyle = 'dotted', label = titles[0])
    plt.plot(np.arange(-prior, gIRF3d[1].shape[0], 1), np.concatenate((np.zeros(prior), gIRF3d[1][:, row, col])), color = 'grey', linestyle = 'dashed', label = titles[1]) 
    plt.plot(np.arange(-prior, gIRF3d[2].shape[0], 1), np.concatenate((np.zeros(prior), gIRF3d[2][:, row, col])), color = 'grey', linestyle = 'dashdot', label = titles[2]) 
    plt.plot(np.arange(-prior, gIRF3d[3].shape[0], 1), np.concatenate((np.zeros(prior), gIRF3d[3][:, row, col])), color = 'black', linestyle = 'dashed', label = titles[3])
    plt.plot(np.arange(-prior, gIRF3d[4].shape[0], 1), np.concatenate((np.zeros(prior), gIRF3d[4][:, row, col])), color = 'black', linestyle = 'dashdot', label = titles[4])
    plt.plot(np.arange(-prior, gIRF3d[5].shape[0], 1), np.concatenate((np.zeros(prior), gIRF3d[5][:, row, col])), color = 'black', linestyle = 'solid', label = titles[5]) 
    plt.xticks(np.arange(-prior, gIRF3d[0].shape[0], 1)) 
    plt.ylabel(dataframe.columns[(col*(amount_col-(amount_col - amount_row))) + row])
    #plt.ylabel('gIRFs by the methods')
    plt.xlabel('quarters in the future')
    plt.title('gIRFs of the different methods')
    plt.legend()
    plt.show()

def plot_gIRF3(prior, row, col, gIRF3d, amount_row, amount_col, dataframe, titles, emp, col_emp):
    fig, ax1 = plt.subplots()
    line1, = ax1.plot(np.arange(-prior, gIRF3d[0].shape[0], 1), np.concatenate((np.zeros(prior), gIRF3d[0][:, row, col])), color = 'black', linestyle = 'dotted', label = titles[0])
    line2, = ax1.plot(np.arange(-prior, gIRF3d[1].shape[0], 1), np.concatenate((np.zeros(prior), gIRF3d[1][:, row, col])), color = 'grey', linestyle = 'dashed', label = titles[1]) 
    line3, = ax1.plot(np.arange(-prior, gIRF3d[2].shape[0], 1), np.concatenate((np.zeros(prior), gIRF3d[2][:, row, col])), color = 'black', linestyle = 'dashdot', label = titles[2]) 
    line4, = ax1.plot(np.arange(-prior, gIRF3d[3].shape[0], 1), np.concatenate((np.zeros(prior), gIRF3d[3][:, row, col])), color = 'black', linestyle = 'dashed', label = titles[3]) 
    #line4, = ax1.plot(np.arange(-prior, gIRF3d[3].shape[0], 1), np.concatenate((np.zeros(prior), gIRF3d[3][:, row, col])), color = 'black', linestyle = 'dashdot', label = titles[3]) 

    ax1.set_xticks(np.arange(-prior, gIRF3d[0].shape[0], 1)) 
    ax1.set_ylabel('gIRFs by the methods')
    ax1.set_xlabel('quarters in the future')
    
    ax2 = ax1.twinx()
    line5, = ax2.plot(np.arange(-prior, gIRF3d[0].shape[0], 1), np.concatenate((np.zeros(prior), emp.iloc[1:12, 4])), color = 'black', linestyle = 'solid', label = 'Exports')
    ax2.set_ylabel('Exports in NL')
    
    lines = [line1, line2, line3, line4, line5]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels)
    plt.show()
