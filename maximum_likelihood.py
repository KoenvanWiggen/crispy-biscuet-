#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:43:42 2024

@author: k.g.vanwiggen
"""

import numpy as np

def create_sigmas(row, col, data):
    T = len(data)
    sig_r_all = np.zeros((row, col * T))
    sig_c_all = np.zeros((col, row * T))
    for t in range(T):
        for i in range(row):
            sig_r_all[i,t*col:(t+1)*col] = data[t,i,:] 
        for c in range(col):
            sig_c_all[c,t*row:(t+1)*row] = data[t,:,c] 
    sig_c = np.cov(sig_c_all, rowvar=True)
    sig_r = np.cov(sig_r_all, rowvar=True)
    return sig_c, sig_r

def update_A_MLE(data, data_lag, A, B, sig_c):
    nominator = np.zeros((len(A),len(A)))
    denominator = np.zeros((len(A),len(A)))
    for t in range(0,len(data)):
       nominator += data[t,:,:] @ np.linalg.inv(sig_c) @ B @ data_lag[t,:,:].T
       denominator += data_lag[t,:,:] @ B.T @ np.linalg.inv(sig_c) @ B @ data_lag[t,:,:].T 
    A_opt = nominator @ np.linalg.inv(denominator)
    return A_opt

def update_B_MLE(data, data_lag, A, B, sig_r):
    nominator = np.zeros((len(B),len(B)))
    denominator = np.zeros((len(B),len(B)))
    for t in range(0,len(data)):
        nominator += data[t,:,:].T @ np.linalg.inv(sig_r) @ A @ data_lag[t,:,:]
        denominator += data_lag[t,:,:].T @ A.T @ np.linalg.inv(sig_r) @ A @ data_lag[t,:,:] 
    B_opt = nominator @ np.linalg.inv(denominator)
    return B_opt

def update_R(data, data_lag, A, B):
    Rt = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    for t in range(len(data)):
        Rt[t,:,:] = data[t,:,:] - A @ data_lag[t,:,:] @ B.T
    return Rt

def update_sig_r(data, Rt, sig_c, sig_r):
    m = len(sig_r)
    n = len(sig_c)
    T = len(data)
    nominator = np.zeros((m,m))
    for t in range(0,T):
        nominator += Rt[t,:,:] @ np.linalg.inv(sig_c) @ Rt[t,:,:].T
    denominator =  n * (T - 1)
    sig_r_opt = nominator / denominator
    return sig_r_opt

def update_sig_c(data, Rt, sig_c, sig_r):
    m = len(sig_r)
    n = len(sig_c)
    T = len(data)
    nominator = np.zeros((n,n))
    for t in range(0,T):
        nominator += Rt[t,:,:].T @ np.linalg.inv(sig_r) @ Rt[t,:,:]
    denominator =  m * (T - 1)
    sig_c_opt = nominator / denominator
    return sig_c_opt

def calculate_maximum_likelihood(iterations, tolerance, initial_A, initial_B, initial_sig_r, initial_sig_c, data, data_lag):
    A = initial_A
    B = initial_B
    sig_r = initial_sig_r
    sig_c = initial_sig_c
    
    for iteration in range(iterations):
        old_A = A.copy()
        old_B = B.copy()
        #old_sig_r = sig_r.copy()
        #old_sig_c = sig_c.copy()


        A = update_A_MLE(data, data_lag, A, B, sig_c)
        A = A/np.linalg.norm(A, ord='fro')
        B = update_B_MLE(data, data_lag, A, B, sig_r)

        Rt = update_R(data, data_lag, A, B)
        sig_r = update_sig_r(data, Rt, sig_c, sig_r)
        sig_r = sig_r/np.linalg.norm(sig_r, ord='fro')
        sig_c = update_sig_c(data, Rt, sig_c, sig_r)
        
        dif_A = np.linalg.norm(A - old_A)
        dif_B = np.linalg.norm(B - old_B)
        #dif_sig_r = np.linalg.norm(sig_r - old_sig_r)
        #dif_sig_c = np.linalg.norm(sig_c - old_sig_c)

        if dif_A < tolerance and dif_B < tolerance:
          print("Converged after", iteration + 1, "iterations.")
          A_MLE, B_MLE, sig_r_MLE, sig_c_MLE = A, B, sig_r, sig_c
          break
    else:
         print("Maximum number of iterations reached without convergence.")
    return A_MLE, B_MLE, sig_r_MLE, sig_c_MLE



