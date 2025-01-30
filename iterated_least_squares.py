#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:01:59 2024

@author: k.g.vanwiggen
"""
import numpy as np

def update_A_LSE(data, data_lag, A, B):
    nominator = np.zeros((len(A),len(A)))
    denominator = np.zeros((len(A),len(A)))
    for t in range(0,len(data)):
       nominator += data[t,:,:] @ B @ data_lag[t,:,:].T
       denominator += data_lag[t,:,:] @ B.T @ B @ data_lag[t,:,:].T 
    A_opt = nominator @ np.linalg.inv(denominator)
    return A_opt

def update_B_LSE(data, data_lag, A, B):
    nominator = np.zeros((len(B),len(B)))
    denominator = np.zeros((len(B),len(B)))
    for t in range(0,len(data)):
        nominator += data[t,:,:].T @ A @ data_lag[t,:,:]
        denominator += data_lag[t,:,:].T @ A.T @ A @ data_lag[t,:,:]
    B_opt = nominator @ np.linalg.inv(denominator)
    return B_opt

def calculate_iterated_least_squares(iterations, tolerance, initial_A, initial_B, data, data_lag):
    A = initial_A
    B = initial_B
    for iteration in range(iterations):
        old_A = A.copy()
        old_B = B.copy()

        A = update_A_LSE(data, data_lag, A, B)
        A = A/np.linalg.norm(A, ord='fro')
        B = update_B_LSE(data, data_lag, A, B)

        dif_A = np.linalg.norm(A - old_A)
        dif_B = np.linalg.norm(B - old_B)

        if dif_B < tolerance and dif_A < tolerance:
            print("Converged after", iteration + 1, "iterations.")
            A_LSE, B_LSE = A, B
            break
    else:
        print("Maximum number of iterations reached without convergence.")
        A_LSE, B_LSE = A, B
    return A_LSE, B_LSE





