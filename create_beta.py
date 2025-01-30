#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 22:06:10 2024

@author: k.g.vanwiggen
"""

import numpy as np

def create_beta_sim(row, col, error, method): 
    if method == 'MAR':
        A = np.random.uniform(-1, 1, (row, row))
        B = np.random.uniform(-1, 1, (col, col))
        eps_mar = np.random.normal(0, error, (col*row, col*row))
        beta = np.kron(B, A) + eps_mar
        eigenvalues, eigenvector = np.linalg.eig(beta)  
        while np.max(np.abs(eigenvalues)) >= 1 - 0.2:
            eigenvalues = eigenvalues / (np.max(np.abs(eigenvalues)) + 0.2)
            beta = eigenvector @ np.diag(eigenvalues) @ np.linalg.inv(eigenvector)
            beta = beta.real
            eigenvalues, eigenvectors = np.linalg.eig(beta)
        
    elif method == 'alasso':
        beta = np.random.uniform(-1, 1, (col*row, col*row))
        eps_alas = np.random.normal(0, error, (col*row, col*row))
        zero_ratio = 0.5
        zeros = int((row*col) * (row*col) * zero_ratio)
        zero_indices = np.random.choice((row*col) * (row*col), zeros, replace=False) 
        flat_beta = beta.flatten() 
        flat_beta[zero_indices] = 0 
        beta = flat_beta.reshape((row*col),(row*col)) + eps_alas 
        eigenvalues, eigenvector = np.linalg.eig(beta)  
        while np.max(np.abs(eigenvalues)) >= 1 - 0.2:
            eigenvalues = eigenvalues / (np.max(np.abs(eigenvalues)) + 0.2)
            beta = eigenvector @ np.diag(eigenvalues) @ np.linalg.inv(eigenvector)
            beta = beta.real
            eigenvalues, eigenvectors = np.linalg.eig(beta)
            
    else: 
        beta = np.random.uniform(-1, 1, (col*row, col*row))
        eigenvalues, eigenvector = np.linalg.eig(beta)  
        while np.max(np.abs(eigenvalues)) >= 1 - 0.2:
            eigenvalues = eigenvalues / (np.max(np.abs(eigenvalues)) + 0.2)
            beta = eigenvector @ np.diag(eigenvalues) @ np.linalg.inv(eigenvector)
            beta = beta.real
            eigenvalues, eigenvectors = np.linalg.eig(beta)
    
    return beta

def create_beta_sim2(row, col, error, method): 
    if method == 'MAR':
        A = np.random.uniform(-1, 1, (row, row))
        B = np.random.uniform(-1, 1, (col, col))
        eps_mar = np.random.normal(0, error, (col*row, col*row))
        beta = np.kron(B, A) + eps_mar
        
    elif method == 'alasso':
        beta = np.random.uniform(-1, 1, (col*row, col*row))
        eps_alas = np.random.normal(0, error, (col*row, col*row))
        zero_ratio = 0.5
        zeros = int((row*col) * (row*col) * zero_ratio)
        zero_indices = np.random.choice((row*col) * (row*col), zeros, replace=False) 
        flat_beta = beta.flatten() 
        flat_beta[zero_indices] = 0 
        beta = flat_beta.reshape((row*col),(row*col)) + eps_alas 
    
    else:
        beta = np.random.uniform(-1, 1, (col*row, col*row))
    
    eigenvector1, eigenvalues, eigenvector2 = np.linalg.svd(beta)  
    if np.max(np.abs(eigenvalues)) >= 1:
        eigenvalues = eigenvalues / (np.max(np.abs(eigenvalues)) + 0.1)        
        beta = eigenvector1 @ np.diag(eigenvalues) @ eigenvector2
    return beta
