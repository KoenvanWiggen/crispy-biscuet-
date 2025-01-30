#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:27:38 2024

@author: k.g.vanwiggen
"""

import numpy as np
from scipy.stats import wishart, multivariate_normal

def nearest_spd(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    return (B + H) / 2

def create_eps_sim(time, row, col, error, structure):
    if structure == 'random':
        dim2 = row * col
        degrees = dim2 + 1
        scale_matrix = np.eye(dim2)
        covariance = wishart.rvs(df=degrees, scale=scale_matrix)
        vec_Et = multivariate_normal.rvs(mean = np.zeros(dim2), cov = covariance, size=time)
        
    if structure == 'kron':
        degrees_r = row + 1
        scale_matrix_r = np.eye(row)
        covariance_r = wishart.rvs(df=degrees_r, scale=scale_matrix_r)
       
        degrees_c = col + 1
        scale_matrix_c = np.eye(col)
        covariance_c = wishart.rvs(df=degrees_c, scale=scale_matrix_c)
        
        cov_kron = np.kron(covariance_c, covariance_r)
        noise = np.random.normal(0, error, (row * col, row * col))
        cov_structure = cov_kron + noise
        cov_structure = nearest_spd(cov_structure)
        
        vec_Et = multivariate_normal.rvs(mean = np.zeros(row*col), cov = cov_structure, size=time)
    return vec_Et

        