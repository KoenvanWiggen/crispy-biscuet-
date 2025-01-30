#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:05:32 2024

@author: k.g.vanwiggen
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV

def fit_Lasso(y, X, alphas, cv):
    X = StandardScaler().fit(X).transform(X)
    y = StandardScaler().fit(y).transform(y)
    n = y.shape[1]
    B_Lasso = np.full((n, n), np.nan)
    for j in range(n):
        Lasso = LassoCV(alphas=alphas, cv=cv, max_iter=10000000).fit(X, y[:, j])
        B_Lasso[:,j] = Lasso.coef_      
    return B_Lasso

def fit_Ridge(y, X, alphas, cv):
    X = StandardScaler().fit(X).transform(X)
    y = StandardScaler().fit(y).transform(y)
    n = y.shape[1]
    B_Ridge = np.full((n, n), np.nan)
    for j in range(n):
        Ridge = RidgeCV(alphas=alphas, cv = cv).fit(X, y[:, j])
        B_Ridge[:,j] = Ridge.coef_
    return B_Ridge

def fit_Alasso(y, X, alphas, cv, initial_beta):
    X = StandardScaler().fit(X).transform(X)
    y = StandardScaler().fit(y).transform(y)
    B_Alasso = np.full((y.shape[1], y.shape[1]), np.nan)
    n = y.shape[1]
    for j in range(n):
        weights = 1 / np.abs(initial_beta[:,j])
        weighted_X = X / weights
        alasso = LassoCV(alphas=alphas, cv=cv, max_iter=10000000).fit(weighted_X, y[:,j])
        B_Alasso[:,j] = alasso.coef_ / weights
    return B_Alasso