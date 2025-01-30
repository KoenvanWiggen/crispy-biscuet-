#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:53:15 2024

@author: k.g.vanwiggen
"""
import numpy as np
import pandas as pd

################################### Input #####################################
###############################################################################
# m represents the amount of rows
# n the amount of columns
# T the time points
m = 4
n = 3
T = 200
lags = 1

periods_ahead = 11
periods_before = 2
row_shock = 2
column_shock = 2
row_response = 1
column_response = 2
Index = (column_shock*(n-(n - m))) + row_shock
shock = 10

repetitions = 10
MSE = np.full((repetitions, periods_ahead+1, 4), np.nan)
gIRF_estimates = np.full((repetitions, periods_ahead, 5), np.nan)
MSE_exp = pd.DataFrame(np.full((periods_ahead+1, 4), np.nan), columns=['VAR', 'MAR', 'Ridge', 'Alasso'])
MSE_sd = pd.DataFrame(np.full((periods_ahead+1, 4), np.nan), columns=['VAR', 'MAR', 'Ridge', 'Alasso'])



for rep in range(repetitions):
    print(rep)
    ################################ Create data ##################################
    ###############################################################################
    from create_data import devectorize_df, create_dataframe_sim, create_data_sim, create_eps_sim
    from Create_beta import create_beta_sim2

    B_real = create_beta_sim2(m, n, 0, 'random')
    vec_Et = create_eps_sim(T+lags+1, m*n)
    vec_Xt, vec_Xt_1, vec_Et = create_data_sim(T, lags, m, n, B_real, vec_Et)
    Xt, Xt_1 = devectorize_df(vec_Xt, T, m, n), devectorize_df(vec_Xt_1, T, m, n)
    df = create_dataframe_sim(vec_Xt, m, n)
    #shock = np.std(vec_Xt[:, Index])
    
    ##################################### VAR #####################################
    ###############################################################################
    from statsmodels.tsa.vector_ar.var_model import VAR
    Var_model = VAR(vec_Xt).fit(maxlags=1, trend='n')
    B_VAR = Var_model.coefs.reshape(m*n,m*n)
    
    ############################## Projection method ##############################
    ###############################################################################
    Phi_hat, _, _, _ = np.linalg.lstsq(vec_Xt_1, vec_Xt, rcond=None)
    
    from projection_method import transform_kron, calculate_projection_method
    Phi_tilde = transform_kron(Phi_hat, m, n)
    A_PROJ, B_PROJ = calculate_projection_method(Phi_tilde, m, n)
    
    ######################## Maximum Likelihood Estimation ########################
    ###############################################################################
    from maximum_likelihood import create_sigmas, calculate_maximum_likelihood
    #ini_sig_c, ini_sig_r = create_sigmas(m, n, Xt)

    PROJ_Et = vec_Xt.T - np.kron(B_PROJ, A_PROJ) @ vec_Xt_1.T
    PROJ_Et = PROJ_Et.T
    PROJ_e_tilde = transform_kron(np.cov(PROJ_Et, rowvar=False), m, n)
    ini_sig_r, ini_sig_c = calculate_projection_method(PROJ_e_tilde, m, n)
    
        
    A_MLE, B_MLE, sig_r_MLE, sig_c_MLE = calculate_maximum_likelihood(100000, 1e-05, A_PROJ, B_PROJ, ini_sig_r, ini_sig_c, Xt, Xt_1)
    
    ############################### The Penalized VAR #############################
    ###############################################################################
    from penalized_VAR import fit_Lasso, fit_Ridge, fit_Alasso
    #B_Lasso = fit_Lasso(vec_Xt, vec_Xt_1, np.logspace(-4, 2, 10) , 5)
    B_Ridge = fit_Ridge(vec_Xt, vec_Xt_1, np.logspace(-4, 2, 10) , 5)
    B_Alasso = fit_Alasso(vec_Xt, vec_Xt_1, np.logspace(-4, 2, 10) , 5, B_Ridge)
    
    ################################### The gIRF ##################################
    ###############################################################################
    from gIRF import create_gIRF_input, create_coef_matrix, gIRF2d, gIRF3d
    # Create variables
    Var_vec_Et, Var_sig_e, Var_sig_small = create_gIRF_input(vec_Xt, vec_Xt_1, B_VAR, 'VAR', None, None, None, None, None)
    Mar_MLE_vec_Et, Mar_MLE_sig_e, Mar_MLE_sig_small = create_gIRF_input(vec_Xt, vec_Xt_1, None, 'MLE', sig_r_MLE, sig_c_MLE, B_MLE, A_MLE, None)
    Ridge_vec_Et, Ridge_sig_e, Ridge_sig_small = create_gIRF_input(vec_Xt, vec_Xt_1, B_Ridge, 'Ridge', None, None, None, None, None)
    Alas_vec_Et, Alas_sig_e, Alas_sig_small = create_gIRF_input(vec_Xt, vec_Xt_1, B_Alasso, 'Alasso', None, None, None, None, None)
    _, real_sig_e, real_sig_small = create_gIRF_input(None, None, None, 'simulation', None, None, None, None, vec_Et)
    
    # Create the alphas
    Var_AgIRF = create_coef_matrix(Xt, B_VAR)
    Mar_MLE_AgIRF = create_coef_matrix(Xt, np.kron(B_MLE, A_MLE))
    Ridge_AgIRF = create_coef_matrix(Xt, B_Ridge)
    Alas_AgIRF = create_coef_matrix(Xt, B_Alasso)
    real_AgIRF = create_coef_matrix(Xt, B_real)
    
    # Create the gIRF data
    Var_gIRF2d = gIRF2d(periods_ahead, shock, row_shock, column_shock, Var_sig_small, Var_sig_e, Var_AgIRF, m, n, df)
    Mar_MLE_gIRF2d = gIRF2d(periods_ahead, shock, row_shock, column_shock, Mar_MLE_sig_small, Mar_MLE_sig_e, Mar_MLE_AgIRF, m, n, df)
    Ridge_gIRF2d = gIRF2d(periods_ahead, shock, row_shock, column_shock, Ridge_sig_small, Ridge_sig_e, Ridge_AgIRF, m, n, df)
    Alas_gIRF2d = gIRF2d(periods_ahead, shock, row_shock, column_shock, Alas_sig_small, Alas_sig_e, Alas_AgIRF, m, n, df)
    real_gIRF2d = gIRF2d(periods_ahead, shock, row_shock, column_shock, real_sig_small, real_sig_e, real_AgIRF, m, n, df)
    
    Var_gIRF3d = gIRF3d(periods_ahead, Var_gIRF2d, m, n)
    Mar_MLE_gIRF3d = gIRF3d(periods_ahead, Mar_MLE_gIRF2d, m, n)
    Ridge_gIRF3d = gIRF3d(periods_ahead, Ridge_gIRF2d, m, n)
    Alas_gIRF3d = gIRF3d(periods_ahead, Alas_gIRF2d, m, n)
    real_gIRF3d = gIRF3d(periods_ahead, real_gIRF2d, m, n)
    
    from Plot_gIRF import plot_gIRF, plot_gIRF2, plot_gIRF3

    # Making a graph of the gIRFs separately
    #plot_gIRF(periods_before, row_response, column_response, Var_gIRF3d, m, n, df, 'VAR')
    #plot_gIRF(periods_before, row_response, column_response, Mar_MLE_gIRF3d, m, n, df, 'MLE') 
    #plot_gIRF(periods_before, row_response, column_response, Ridge_gIRF3d, m, n, df, 'Ridge')
    #plot_gIRF(periods_before, row_response, column_response, Alas_gIRF3d, m, n, df, 'Alas')
    #plot_gIRF(periods_before, row_response, column_response, real_gIRF3d, m, n, df, 'Real')
    
    # Plotting all graphs in one plot
    plot_gIRF2(periods_before, row_response, column_response, [Var_gIRF3d, Mar_MLE_gIRF3d, Ridge_gIRF3d, Alas_gIRF3d, real_gIRF3d], m, n, df, ['VAR', 'MLE', 'Ridge', 'Alasso', 'Real'])
    
    ######################### calculating the differences #########################
    ###############################################################################
    from difference import calculate_difference, difference_table, IRF_values
    Var_MSE = calculate_difference(periods_ahead, row_response, column_response, m, n, Var_gIRF2d, real_gIRF2d)
    Mar_MLE_MSE = calculate_difference(periods_ahead, row_response, column_response, m, n, Mar_MLE_gIRF2d, real_gIRF2d)
    Ridge_MSE = calculate_difference(periods_ahead, row_response, column_response, m, n, Ridge_gIRF2d, real_gIRF2d)
    Alas_MSE = calculate_difference(periods_ahead, row_response, column_response, m, n, Alas_gIRF2d, real_gIRF2d)

    ############################# Saving the results ##############################
    ###############################################################################
    MSE[rep, :, :] = np.hstack((np.round(Var_MSE.reshape(-1,1), 3), np.round(Mar_MLE_MSE.reshape(-1,1), 3), np.round(Ridge_MSE.reshape(-1,1), 3), np.round(Alas_MSE.reshape(-1,1), 3)))
    gIRF_estimates[rep, :, :] = np.hstack((IRF_values(row_response, column_response, m, n, Var_gIRF2d), IRF_values(row_response, column_response, m, n, Mar_MLE_gIRF2d), IRF_values(row_response, column_response, m, n, Ridge_gIRF2d), IRF_values(row_response, column_response, m, n, Alas_gIRF2d), IRF_values(row_response, column_response, m, n, real_gIRF2d)))

for method in range(MSE.shape[2]):
    for t in range(MSE.shape[1]):
        MSE_exp.iloc[t, method] = np.round(np.mean(MSE[:, t, method]), 3)
        MSE_sd.iloc[t, method] = np.round(np.std(MSE[:, t, method]), 3)   

Table = pd.DataFrame(np.vstack((MSE_exp.iloc[periods_ahead,:], MSE_sd.iloc[periods_ahead,:])), columns=['VAR', 'MAR', 'Ridge', 'Alasso'], index = ['Exp', 'sd'])
difference_table(Table)

from box_plot import create_box_plot
create_box_plot(gIRF_estimates, 5)
