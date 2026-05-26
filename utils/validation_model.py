import numpy as np
import scipy.linalg

from scot.datatools import cat_trials

def biggest_eigenvector_mvar(var_source):
    m, mp = var_source.coef.shape
    p_lags = mp // m

    #companion matrix (or transition matrix)
    top_block = np.hstack([var_source.coef[:, i::p_lags] for i in range(p_lags)])
    if p_lags > 1:
        im = np.eye(m)
        eye_block = im
        for i in range(p_lags - 2):
            eye_block = scipy.linalg.block_diag(im, eye_block)
        eye_block = np.hstack([eye_block, np.zeros((m * (p_lags - 1), m))])
        companion_matrix = np.vstack([top_block, eye_block])
    else:
        companion_matrix = top_block

    biggest_eigenvector = np.max(np.abs(np.linalg.eigvals(companion_matrix)))

    return biggest_eigenvector

def calculate_bic(var_model, mvar_result, n_samples, n_trials):
    p = var_model.p
    K = var_model.coef.shape[0]
    T = (n_samples - p) * n_trials

    Sigma_u_mle = mvar_result.c * ((T - 1) / T) 
    
    sign, log_det_Sigma = np.linalg.slogdet(Sigma_u_mle)
    if sign <= 0:
        return np.inf  # Penalidade máxima se a matriz de covariância for singular
        
    k_total = (p * K) * K
    return log_det_Sigma + (np.log(T) / T) * k_total