import pandas as pd
import numpy as np
import scipy.optimize as opt
from scipy.linalg import cho_solve, cho_factor

def neg_log_likelihood(theta, y, K):
    sigma_g, sigma_e = theta[0], theta[1]
    n = len(y)
    V = sigma_g * K + sigma_e * np.eye(n)  
    L = cho_factor(V) 
    Vy = cho_solve(L, y)
    log_det_V = 2 * np.sum(np.log(np.diag(L[0])))
    nll = 0.5 * (np.dot(y, Vy) + log_det_V + n * np.log(2 * np.pi))
    return nll

fixed_phe = pd.read_csv('adjuested_phe_1496.csv',index_col=0)
fixed_phe = fixed_phe.drop(fixed_phe.columns[5], axis=1)
Genotypes= pd.read_pickle('snp_1496_592929_mode')

for i in range(19):
    phe = pd.DataFrame(fixed_phe.iloc[:,i])
    missing_phe_index = phe[phe.isna().any(axis=1)].index
    y = phe.drop(index=missing_phe_index)
    snp = Genotypes.drop(index=missing_phe_index)
    
    Z = snp.values
    n, m = Z.shape
    p = np.sum(Z, axis=0) / (2 * n)
    M = (Z - 2 * p) / np.sqrt(2 * p * (1 - p))
    G = pd.DataFrame(np.dot(M, M.T) / m)
    G.index = y.index
    theta_0 = [1.0, 1.0]

    y = y.values.ravel()
    result = opt.minimize(neg_log_likelihood, theta_0, args=(y, G), 
                          bounds=[(1e-5, None), (1e-5, None)])
    sigma_g_est, sigma_e_est = result.x
    h2 = sigma_g_est / (sigma_g_est + sigma_e_est)
    trait_name = phe.columns[0]
    print(f"h:{trait_name} : {h2:.3f}")
    
    

