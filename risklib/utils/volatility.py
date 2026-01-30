import numpy as np
from risklib.utils.covariance import calculate_cov_matrix

def portfolio_std_dev(returns, weights):
    cov_matrix = calculate_cov_matrix(returns, weights)
    w = weights.values.reshape(-1, 1)
    sigma2_p = (w.T @ cov_matrix.values @ w).item()
    
    return np.sqrt(sigma2_p)