import numpy as np
import pandas as pd
from scipy.stats import norm
from risklib.utils.covariance import calculate_cov_matrix
from risklib.utils.portfolio import portfolio_std_dev

'''def calculate_marginal_var(returns, weights, confidence_level):
    cov_matrix = calculate_cov_matrix(returns, weights)
    sigma_w = cov_matrix.values @ weights
    sigma_p = portfolio_std_dev(returns, weights)
    z_abs = abs(norm.ppf(1-confidence_level))
    marginal_var = z_abs * (sigma_w / sigma_p)
    marginal_var = marginal_var.flatten()
    return marginal_var'''

def calculate_marginal_var(returns, weights, confidence_level):
    weights = np.asarray(weights).flatten()
    cov_matrix = returns.cov().values
    
    sigma_w = cov_matrix @ weights
    sigma_p = np.sqrt(weights.T @ cov_matrix @ weights)
    
    z_abs = abs(norm.ppf(1 - confidence_level))
    
    marginal_var = z_abs * (sigma_w / sigma_p)
    return marginal_var