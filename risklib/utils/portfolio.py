import numpy as np
import pandas as pd
from risklib.utils.covariance import calculate_cov_matrix

def portfolio_std_dev(returns, weights):
    cov_matrix = calculate_cov_matrix(returns, weights)
    w = np.asarray(weights).reshape(-1,1)
    sigma2_p = (w.T @ cov_matrix.values @ weights).item()
    
    return np.sqrt(sigma2_p)

def portfolio_returns(data, weights, log_returns = False):
    if log_returns:
        returns = np.log(data).diff()
    else:
        returns = data.pct_change(periods = -1)

    portfolio_returns = returns @ weights

    return portfolio_returns.dropna()