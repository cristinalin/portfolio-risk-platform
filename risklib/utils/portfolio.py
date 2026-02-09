import numpy as np
import pandas as pd
from risklib.utils.covariance import calculate_cov_matrix

def portfolio_std_dev(returns, weights):
    cov_matrix = calculate_cov_matrix(returns, weights)
    w = weights.values.reshape(-1, 1)
    sigma2_p = (w.T @ cov_matrix.values @ w).item()
    
    return np.sqrt(sigma2_p)

def portfolio_returns(dataframe, weights, log_returns = False):
    if type(weights) == pd.DataFrame:
        weights = weights.iloc[1, 1:]
    if type(weights) == pd.Series:
        weights = weights.values
    else:
        weights = np.array(weights)
    
    print(weights)

    if log_returns:
        returns = np.log(dataframe / dataframe.shift(1))
    else:
        returns = dataframe.pct_change()

    portfolio_returns = returns.dot(weights)

    return portfolio_returns.dropna()