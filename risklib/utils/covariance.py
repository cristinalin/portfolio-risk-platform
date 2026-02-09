import numpy as np

def calculate_cov_matrix(returns, tickers):
    cov_matrix = returns.cov()
    #cov_matrix = cov_matrix.loc[tickers, tickers]
    return cov_matrix