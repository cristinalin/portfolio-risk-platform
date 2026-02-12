import numpy as np

def calculate_cov_matrix(returns, tickers):
    cov_matrix = returns.cov()
    return cov_matrix