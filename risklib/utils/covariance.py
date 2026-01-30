import numpy as np

def calculate_cov_matrix(returns, weights):
    cov_matrix = returns.cov()
    cov_matrix = cov_matrix.loc[weights.index, weights.index]
    return cov_matrix