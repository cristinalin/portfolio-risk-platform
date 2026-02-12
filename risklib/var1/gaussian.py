import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_gaussian_var(returns, confidence_level):
    alpha = 1 - confidence_level
    z_alpha = norm.ppf(alpha)
    z_abs = abs(z_alpha)
    sigma = np.sqrt(returns.var())
    mu = returns.mean()
    
    gaussian_var = z_abs * sigma
    return gaussian_var
