import numpy as np
import pandas as pd
from risklib.var1.historical import calculate_historical_var

def calculate_ewma_var(returns, lambda_param, confidence_level):
    returns = pd.Series(returns).dropna()
    returns_scaled = pd.Series()
    ewma_vol = []
    if len(returns) == 0:
        return np.nan
    sigma2_0 = returns.var()
    ewma_sigma2 = sigma2_0
    for r in returns:
        ewma_sigma2 = lambda_param * ewma_sigma2 + (1 - lambda_param) * (r ** 2)
        ewma_vol.append(ewma_sigma2)
    scaling_factor = np.sqrt(ewma_sigma2)
    for i in range(0,len(returns)):
        returns_scaled[i] = (returns[i] / np.sqrt(ewma_vol[i])) * scaling_factor

    ewma_var = calculate_historical_var(returns_scaled, confidence_level)
    return ewma_var