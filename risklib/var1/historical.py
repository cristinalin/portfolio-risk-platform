import numpy as np
import pandas as pd

def calculate_historical_var(returns, confidence_level):
    returns = pd.Series(returns).dropna()
    sorted_returns = np.sort(returns)
    index = int(len(sorted_returns) * (1 - confidence_level))
    hist_var = sorted_returns[index]
    return hist_var