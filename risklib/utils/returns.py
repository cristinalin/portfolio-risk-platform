import numpy as np
import pandas as pd

def portfolio_returns(dataframe, weights, log_returns = False):
    if isinstance(weights, pd.DataFrame):
        weights = weights.iloc[0]
    if isinstance(weights, pd.Series):
        weights = weights.values
    else:
        weights = np.array(weights)
    
    weights = weights / weights.sum()

    if log_returns:
        returns = np.log(dataframe / dataframe.shift(1))
    else:
        returns = dataframe.pct_change()

    portfolio_returns = returns.dot(weights)

    return portfolio_returns.dropna()
