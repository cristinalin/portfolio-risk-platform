import numpy as np
import pandas as pd

def portfolio_returns(dataframe, weights, log_returns = False):
    if type(weights) == pd.DataFrame:
        weights = weights.iloc[1, 1:]
    if type(weights) == pd.Series:
        weights = weights.values
    else:
        weights = np.array(weights)

    if log_returns:
        returns = np.log(dataframe / dataframe.shift(1))
    else:
        returns = dataframe.pct_change()

    portfolio_returns = returns.dot(weights)

    return portfolio_returns.dropna()