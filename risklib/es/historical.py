import numpy as np
import pandas as pd
from risklib.var1.historical import calculate_historical_var

def calculate_historical_es(returns, confidence_level):
    returns = pd.Series(returns).dropna()
    var = calculate_historical_var(returns, confidence_level)
    es = returns[returns <= var]
    if len(es) == 0:
        return np.nan
    else:
        return es.mean()