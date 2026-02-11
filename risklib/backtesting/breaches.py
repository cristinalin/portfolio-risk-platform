import numpy as np
from risklib.var1.historical import calculate_historical_var

def calculate_breaches(returns, window, confidence_level):
    historical_vars = []
        
    for j in range(len(returns)):
        if j + window < len(returns):
            window_returns = returns[j+1 : j+1+window]
            var_value = calculate_historical_var(window_returns, confidence_level)
            historical_vars.append(var_value)
        else:
            historical_vars.append(np.nan)
    breaches = 0
    breach_sequence = []
    days = 0
    
    for k in range(len(returns)):
        if np.isnan(historical_vars[k]):
            continue
        elif returns.iloc[k] < historical_vars[k]:
            breaches += 1
            breach_sequence.append(1)
        else:
            breach_sequence.append(0)
    
    return breach_sequence, breaches