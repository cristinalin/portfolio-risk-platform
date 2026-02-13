import numpy as np
from risklib.var1.marginal import calculate_marginal_var

def calculate_component_var(returns, weights, confidence_level):
    weights = np.asarray(weights).flatten()
    marginal_var = calculate_marginal_var(returns, weights, confidence_level)
    component_var = weights * marginal_var
    return component_var