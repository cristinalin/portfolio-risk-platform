from risklib.var1.marginal import calculate_marginal_var

def calculate_component_var(returns, weights, confidence_level):
    component_var = weights * calculate_marginal_var(returns, weights, confidence_level)
    return component_var