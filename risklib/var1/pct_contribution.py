from risklib.var1.component import calculate_component_var

def pct_contribution_var(returns, weights, confidence_level, var_portfolio):
    pct_contrib_var = calculate_component_var(returns, weights, confidence_level) / var_portfolio
    return pct_contrib_var