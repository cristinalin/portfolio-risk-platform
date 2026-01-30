from risklib.es.historical import calculate_historical_es
from risklib.es.marginal import calculate_marginal_es

def calculate_component_es(returns, weights, confidence_level):
    marginal_es = calculate_marginal_es(returns, weights, confidence_level)
    w_vec = weights.values

    component_es = w_vec * marginal_es
    return component_es

def pct_contribution_es(returns, weights, confidence_level):
    component_es = calculate_component_es(returns, weights, confidence_level)
    es_port_from_components = component_es.sum()
    pct_contrib_es = component_es / es_port_from_components
    return pct_contrib_es