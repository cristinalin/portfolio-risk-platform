from risklib.es.historical import calculate_historical_es
from risklib.utils.portfolio import portfolio_std_dev
from risklib.utils.covariance import calculate_cov_matrix

def calculate_marginal_es(returns, weights, confidence_level):
    es_port = calculate_historical_es(returns, confidence_level)
    sigma_p = portfolio_std_dev(returns, weights)
    es_mult = es_port / sigma_p
    w = weights.values.reshape(-1, 1)
    sigma_w = calculate_cov_matrix(returns, weights) @ w

    marginal_es = es_mult * (sigma_w/sigma_p)
    marginal_es = marginal_es.flatten()

    return marginal_es