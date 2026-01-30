import numpy as np

def variance_concentration(explained_variance_ratio, k=1):
    evr = np.asarray(explained_variance_ratio)
    return evr[:k].sum()

def diversification_ratio(returns, weights):
    cov = np.cov(returns, rowvar=False)
    vols = np.sqrt(np.diag(cov))

    numerator = weights @ vols
    denominator = np.sqrt(weights @ cov @ weights)

    return numerator / denominator