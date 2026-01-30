import numpy as np

def variance_concentration(explained_variance_ratio, k=1):
    evr = np.asarray(explained_variance_ratio)
    return evr[:k].sum()