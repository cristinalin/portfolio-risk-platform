import numpy as np

def pca(returns, n_components=None, standardize=True, use_correlation=True):

    # --- Centering
    Xc = returns - returns.mean(axis=0)

    # --- Scaling (equivalent to scale() in R)
    if standardize:
        std = Xc.std(axis=0, ddof=1)
        Xc = Xc / std

    # --- Covariance / Correlation matrix
    if use_correlation:
        Sigma = np.corrcoef(Xc, rowvar=False)
    else:
        Sigma = np.cov(Xc, rowvar=False)

    # --- Eigen decomposition (symmetric matrix → eigh)
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

    # --- Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # --- Components selection
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]
        eigenvalues = eigenvalues[:n_components]

    # --- Explained variance
    total_variance = eigenvalues.sum()
    explained_variance_ratio = eigenvalues / total_variance

    # --- Scores
    scores = Xc @ eigenvectors

    # --- Loadings
    loadings = eigenvectors

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained_variance_ratio": explained_variance_ratio,
        "scores": scores,
        "loadings": loadings,
    }