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

    # --- NEW: Identify top 3 assets contributing most to PC1
    abs_loadings_pc1 = np.abs(loadings[:, 0])
    try:
        # For DataFrame: get column names
        sorted_idx = np.argsort(abs_loadings_pc1)[::-1]  # descending order
        top_3_assets = returns.columns[sorted_idx[:3]].tolist()
    except AttributeError:
        # fallback if returns is a NumPy array
        sorted_idx = np.argsort(abs_loadings_pc1)[::-1]
        top_3_assets = sorted_idx[:3].tolist()

    return eigenvalues, eigenvectors, explained_variance_ratio, scores, loadings, top_3_assets