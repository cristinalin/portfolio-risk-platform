import numpy as np
import pandas as pd
import risklib as rm
# call the functions using rm.(function name here)()

# import data
prices = pd.read_excel(io = '/Users/cristina_yj/Desktop/LIS/Risk/projects/all/Data_asset daily prices (2).xlsx', index_col=0)

# --- Liquidity data (e.g. bid-ask, volumes, etc.)
liquidity = pd.read_excel(io = '/Users/cristina_yj/Desktop/LIS/Risk/projects/all/Data_liquidity.xlsx', index_col=0)

# --- Returns
returns = prices.pct_change().dropna()

# --- Portfolio weights (example: equal-weighted)
n_assets = returns.shape[1]
weights = np.ones(n_assets) / n_assets


# =========================
# 2. HISTORICAL / PARAMETRIC VAR
# =========================

alpha = 0.99

hist_var = calculate_historical_var(returns.iloc[:, 0], alpha=alpha)
ewma_var_ = calculate_ewma_var(returns.iloc[:, 0], alpha=alpha)
port_var = calculate_historical_var(returns.values, weights, alpha=alpha)

print("=== VaR ===")
print(f"Historical VaR (asset 1): {hist_var:.4f}")
print(f"EWMA VaR (asset 1):       {ewma_var_:.4f}")
print(f"Portfolio VaR:            {port_var:.4f}\n")


# =========================
# 3. COMPONENT & MARGINAL VAR
# =========================

from risklib.var.marginal import marginal_var, component_var

mvar = marginal_var(returns.values, weights, alpha=alpha)
cvar = component_var(returns.values, weights, alpha=alpha)

print("=== Marginal / Component VaR ===")
print("Marginal VaR:", mvar)
print("Component VaR:", cvar)
print("Sum Component VaR:", cvar.sum(), "\n")


# =========================
# 4. MONTE CARLO VAR
# =========================

from risklib.monte_carlo import monte_carlo_var

mc_var = monte_carlo_var(
    returns.values,
    weights,
    alpha=alpha,
    n_sim=50_000
)

print("=== Monte Carlo ===")
print(f"Monte Carlo VaR: {mc_var:.4f}\n")


# =========================
# 5. BACKTESTING
# =========================

from risklib.backtesting import kupiec_test, christoffersen_test

# Rolling VaR (example: historical, 250-day window)
rolling_var = returns.rolling(250).apply(
    lambda x: historical_var(x, alpha),
    raw=False
).dropna()

test_returns = returns.iloc[-len(rolling_var):, 0].values
test_var = rolling_var.iloc[:, 0].values

kupiec = kupiec_test(test_returns, test_var, alpha=alpha)
christoffersen = christoffersen_test(test_returns, test_var, alpha=alpha)

print("=== Backtesting ===")
print("Kupiec test:", kupiec)
print("Christoffersen test:", christoffersen, "\n")


# =========================
# 6. PCA & DIVERSIFICATION
# =========================

from risklib.correlation import (
    pca,
    variance_concentration,
    effective_number_of_bets,
    diversification_ratio
)

eigvals, eigvecs, evr, scores, loadings = pca(returns.values)

enb = effective_number_of_bets(evr)
c1 = variance_concentration(evr, k=1)
dr = diversification_ratio(returns.values, weights)

print("=== PCA & Diversification ===")
print(f"Variance explained by PC1: {c1:.2%}")
print(f"Effective number of bets:  {enb:.2f}")
print(f"Diversification ratio:     {dr:.2f}\n")


# =========================
# 7. LIQUIDITY-ADJUSTED VAR
# =========================

from risklib.liquidity import liquidity_adjusted_var

# Example liquidity input:
# assume liquidity dataframe aligned with returns columns
liq_var = liquidity_adjusted_var(
    returns.values,
    weights,
    liquidity.values,
    alpha=alpha
)

print("=== Liquidity Adjustment ===")
print(f"Liquidity-adjusted VaR: {liq_var:.4f}\n")


# =========================
# 8. END
# =========================

print("Risk pipeline executed successfully.")
