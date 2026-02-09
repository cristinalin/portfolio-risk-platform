import numpy as np
import pandas as pd
import risklib as rm
# call the functions using rm.(function name here)()

# import data
prices = pd.read_excel(io = '/Users/cristina_yj/Desktop/LIS/Risk/projects/all/Data_asset daily prices.xlsx', sheet_name='prices')
weights = pd.read_excel(io = '/Users/cristina_yj/Desktop/LIS/Risk/projects/all/Data_asset daily prices.xlsx', sheet_name='weights')
liquidity = pd.read_excel(io = '/Users/cristina_yj/Desktop/LIS/Risk/projects/all/Data_liquidity.xlsx', sheet_name = 'Liquidity')
print(prices, weights, liquidity)

#returns = rm.portfolio_returns(prices, weights)
log_returns = False

if isinstance(weights, pd.DataFrame):
    weights = weights.iloc[0]
if isinstance(weights, pd.Series):
    weights = weights.values
else:
    weights = np.array(weights)

weights = weights / np.sum(weights)

if log_returns:
    returns = np.log(prices / prices.shift(1))
else:
    returns = prices.pct_change()

portfolio_returns = returns.dot(weights)
print(portfolio_returns)

n_assets = returns.shape[1]
weights = np.ones(n_assets) / n_assets

alpha = 0.99
window = 1000
hist_var = rm.calculate_historical_var(returns.iloc[:, 0], alpha=alpha)
ewma_var_ = rm.calculate_ewma_var(returns.iloc[:, 0], alpha=alpha)
port_var = rm.calculate_historical_var(returns.values, weights, alpha=alpha)

print("=== VaR ===")
print(f"Historical VaR (asset 1): {hist_var:.4f}")
print(f"EWMA VaR (asset 1):       {ewma_var_:.4f}")
print(f"Portfolio VaR:            {port_var:.4f}\n")

mvar = rm.calculate_marginal_var(returns.values, weights, alpha=alpha)
cvar = rm.calculate_component_var(returns.values, weights, alpha=alpha)

print("=== Marginal / Component VaR ===")
print("Marginal VaR:", mvar)
print("Component VaR:", cvar)
print("Sum Component VaR:", cvar.sum(), "\n")

mc_var = rm.monte_carlo(
    returns.values,
    weights,
    alpha=alpha,
    n_sim=50_000
)

print("=== Monte Carlo ===")
print(f"Monte Carlo VaR: {mc_var:.4f}\n")


# Rolling VaR (example: historical, 250-day window)
rolling_var = returns.rolling(250).apply(
    lambda x: rm.calculate_historical_var(x, alpha),
    raw=False
).dropna()

test_returns = returns.iloc[-len(rolling_var):, 0].values
test_var = rolling_var.iloc[:, 0].values

kupiec = rm.kupiec_test(test_returns, test_var, alpha=alpha)
breaches = rm.calculate_breaches(test_returns, window, 1-alpha)
christoffersen = rm.christoffersen_test(breaches)

print("=== Backtesting ===")
print("Kupiec test:", kupiec)
print("Christoffersen test:", christoffersen, "\n")

eigvals, eigvecs, evr, scores, loadings = rm.pca(returns.values)

enb = rm.calculate_breaches(evr)
c1 = rm.variance_concentration(evr, k=1)
dr = rm.diversification_ratio(returns.values, weights)

print("=== PCA & Diversification ===")
print(f"Variance explained by PC1: {c1:.2%}")
print(f"Effective number of bets:  {enb:.2f}")
print(f"Diversification ratio:     {dr:.2f}\n")

# Example liquidity input:
# assume liquidity dataframe aligned with returns columns
liq_var = rm.liquidity_adjusted(
    returns.values,
    weights,
    liquidity.values,
    alpha=alpha
)

print("=== Liquidity Adjustment ===")
print(f"Liquidity-adjusted VaR: {liq_var:.4f}\n")

print("Risk pipeline executed successfully.")
