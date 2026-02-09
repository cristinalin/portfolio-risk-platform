import numpy as np
import pandas as pd
import risklib as rm
# call the functions using rm.(function name here)()

# import data
prices = pd.read_excel(io = '/Users/cristina_yj/Desktop/LIS/Risk/projects/all/Data_asset daily prices.xlsx', sheet_name='prices')
weights = pd.read_excel(io = '/Users/cristina_yj/Desktop/LIS/Risk/projects/all/Data_asset daily prices.xlsx', sheet_name='weights', header = None)
tickers = weights.iloc[0, 1:].tolist()
w = weights.iloc[1, 1:].to_list()
liquidity = pd.read_excel(io = '/Users/cristina_yj/Desktop/LIS/Risk/projects/all/Data_liquidity.xlsx', sheet_name = 'Liquidity')

returns = rm.portfolio_returns(prices.iloc[:,1:], w, log_returns = False)
n_assets = len(w)

confidence_level = 0.99
# alpha = 0.01 # significance level (confidence_level = 1 - alpha)
lambda_param = 0.97
window = 1000
hist_var_company_level = rm.calculate_historical_var(prices.iloc[:, 1].pct_change(periods=-1), confidence_level)
ewma_var_ = rm.calculate_ewma_var(returns, lambda_param, confidence_level)
port_var = rm.calculate_historical_var(returns, confidence_level)

print("=== VaR ===")
print(f"Historical VaR (asset 1): {hist_var_company_level:.4f}")
print(f"EWMA VaR (asset 1):       {ewma_var_:.4f}")
print(f"Portfolio VaR:            {port_var:.4f}\n")

mvar = rm.calculate_marginal_var(prices.iloc[:, 1:], w, confidence_level)
cvar = rm.calculate_component_var(prices.iloc[:, 1:], w, confidence_level)

print("=== Marginal / Component VaR ===")
print("Marginal VaR:", mvar)
print("Component VaR:", cvar)
print("Sum Component VaR:", cvar.sum(), "\n")

mc_var = rm.monte_carlo(
    returns,
    weights,
    confidence_level,
    n_sim=50_000
)

print("=== Monte Carlo ===")
print(f"Monte Carlo VaR: {mc_var:.4f}\n")

# Rolling VaR (example: historical, 250-day window)
rolling_var = returns.rolling(250).apply(
    lambda x: rm.calculate_historical_var(x, confidence_level),
    raw=False
).dropna()

test_returns = returns.iloc[-len(rolling_var):, 0].values
test_var = rolling_var.iloc[:, 0].values

kupiec = rm.kupiec_test(test_returns, test_var, confidence_level)
breaches = rm.calculate_breaches(test_returns, window, confidence_level)
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
    confidence_level
)

print("=== Liquidity Adjustment ===")
print(f"Liquidity-adjusted VaR: {liq_var:.4f}\n")

print("Risk pipeline executed successfully.")