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
single_name_returns = prices.iloc[:, 1].pct_change(periods=-1)
asset_returns = prices.iloc[:, 1:].pct_change().dropna()

confidence_level = 0.99
# alpha = 0.01 # significance level (confidence_level = 1 - alpha)
lambda_param = 0.97
window = 1000
hist_var_company_level = rm.calculate_historical_var(single_name_returns, confidence_level)
ewma_var_ = rm.calculate_ewma_var(returns, lambda_param, confidence_level)
port_hvar = rm.calculate_historical_var(returns, confidence_level)
port_gvar = rm.calculate_gaussian_var(returns, confidence_level)

print('=== VaR ===')
print(f'Historical VaR (asset 1): {abs(hist_var_company_level):.4f}')
print(f'EWMA VaR:                 {abs(ewma_var_):.4f}')
print(f'Portfolio HVaR:           {abs(port_hvar):.4f}')
print(f'Portfolio Gaussian VaR:   {port_gvar:.4f}\n')

mvar = rm.calculate_marginal_var(asset_returns, w, confidence_level)
cvar = rm.calculate_component_var(asset_returns, w, confidence_level)

print('=== Marginal / Component VaR ===')
print('Marginal VaR:', mvar)
print('Component VaR:', cvar)
print(f'Sum Component VaR: {cvar.sum():.4f}\n')

n_simulations = 10000
mc = rm.monte_carlo(returns, window, n_simulations, confidence_level)
mc_var = rm.monte_carlo_var(mc, 'VaR', 'boot', confidence_level)

print('=== Monte Carlo ===')
print(f'Monte Carlo VaR: {mc_var:.4f}\n')

# Rolling VaR (example: historical, 250-day window)
window = 250
rolling_var = returns[::-1].rolling(window).apply(
    lambda x: rm.calculate_historical_var(x, confidence_level),
    raw=False
).dropna()

test_returns = returns[::-1][:len(rolling_var)]
test_returns = test_returns[::-1]
test_var = rolling_var.values

breach_sequence, breaches = rm.calculate_breaches(test_returns, window, confidence_level)
kupiec = rm.kupiec_test(breaches, window, confidence_level)
christoffersen = rm.christoffersen_test(breach_sequence)
expected_breaches = np.round((1-confidence_level)*window, 0)

print('=== Backtesting ===')
print('Kupiec test:', kupiec)
print('Christoffersen test:', christoffersen)
print('Number of breaches:', breaches)
print('Expected number of breaches:', expected_breaches, '\n')

eigvals, eigvecs, evr, scores, loadings, top3 = rm.pca(asset_returns)

bsequence, enb = rm.calculate_breaches(evr, window, confidence_level)
c1 = rm.variance_concentration(evr, k=1)
dr = rm.diversification_ratio(asset_returns, w)

print('=== PCA & Diversification ===')
print(f'Variance explained by PC1: {c1:.2%}')
print(f'Top 3 assets that most explain the PC1: {top3}')
print(f'Effective number of bets:  {enb:.2f}')
print(f'Diversification ratio:     {dr:.2f}\n')

print('Risk pipeline executed successfully.')