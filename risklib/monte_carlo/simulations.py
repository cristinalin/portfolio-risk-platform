import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def monte_carlo(returns, window, n_simulations, confidence_level):
    VaR_MC_param, ES_MC_param, VaR_MC_boot, ES_MC_boot = [], [], [], []

    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window : i]

        mu, sigma = window_data.mean(), window_data.std()
        simulated_param_returns = np.random.normal(mu, sigma, n_simulations)
        var_param = -np.quantile(simulated_param_returns, 1 - confidence_level)
        es_param = -simulated_param_returns[simulated_param_returns < -var_param].mean()

        VaR_MC_param.append(var_param)
        ES_MC_param.append(es_param)

        simulated_boot_returns = np.random.choice(window_data, size=n_simulations, replace=True)
        var_boot = -np.quantile(simulated_boot_returns, 1 - confidence_level)
        es_boot = -simulated_boot_returns[simulated_boot_returns < -var_boot].mean()

        VaR_MC_boot.append(var_boot)
        ES_MC_boot.append(es_boot)
    
    dates = returns.index[window:]
    VaR_MC_param = pd.Series(VaR_MC_param, index=dates, name='VaR_MC_param')
    ES_MC_param = pd.Series(ES_MC_param, index=dates, name='ES_MC_param')
    VaR_MC_boot = pd.Series(VaR_MC_boot, index=dates, name='VaR_MC_boot')
    ES_MC_boot = pd.Series(ES_MC_boot, index=dates, name='ES_MC_boot')
    
    all_results = pd.concat([VaR_MC_param, ES_MC_param, VaR_MC_boot, ES_MC_boot], axis=1)

    return all_results

def montecarlo_histogram(returns, window, n_simulations, confidence_level, all_results):
    latest_sim_boot = np.random.choice(returns.iloc[-window:], 
                                   size=n_simulations, replace=True)
    latest_VaR_param = all_results['VaR_MC_param'].iloc[-1]
    latest_VaR_boot = all_results['VaR_MC_boot'].iloc[-1]
    plt.figure(figsize=(10, 6))

    plt.hist(latest_sim_boot, bins=50, alpha=0.7, color='grey', 
            label='Bootstrap Simulated Returns (Latest Window)')

    plt.axvline(-latest_VaR_param, color='red', linestyle='--', linewidth=2,
                label=f'Parametric VaR ({confidence_level*100:.0f}%)')
    plt.axvline(-latest_VaR_boot, color='blue', linestyle='-', linewidth=2,
                label=f'Bootstrap VaR ({confidence_level*100:.0f}%)')

    plt.title('Monte Carlo Simulated 1-Day Portfolio Returns (Distribution)')
    plt.xlabel('Portfolio Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.savefig('monte_carlo_var_histogram_final.png')
    plt.close()

def montecarlo_series(all_results):
    all_results.to_csv('monte_carlo_var_series.csv', header=True)
    mc_series_df = pd.read_csv("monte_carlo_var_series.csv", index_col=0, parse_dates=True)
    
    plt.figure(figsize=(14, 7))

    plt.plot(mc_series_df.index, mc_series_df['VaR_MC_param'], 
            label='Parametric VaR (MC)', color='red', linestyle='--')
    plt.plot(mc_series_df.index, mc_series_df['VaR_MC_boot'], 
            label='Bootstrap VaR (MC)', color='blue', linestyle='-')


    plt.plot(mc_series_df.index, mc_series_df['ES_MC_param'], 
            label='Parametric ES (MC)', color='red', linestyle=':', alpha=0.7)
    plt.plot(mc_series_df.index, mc_series_df['ES_MC_boot'], 
            label='Bootstrap ES (MC)', color='blue', linestyle='--', alpha=0.7)

    plt.title('Rolling 1-Day Monte Carlo VaR and ES (99% Confidence)')
    plt.xlabel('Date')
    plt.ylabel('Risk Measure Value (% Portfolio Loss)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('monte_carlo_var_timeseries.png')
    plt.close()
