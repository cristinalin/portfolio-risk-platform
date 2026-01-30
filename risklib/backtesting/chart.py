import numpy as np
import matplotlib.pyplot as plt

def plot_breach_chart(portfolio_returns, historical_vars, breach_sequence, confidence_level):
    """Cria o grafico de breaches"""
    # Preparar dados válidos (sem NaN)
    valid_indices = [i for i in range(len(historical_vars)) if not np.isnan(historical_vars[i])]
    
    dates = range(len(valid_indices))
    returns_valid = [portfolio_returns.iloc[i] for i in valid_indices]
    var_valid = [historical_vars[i] for i in valid_indices]
    
    # Identificar breaches
    breach_dates = [i for i in range(len(breach_sequence)) if breach_sequence[i] == 1]
    breach_returns = [returns_valid[i] for i in breach_dates]
    
    # Criar gráfico
    plt.figure(figsize=(14, 7))
    plt.plot(dates, returns_valid, label='Portfolio Returns', color='blue', linewidth=0.8, alpha=0.7)
    plt.plot(dates, var_valid, label=f'VaR ({confidence_level*100:.0f}%)', color='green', linewidth=1.2)
    plt.scatter(breach_dates, breach_returns, color='red', s=30, label='Breaches', zorder=5)
    
    plt.xlabel('Dias', fontsize=12)
    plt.ylabel('Retorno', fontsize=12)
    plt.title(f'VaR Backtesting - Breaches Visualization (Confidence Level: {confidence_level*100:.0f}%)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()