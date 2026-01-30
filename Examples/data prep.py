import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path

# importar os dados
# calcular o portfolio variance e os returns do portfolio para dar como input
# plot graph

def main():
    data = pd.read_excel(io = '/Users/cristina_yj/Desktop/LIS/Risk/projects/all/Data_asset daily prices (2).xlsx')

returns_win = returns.iloc[-window:]
tickers = returns_win.columns
n_assets = len(tickers)

weights = pd.Series(1.0 / n_assets, index=tickers, name="weight")

if LR_uc is not None and p_uc is not None:
        if p_uc > 0.05:
            print(" Teste de Kupiec: PASSA (cobertura incondicional adequada)")
        else:
            print(" Teste de Kupiec: NAO PASSA (cobertura incondicional inadequada)")
    else:
        print(f" Teste de Kupiec: {status_kupiec}")
    
    if LR_ind is not None and p_ind is not None:
        if p_ind > 0.05:
            print(" Teste de Christoffersen: PASSA (breaches independentes)")
        else:
            print(" Teste de Christoffersen: NAO PASSA (breaches dependentes)")
    else:
        print(f" Teste de Christoffersen: {status_chris}")
    
    print("=" * 70)