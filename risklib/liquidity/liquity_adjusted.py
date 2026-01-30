import numpy as np
import pandas as pd


def liquidity_adjusted(tickers, weights, total_value, bids, asks, var):
    results = []

    for ticker in tickers:
        w = weights[ticker]
        position_value = total_value * w
        
        bid = bids[ticker]
        ask = asks[ticker]
        
        # Cálculos
        mid = (bid + ask) / 2
        bidAsk_pct = (ask - bid) / mid
        adv = position_value
        ttl = position_value / adv
        lac = 0.5 * bidAsk_pct * position_value
        
        results.append({
            'Ticker': ticker,
            'Weight': w,
            'Position Value': position_value,
            'Bid': bid,
            'Ask': ask,
            'Spread_%': bidAsk_pct * 100,
            'TTL': ttl,
            'LAC': lac
        })
    df = pd.DataFrame(results)
    lac_total = df['LAC'].sum()

    lac_pct = lac_total / total_value
    var_adjusted = var + lac_pct # the input VaR can be changed to ES as well
    return var_adjusted
