import numpy as np
import math
from scipy.stats import chi2
from risklib.var.historical import calculate_historical_var

def calculate_christoffersen_test(breach_sequence, days, returns, window):

    # Matrix of transitions
    n00 = n01 = n10 = n11 = 0
    
    for i in range(len(breach_sequence) - 1):
        if breach_sequence[i] == 0 and breach_sequence[i+1] == 0:
            n00 += 1
        elif breach_sequence[i] == 0 and breach_sequence[i+1] == 1:
            n01 += 1
        elif breach_sequence[i] == 1 and breach_sequence[i+1] == 0:
            n10 += 1
        elif breach_sequence[i] == 1 and breach_sequence[i+1] == 1:
            n11 += 1
    
    # Probabilities of transition
    pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    # likelihood ratio
    if pi_0 > 0 and pi_1 > 0 and pi > 0 and (1-pi_0) > 0 and (1-pi_1) > 0 and (1-pi) > 0:
        LR_ind = -2 * (
            math.log((1-pi)**(n00 + n10) * pi**(n01 + n11)) -
            math.log((1-pi_0)**n00 * pi_0**n01 * (1-pi_1)**n10 * pi_1**n11)
        )
        p_value = 1 - chi2.cdf(LR_ind, df=1)
        return (LR_ind, p_value)
    else:
        return (None, None)