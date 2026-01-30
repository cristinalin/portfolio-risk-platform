import math
from scipy.stats import chi2

def calculate_kupiec_test(breaches, days, expected_breaches_rate, confidence_level):
    expected_breaches_rate = (1 - confidence_level)
    expected_breaches = days * expected_breaches_rate
    actual_breach_rate = breaches / days
    
    if breaches == 0 or breaches == days:
        return (None, None, "It is not possible to calculate")
    
    L_O = breaches * math.log(actual_breach_rate) + (days - breaches) * math.log(1 - actual_breach_rate)
    L_A = breaches * math.log(expected_breaches_rate) + (days - breaches) * math.log(1 - expected_breaches_rate)
    LR_uc = -2 * (L_A - L_O)
    p_value = 1 - chi2.cdf(LR_uc, df=1)
    
    return (LR_uc, p_value, "OK")