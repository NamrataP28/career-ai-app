import numpy as np

def percentile_rank(user_score):
    market_distribution = np.random.normal(60, 15, 1000)
    percentile = sum(market_distribution < user_score) / len(market_distribution) * 100
    return round(percentile,2)
